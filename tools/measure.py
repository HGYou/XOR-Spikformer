#!/usr/bin/env python
"""Revision measurements for one checkpoint, from a single validation pass.

Writes <out>_layers.csv, <out>_fusion_sites.csv and <out>_summary.json:
residual bit-width and value set per fusion site, activation memory, the Eq. (7)
spike-operation count and NormComp, op_total_scheduled, firing rate and overlap.

Usage:
  python measure.py --dir <dir containing model.py> --cfg <args.yaml> --ckpt <checkpoint>
                    --data <dataset root> --out <prefix> [--gpu 0] [--no-amp]

Forward only, eval mode; weights are never updated.
"""
import argparse, csv, json, math, os, sys
from collections import OrderedDict

import torch
import torch.nn as nn
import yaml


@torch.no_grad()
def _choose_frac_bits(amax, signed, max_frac_bits=30):
    """Largest fractional bit count that cannot overflow: F = floor(log2(max_int / amax))."""
    amax = torch.clamp(amax, min=torch.tensor(1e-12, device=amax.device, dtype=amax.dtype))
    max_int = 127.0 if signed else 255.0
    f = torch.floor(torch.log2(torch.tensor(max_int, device=amax.device, dtype=amax.dtype) / amax))
    return torch.clamp(f, min=0, max=max_frac_bits).to(torch.int64)


@torch.no_grad()
def fixed8_pow2(x, signed=True, per_channel=False, ch_axis=0):
    """step = 2^-F, q = clamp(round(x/step)), dq = q*step. F comes from amax."""
    x_fp = x.detach()
    if per_channel:
        dims = tuple(d for d in range(x_fp.ndim) if d != ch_axis)
        F = _choose_frac_bits(x_fp.abs().amax(dim=dims, keepdim=True), signed)
    else:
        F = _choose_frac_bits(x_fp.abs().amax(), signed)
    step = torch.pow(torch.tensor(2.0, device=x_fp.device, dtype=x_fp.dtype), -F.to(x_fp.dtype))
    q = torch.round(x_fp / step)
    q = torch.clamp(q, -128, 127).to(torch.int8) if signed else torch.clamp(q, 0, 255).to(torch.uint8)
    return q.to(x_fp.dtype) * step, F


@torch.no_grad()
def quantize_weights(model, signed=True, per_channel=False):
    """Quantize-dequantize every Conv/Linear weight to fixed-8 power-of-two, in place."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            w_qdq, _ = fixed8_pow2(m.weight.data, signed=signed,
                                   per_channel=per_channel, ch_axis=0)
            m.weight.data.copy_(w_qdq)
            n += 1
    return n


@torch.no_grad()
def quantize_input(x, frac, signed=True):
    """Round-trip the input image through fixed point Q(8-frac).frac. frac=4 is Q4.4."""
    step = 2.0 ** (-frac)
    q = torch.round(x / step)
    q = torch.clamp(q, -128, 127) if signed else torch.clamp(q, 0, 255)
    return q * step


def _planes_for_terms(t):
    """Bit planes needed to carry an integer in {0..t}."""
    n = 1
    while (1 << n) - 1 < t:
        n += 1
    return n


def residual_plane_schedule(fusion, depths):
    """name -> bit planes the accelerator streams into that layer.

    Only ADD propagates a multi-bit residual. XOR, OR and clipped ADD keep it at
    one bit, so every spiking layer is a single plane in those builds. The first
    convolution is the analog image, streamed as eight pre-separated Q4.4 planes
    in every build.
    """
    planes = {"patch_embed.proj_conv": 8}
    if fusion != "add":
        return planes
    for i in range(depths):
        for n in ("q_linear", "k_linear", "v_linear", "q_conv", "k_conv", "v_conv"):
            planes["block.%d.attn.%s" % (i, n)] = _planes_for_terms(2 * i + 2)
        for n in ("fc1_linear", "fc1_conv"):
            planes["block.%d.mlp.%s" % (i, n)] = _planes_for_terms(2 * i + 3)
    planes["head"] = _planes_for_terms(2 * depths + 2)
    return planes


# ---------------------------------------------------------------------
def load_cfg(path, experiment=None):
    """Read args.yaml.

    results/args/<dataset>.yaml stores the shared settings under `common` and the
    per-experiment overrides under `experiments`; a run is `common | experiments[experiment]`. A raw args.yaml
    written by the training script is already flat and is returned unchanged.
    """
    raw = yaml.safe_load(open(path))
    if "common" not in raw:
        return raw
    if experiment is None:
        sys.exit(f"{path} holds several experiments; pass --experiment <one of "
                 f"{sorted(raw.get('experiments', {}))}>")
    if experiment not in raw.get("experiments", {}):
        sys.exit(f"experiment {experiment!r} not in {path}; available: {sorted(raw.get('experiments', {}))}")
    return {**raw["common"], **raw["experiments"][experiment]}


def build_model(mod, cfg, fusion):
    """Inject the architecture arguments from args.yaml; fall back to model.py defaults."""
    kw = dict(fusion=fusion)
    if cfg.get("dim") is not None:            # CIFAR family
        kw.update(img_size_h=cfg["img_size"], img_size_w=cfg["img_size"],
                  patch_size=cfg["patch_size"], embed_dims=cfg["dim"],
                  num_heads=cfg["num_heads"], mlp_ratios=cfg["mlp_ratio"],
                  in_channels=3, num_classes=cfg["num_classes"], qkv_bias=False,
                  depths=cfg["layer"], sr_ratios=1, T=cfg["time_step"])
    return mod.spikformer(**kw)


def load_weights(model, ckpt, allow_partial=False):
    """Load a checkpoint. Strict by default: a key mismatch means the checkpoint
    does not belong to this architecture, and measuring it would report numbers
    for a partly random model. --allow-partial downgrades the failure to a warning."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        sd = sd.get("state_dict", sd)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    if not allow_partial:
        model.load_state_dict(sd, strict=True)
        return model
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [warn] --allow-partial: missing={len(missing)} "
              f"unexpected={len(unexpected)}")
        if missing:
            print(f"         missing e.g. {missing[:3]}")
    return model


def build_val_loader(mod_dir, cfg, max_batches):
    """Rebuild exactly the validation pipeline that run used."""
    sys.path.insert(0, mod_dir)
    from timm.data import create_dataset
    # Use the same loader the run used (CIFAR = vendored local, Tiny = timm)
    try:
        from loader import create_loader          # CIFAR family
        loader_src = "local loader.py (aa_snn)"
    except ImportError:
        from timm.data import create_loader       # Tiny family
        loader_src = "timm.data"
    ds = create_dataset(cfg["dataset"], root=cfg["data_dir"], split=cfg["val_split"],
                        is_training=False, batch_size=cfg["batch_size"])
    dc = dict(input_size=(3, cfg["img_size"], cfg["img_size"]),
              interpolation=cfg.get("interpolation", "bicubic"),
              mean=tuple(cfg["mean"]), std=tuple(cfg["std"]),
              crop_pct=cfg.get("crop_pct", 1.0))
    loader = create_loader(ds, input_size=dc["input_size"],
                           batch_size=cfg.get("val_batch_size") or cfg["batch_size"],
                           is_training=False, use_prefetcher=not cfg.get("no_prefetcher", False),
                           interpolation=dc["interpolation"], mean=dc["mean"], std=dc["std"],
                           num_workers=cfg.get("workers", 4), crop_pct=dc["crop_pct"],
                           pin_memory=cfg.get("pin_mem", False))
    return loader, loader_src


# ---------------------------------------------------------------------
class Probe:
    """Accumulate conv/linear input statistics, SSA spike counts and fusion-site statistics."""

    def __init__(self, model, fusion="xor", depths=None, frac_bits=4):
        self.model = model
        self.mode = fusion
        self.frac_bits = frac_bits
        self.lin = OrderedDict()      # name -> dict(nz, total, c_out, kind)
        self.fusion = OrderedDict()   # name -> dict(a_nz,b_nz,ab_nz,total,out_min,out_max,...)
        self.ssa = OrderedDict()      # name -> dict(q_nz, k_nz, d_head)
        self.first_conv_name = 'patch_embed.proj_conv'
        if depths is None:
            depths = len(getattr(model, "block", []))
        self.depths = depths
        self.planes = residual_plane_schedule(fusion, depths)
        self.head_ops = 0.0           # Eq. (7): C_out * active input events
        self.head_bitplane_ops = 0.0  # the same layer charged by bit planes
        self.head_dense_ops = 0.0
        self.head_est_ms_ops = 0.0    # MS only: a signed Q(F).F bit-plane estimate
        self.handles = []

    def plane_count(self, name):
        """Bit planes the firmware streams into this layer. Topology, not data."""
        return self.planes.get(name, 1)

    def fusion_sites_expected(self):
        """|R| = 2L + 1: the SPS RPE residual plus two per encoder block."""
        return range(2 * self.depths + 1)

    def _lin_hook(self, name, c_out, kind, mod_ref=None):
        """Bit-serial accelerator convention: decompose the input into 8 bit-planes
        and count per plane.

        In XOR/OR models activations are {0,1}, so only plane 0 is set and the count
        matches the binary one; in ADD models activations are multi-bit, so popcount(v)
        accumulations are charged. Only the first conv sees an analog input, which is
        quantized as 2's-complement signed with frac_bits=4 (Q4.4).
        """
        def h(mod, inp, out):
            x = inp[0].detach()
            d = self.lin.setdefault(name, dict(nz=0, total=0, n_act=0, ops=0.0,
                                               dense_ops=0.0, plane_bits=0.0,
                                               planes=self.plane_count(name),
                                               c_out=c_out, kind=kind))
            d["nz"] += int(torch.count_nonzero(x))
            d["total"] += x.numel()

            first = (name == self.first_conv_name)
            frac, signed = (4, True) if first else (0, False)   # Q4.4 deployed format
            scale = float(1 << frac)
            q = torch.round(x.to(torch.float32) * scale)
            if signed:
                q = (torch.clamp(q, -128, 127).to(torch.int16) & 0xFF).to(torch.int32)
            else:
                q = torch.clamp(q, 0, 255).to(torch.int32)

            ops = 0.0
            for b in range(8):
                plane = ((q >> b) & 1).to(torch.float32)
                if int(plane.sum()) == 0:
                    continue
                if kind == "conv":
                    ones = torch.ones_like(mod.weight, dtype=torch.float32)
                    y = torch.nn.functional.conv2d(plane, ones, None, mod.stride,
                                                   mod.padding, mod.dilation, mod.groups)
                elif kind == "conv1d":
                    ones = torch.ones_like(mod.weight, dtype=torch.float32)
                    y = torch.nn.functional.conv1d(plane, ones, None, mod.stride,
                                                   mod.padding, mod.dilation, mod.groups)
                else:
                    ones = torch.ones((mod.out_features, mod.in_features),
                                      dtype=torch.float32, device=x.device)
                    y = torch.nn.functional.linear(plane, ones, None)
                ops += float(y.sum(dtype=torch.float64))
            d["ops"] += ops
            d["n_act"] += int(ops / c_out) if c_out else 0

            planes = self.plane_count(name)
            ones_in = torch.ones_like(x, dtype=torch.float32)
            if kind == "conv":
                w1 = torch.ones_like(mod.weight, dtype=torch.float32)
                yd = torch.nn.functional.conv2d(ones_in, w1, None, mod.stride,
                                                mod.padding, mod.dilation, mod.groups)
            elif kind == "conv1d":
                w1 = torch.ones_like(mod.weight, dtype=torch.float32)
                yd = torch.nn.functional.conv1d(ones_in, w1, None, mod.stride,
                                                mod.padding, mod.dilation, mod.groups)
            else:
                w1 = torch.ones((mod.out_features, mod.in_features),
                                dtype=torch.float32, device=x.device)
                yd = torch.nn.functional.linear(ones_in, w1, None)
            d["dense_ops"] += float(yd.sum(dtype=torch.float64)) * max(1, planes)

            d["plane_bits"] += float(x.numel()) * planes
            d["planes"] = planes
        return h

    # -- ||.||_0 of the q_lif / k_lif outputs (the SSA term of Eq. (7)) --------------
    def _ssa_hook(self, blk_name, which, d_head):
        def h(mod, inp, out):
            d = self.ssa.setdefault(blk_name, dict(q_nz=0, k_nz=0, q_all=0, k_all=0,
                                                   d_head=d_head))
            o = out.detach()
            d[f"{which}_nz"] += int(torch.count_nonzero(o))
            d[f"{which}_all"] += o.numel()   # dense: with no zero-skip every position is processed
        return h

    # -- fusion site: operand binarity / overlap / output bit-width -------
    def record_fusion(self, name, a, b, out):
        """Per-site operand check, overlap, and the observed output range.

        Both ends of the range are kept: XOR, OR and ADD produce non-negative
        values, but the membrane shortcut carries a signed continuous potential
        whose minimum is what sets the width of a two's-complement word.
        """
        d = self.fusion.setdefault(name, dict(a_nz=0, b_nz=0, ab_nz=0, total=0,
                                              out_min=float("inf"),
                                              out_max=float("-inf"),
                                              out_vals=set(),
                                              a_binary=True, b_binary=True))
        a = a.detach(); b = b.detach(); out = out.detach()
        d["a_nz"] += int(torch.count_nonzero(a))
        d["b_nz"] += int(torch.count_nonzero(b))
        d["ab_nz"] += int(torch.count_nonzero(a * b))
        d["total"] += a.numel()
        d["out_min"] = min(d["out_min"], float(out.min()))
        d["out_max"] = max(d["out_max"], float(out.max()))
        if len(d["out_vals"]) < 64:
            d["out_vals"].update(float(v) for v in torch.unique(out)[:64].tolist())
        for t, key in ((a, "a_binary"), (b, "b_binary")):
            if d[key] and bool(((t != 0) & (t != 1)).any()):
                d[key] = False

    def attach(self):
        m = self.model
        pe = getattr(m, "patch_embed")
        for n, mod in pe.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                self.handles.append(mod.register_forward_hook(
                    self._lin_hook(f"patch_embed.{n}", mod.out_channels, "conv")))
        for i, blk in enumerate(getattr(m, "block")):
            for n, mod in blk.named_modules():
                if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
                    if isinstance(mod, torch.nn.Linear):
                        c_out, kind = mod.out_features, "linear"
                    elif isinstance(mod, torch.nn.Conv1d):
                        c_out, kind = mod.out_channels, "conv1d"
                    else:
                        c_out, kind = mod.out_channels, "conv"
                    self.handles.append(mod.register_forward_hook(
                        self._lin_hook(f"block.{i}.{n}", c_out, kind)))
            attn = blk.attn
            d_head = attn.dim // attn.num_heads
            self.handles.append(attn.q_lif.register_forward_hook(
                self._ssa_hook(f"block.{i}", "q", d_head)))
            self.handles.append(attn.k_lif.register_forward_hook(
                self._ssa_hook(f"block.{i}", "k", d_head)))
        nc = m.head.out_features if isinstance(m.head, torch.nn.Linear) else 0
        head_planes = self.plane_count("head")

        def _head_hook(mod, inp, out, nc=nc, planes=head_planes):
            t = out[0] if isinstance(out, tuple) else out
            x = t.detach()
            self.head_ops += float(torch.count_nonzero(x)) * nc
            if self.mode == "ms":
                scale = 1 << self.frac_bits
                q = torch.round(x.to(torch.float32) * scale).to(torch.int64)
                w = _signed_bits(int(q.min()), int(q.max()))
                self.head_est_ms_ops += float(x.numel()) * w * nc
                return
            q = torch.clamp(torch.round(x.to(torch.float32)), 0, 255).to(torch.int32)
            ops = 0.0
            for b in range(8):
                ops += float(((q >> b) & 1).sum(dtype=torch.float64))
            self.head_bitplane_ops += ops * nc
            self.head_dense_ops += float(x.numel()) * planes * nc
        self.handles.append(getattr(m, 'block')[-1].register_forward_hook(_head_hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # -- aggregation ------------------------------------------------
    def op_total(self):
        """Eq. (7): the useful work, counting nonzero events only."""
        conv_lin = sum(d["ops"] for d in self.lin.values()) + self.head_ops
        ssa = sum(d["d_head"] * (d["q_nz"] + d["k_nz"]) for d in self.ssa.values())
        return conv_lin, ssa, conv_lin + ssa

    def op_total_scheduled(self):
        """The work the accelerator issues, from the fixed schedule.

        There is no zero-skip and no data-dependent control in this design, so a
        zero activation still costs a cycle and the issued work is set by the model
        topology and the firmware's bit-plane schedule alone. It is therefore the
        same for an all-zero input and for a real image, and it is the number a
        measured latency divides into. Its ratio to the Eq. (7) count is the
        sparsity headroom -- the upper bound on what a zero-skip datapath could
        recover. Returns (None, None, None) for the membrane shortcut, which has
        no hardware build."""
        if self.mode == "ms":
            return None, None, None
        conv_lin = sum(d["dense_ops"] for d in self.lin.values()) + self.head_dense_ops
        ssa = sum(d["d_head"] * (d["q_all"] + d["k_all"]) for d in self.ssa.values())
        return conv_lin, ssa, conv_lin + ssa

    # kept as an alias: the same fixed-schedule count under the older name
    op_total_dense = op_total_scheduled

    def firing(self, spiking_only=True):
        """Mean firing rate: the fraction of nonzero inputs over the hooked layers.

        spiking_only=False keeps every instrumented computation layer, the first
        convolution included, and is written as mean_firing_rate_all_layers; that is
        the scope the response-letter tables use, the same one for every experiment.
        spiking_only=True drops the first convolution, whose input is the analog
        image, and is written as mean_firing_rate, an additional diagnostic. The
        first convolution is charged in the Eq. (7) count either way, as a Q4.4
        input."""
        items = [(n, d) for n, d in self.lin.items()
                 if spiking_only is False or n != self.first_conv_name]
        nz = sum(d["nz"] for _, d in items)
        tot = sum(d["total"] for _, d in items)
        return nz / tot if tot else 0.0


def patch_fusion_for_probe(mod, probe):
    """Wrap model.fuse to collect per-site statistics; sites are named in call order."""
    orig = mod.fuse
    counter = {"i": 0}

    def wrapped(x, z, mode):
        out = orig(x, z, mode)
        probe.record_fusion(f"site_{counter['i']:02d}", x, z, out)
        counter["i"] += 1
        return out

    mod.fuse = wrapped
    return orig, counter


def bits_for(maxval):
    """Unsigned bits required for the observed maximum (the form of Eq. (3))."""
    import math
    n = int(round(maxval))
    return 1 if n <= 1 else int(math.floor(math.log2(n))) + 1


def _signed_bits(lo, hi):
    """Smallest two's-complement width holding every integer in [lo, hi]."""
    n = 1
    while lo < -(1 << (n - 1)) or hi > (1 << (n - 1)) - 1:
        n += 1
    return n


def fixed_point_bits(out_min, out_max, frac_bits=4):
    """Width of the fixed-point word that holds an observed range.

    Non-negative ranges are read as unsigned integers, the form of Eq. (3) that
    ADD follows. A range with a negative end -- the membrane shortcut -- is read
    as a signed two's-complement word at the accelerator's existing frac_bits
    fractional bits. This is an estimate from an observed range under an existing
    precision convention, not a measured minimum precision and not the width of
    any implemented datapath.
    """
    if out_min >= 0:
        return bits_for(out_max), "unsigned", 0
    scale = 1 << frac_bits
    lo = int(math.floor(out_min * scale))
    hi = int(math.ceil(out_max * scale))
    return _signed_bits(lo, hi), "signed", frac_bits


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--experiment", default=None,
                    help="experiment name, required when --cfg is a results/args/<dataset>.yaml "
                         "with a common/experiments structure")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=None,
                    help="dataset root; overrides data_dir from the config, which is "
                         "a <data_root> placeholder in the released files")
    ap.add_argument("--fusion", default=None, help="defaults to the fusion recorded in args.yaml")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = the whole validation set")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true",
                    help="evaluate in fp32 instead of following args.yaml's amp setting; "
                         "top-1 will then not match the reported value")
    ap.add_argument("--allow-partial", action="store_true",
                    help="load the checkpoint with strict=False and warn instead of "
                         "failing on a key mismatch")
    ap.add_argument("--quant", action="store_true",
                    help="weight-only PTQ: fixed-8 pow-2 weights + fixed-point input "
                         "round-trip. BatchNorm folding is not performed here, so this "
                         "is a software-side approximation of the deployed format, not "
                         "the full hardware quantization flow")
    ap.add_argument("--frac-bits", type=int, default=4,
                    help="fractional bits of the accelerator's activation convention, "
                         "used to size a signed residual (the membrane shortcut)")
    ap.add_argument("--in-frac", type=int, default=4,
                    help="fractional bits of the input fixed-point format; 4 = Q4.4 (hardware)")
    ap.add_argument("--w-per-channel", action="store_true",
                    help="quantize weights per output channel (default: per tensor)")
    a = ap.parse_args()

    out_dir = os.path.dirname(os.path.abspath(a.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cfg = load_cfg(a.cfg, a.experiment)
    if a.data_dir:
        cfg["data_dir"] = a.data_dir
    if not cfg.get("data_dir") or str(cfg["data_dir"]).startswith("<"):
        sys.exit(f"data_dir is {cfg.get('data_dir')!r}; pass --data-dir <path>.")
    fusion = a.fusion or cfg.get("fusion")
    if fusion is None:
        sys.exit("fusion not found in args.yaml; pass it explicitly with --fusion.")

    sys.path.insert(0, a.dir)
    import model as M
    from spikingjelly.clock_driven import functional

    torch.cuda.set_device(a.gpu)
    print(f"== {os.path.basename(a.out)}  fusion={fusion}  ckpt={os.path.basename(a.ckpt)}")

    model = build_model(M, cfg, fusion)
    load_weights(model, a.ckpt, allow_partial=a.allow_partial)
    model = model.cuda().eval()

    if a.quant:
        nq = quantize_weights(model, signed=True, per_channel=a.w_per_channel)
        print(f"   weight-only PTQ (no BN folding): fixed8 pow-2 weights on {nq} modules "
              f"({'per-channel' if a.w_per_channel else 'per-tensor'}), "
              f"input Q{8-a.in_frac}.{a.in_frac}")

    loader, loader_src = build_val_loader(a.dir, cfg, a.max_batches)
    print(f"   val loader: {loader_src}   batches={len(loader)}")

    depths = cfg.get("layer") or len(getattr(model, "block", []))
    probe = Probe(model, fusion=fusion, depths=depths, frac_bits=a.frac_bits)
    probe.attach()
    sched = sorted(set(probe.planes.values()))
    print(f"   fixed schedule: L={depths}, {len(probe.fusion_sites_expected())} residual "
          f"sites expected, bit planes in use {sched}")
    orig_fuse, counter = patch_fusion_for_probe(M, probe)

    amp = (not a.no_amp) and bool(cfg.get("amp", True))
    print(f"   eval precision: {'amp(fp16)' if amp else 'fp32'}")
    correct = seen = 0
    functional.reset_net(model)
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if a.max_batches and i >= a.max_batches:
                break
            counter["i"] = 0
            x = x.cuda(non_blocking=True); y = y.cuda(non_blocking=True)
            if a.quant:
                x = quantize_input(x, a.in_frac, signed=True)
            with torch.autocast("cuda", enabled=amp):
                out = model(x)
            functional.reset_net(model)   # after the forward, as in train.py's validate()
            correct += int((out.argmax(1) == y).sum()); seen += y.numel()
    M.fuse = orig_fuse
    probe.detach()

    conv_lin, ssa, op_total = probe.op_total()
    d_conv_lin, d_ssa, op_total_sched = probe.op_total_scheduled()
    top1 = 100.0 * correct / seen if seen else float("nan")

    n_sites = len(probe.fusion)
    n_expect = 2 * probe.depths + 1
    if n_sites != n_expect:
        print(f"   [warn] {n_sites} residual sites recorded, expected 2L+1 = {n_expect}")

    # ---- per-layer CSV (C-4 firing rate + C-3 OP) --------------------
    with open(f"{a.out}_layers.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "kind", "c_out", "nz", "n_act", "total", "nz_rate", "sparsity",
                    "ops", "scheduled_ops", "scheduled_planes", "plane_bits",
                    "in_firing_rate_scope"])
        for n, d in probe.lin.items():
            r = d["nz"] / d["total"] if d["total"] else 0
            w.writerow([n, d["kind"], d["c_out"], d["nz"], d["n_act"], d["total"],
                        f"{r:.6f}", f"{1-r:.6f}", int(d["ops"]), int(d["dense_ops"]),
                        d["planes"], int(d["plane_bits"]),
                        int(n != probe.first_conv_name)])

    # ---- fusion site CSV (C-1 bit-width + overlap + C-2 footprint) ---
    with open(f"{a.out}_fusion_sites.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site", "a_binary", "b_binary", "a_rate", "b_rate", "overlap_rate",
                    "out_min", "out_max", "out_distinct", "format", "frac_bits",
                    "bits_required", "elems_per_pass", "bytes_1bit", "bytes_actual"])
        for n, d in probe.fusion.items():
            tot = d["total"] or 1
            bits, fmt, frac = fixed_point_bits(d["out_min"], d["out_max"], a.frac_bits)
            w.writerow([n, d["a_binary"], d["b_binary"],
                        f"{d['a_nz']/tot:.6f}", f"{d['b_nz']/tot:.6f}",
                        f"{d['ab_nz']/tot:.6f}",
                        f"{d['out_min']:.4f}", f"{d['out_max']:.4f}",
                        len(d["out_vals"]), fmt, frac, bits, tot,
                        f"{tot/8:.0f}", f"{tot*bits/8:.0f}"])

    # ---- summary JSON ------------------------------------------------
    site_bits = [fixed_point_bits(d["out_min"], d["out_max"], a.frac_bits)
                 for d in probe.fusion.values()]
    maxbits = max((b for b, _, _ in site_bits), default=1)
    allbin = all(d["a_binary"] and d["b_binary"] for d in probe.fusion.values())
    outbin = all(d["out_min"] >= 0 and d["out_max"] <= 1 for d in probe.fusion.values())
    signed = any(f == "signed" for _, f, _ in site_bits)
    ov_mean = (sum(d["ab_nz"] / (d["total"] or 1) for d in probe.fusion.values())
               / max(1, len(probe.fusion)))
    summary = dict(
        name=os.path.basename(a.out), fusion=fusion, checkpoint=a.ckpt,
        dataset=cfg.get("dataset"), val_split=cfg.get("val_split"),
        samples=seen, top1=round(top1, 4),
        precision=("amp_fp16" if amp else "fp32"),
        quantized=bool(a.quant),
        quant_scheme=(f"input Q{8-a.in_frac}.{a.in_frac} signed + fixed8 pow-2 weights "
                      f"({'per-channel' if a.w_per_channel else 'per-tensor'}), "
                      f"no BatchNorm folding"
                      if a.quant else None),
        depths=probe.depths,
        fusion_sites=len(probe.fusion),
        fusion_sites_expected=2 * probe.depths + 1,
        operands_all_binary=allbin,
        fusion_output_binary=outbin,
        residual_format=("signed" if signed else "unsigned"),
        frac_bits=(a.frac_bits if signed else 0),
        min_activation_value=min((d["out_min"] for d in probe.fusion.values()), default=0.0),
        max_activation_value=max((d["out_max"] for d in probe.fusion.values()), default=0.0),
        residual_bits_required=maxbits,
        mean_overlap_rate=(round(ov_mean, 8) if allbin else None),
        support_coactivation_rate=(round(ov_mean, 8)
                                   if (not allbin and probe.mode != "ms") else None),
        overlap_defined=allbin,
        # additional diagnostic: the same fraction with the analog input of the
        # first convolution dropped
        mean_firing_rate=round(probe.firing(), 8),
        firing_rate_excludes=probe.first_conv_name,
        firing_rate_definition=("fraction of nonzero input activations across all "
                                "spiking layers, excluding the analog input to the "
                                "first convolution"),
        # the scope the response-letter tables use, the same one for every experiment
        mean_firing_rate_all_layers=round(probe.firing(spiking_only=False), 8),
        # Eq. (7): intrinsic, input dependent
        op_conv_lin=conv_lin, op_ssa=ssa, op_total=op_total,
        # the classifier term, under both readings. op_total carries op_head.
        op_head=probe.head_ops,
        op_head_bitplane=(probe.head_bitplane_ops if probe.mode != "ms" else None),
        op_total_head_bitplane=((op_total - probe.head_ops + probe.head_bitplane_ops)
                                if probe.mode != "ms" else None),
        scheduled_bit_planes={k: v for k, v in sorted(probe.planes.items())},
        op_conv_lin_scheduled=d_conv_lin, op_ssa_scheduled=d_ssa,
        op_total_scheduled=op_total_sched,
        op_total_dense=op_total_sched,     # alias, kept for output compatibility
        sparsity_headroom=(op_total_sched / op_total
                           if (op_total and op_total_sched) else None),
        estimated_ms_classifier_ops=(probe.head_est_ms_ops
                                     if probe.mode == "ms" else None),
    )
    with open(f"{a.out}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"   top1={top1:.2f}  sites={len(probe.fusion)}/{2*probe.depths+1}  "
          f"operands_binary={allbin}  out_binary={outbin}  "
          f"range=[{summary['min_activation_value']:.2f}, "
          f"{summary['max_activation_value']:.2f}] -> {maxbits} bit "
          f"{summary['residual_format']}")
    print(f"   firing={summary['mean_firing_rate']:.6f}  "
          f"{'overlap' if allbin else 'operand_intersection'}={ov_mean:.6f}  "
          f"OP_total={op_total:,}  OP_scheduled="
          f"{op_total_sched if op_total_sched is None else format(int(op_total_sched), ',')}")
    print(f"   -> {a.out}_layers.csv / _fusion_sites.csv / _summary.json")


if __name__ == "__main__":
    main()
