#!/usr/bin/env python3
# Derived from pytorch-image-models (timm) (https://github.com/huggingface/pytorch-image-models),
# Apache License 2.0, Copyright 2019-2020 Ross Wightman. Modified for XOR-Spikformer:
# spiking model construction, the --fusion / --lam-ov / --ov-warmup-epochs
# arguments and the overlap-regularization loss term. See LICENSES/.
# This is a slightly modified version of timm's training script
""" Spikformer ImageNet Test Script (with spike(nonzero) counting)

Counts nonzero activations on:
  - Inputs to Conv/Linear layers (Conv2d/Conv1d/Linear)
  - Inputs to SSA matmuls (q, k, v after LIF)  <-- patched SSA.forward

Excludes:
  - The very first Conv in SPS: patch_embed.proj_conv
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import csv

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from spikingjelly.clock_driven import functional

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, convert_splitbn_model
from timm.utils import *
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

# IMPORTANT: avoid shadowing module name "model"
import model as spk_model

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

# PyTorch 2.6+ checkpoint safety (argparse.Namespace appears in timm checkpoints)
import torch.serialization
torch.serialization.add_safe_globals([argparse.Namespace])

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


# ---------------------------
# Spike/nonzero counter
# ---------------------------
class SpikeInputCounter:
    """
    Count nonzero elements in the INPUT activation of selected layers.
    For binary spikes (0/1), count_nonzero == number of ones.
    Also supports SSA matmul operand counting via SSA.attach_counter + patched SSA.forward.
    """
    def __init__(
        self,
        include_types=(nn.Conv2d, nn.Conv1d, nn.Linear),
        exclude_name_suffixes=("patch_embed.proj_conv",),  # exclude first conv only
    ):
        self.include_types = include_types
        self.exclude_name_suffixes = exclude_name_suffixes
        self.handles = []
        self.stats = OrderedDict()

    def _excluded(self, name: str) -> bool:
        return any(name.endswith(suf) for suf in self.exclude_name_suffixes)

    @torch.no_grad()
    def count(self, name: str, x: torch.Tensor):
        if x is None or (not torch.is_tensor(x)):
            return
        nz = x.count_nonzero().item()
        total = x.numel()
        if name not in self.stats:
            self.stats[name] = {
                "ones": 0,   # keep key name as "ones" (meaning nonzero count)
                "total": 0,
                "shape": tuple(x.shape),
                "dtype": str(x.dtype)
            }
        self.stats[name]["ones"] += int(nz)
        self.stats[name]["total"] += int(total)

    def _hook_fn(self, name: str):
        def hook(module, inputs, output):
            if not inputs:
                return
            x = inputs[0]
            if torch.is_tensor(x):
                self.count(name, x)
        return hook

    def register(self, model: nn.Module):
        base = model.module if hasattr(model, "module") else model

        for name, m in base.named_modules():
            # Conv/Linear inputs
            if isinstance(m, self.include_types) and not self._excluded(name):
                self.handles.append(m.register_forward_hook(self._hook_fn(name)))

            # SSA (or any module) can expose attach_counter(counter, prefix)
            if hasattr(m, "attach_counter") and callable(getattr(m, "attach_counter")):
                m.attach_counter(self, name)

        return self

    def reset(self):
        self.stats.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def summary(self, sort_by="rate"):
        rows = []
        for name, s in self.stats.items():
            ones, total = s["ones"], s["total"]
            rate = ones / (total + 1e-12)
            rows.append((name, ones, total, rate, s["shape"], s["dtype"]))
        if sort_by == "ones":
            rows.sort(key=lambda x: x[1], reverse=True)
        elif sort_by == "total":
            rows.sort(key=lambda x: x[2], reverse=True)
        else:
            rows.sort(key=lambda x: x[3], reverse=True)
        return rows


# ---------------------------
# Patch SSA to count matmul operands (q,k,v after LIF)
# (no need to modify model.py)
# ---------------------------
def _ssa_attach_counter(self, counter, prefix: str):
    self._spike_counter = counter
    self._spike_prefix = prefix

def _ssa_forward_with_count(self, x, res_attn):
    # Copy of SSA.forward, with 3 count() calls added
    T, B, C, H, W = x.shape
    x = x.flatten(3)
    T, B, C, N = x.shape
    x_for_qkv = x.flatten(0, 1)

    q_conv_out = self.q_conv(x_for_qkv)
    q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
    q_conv_out = self.q_lif(q_conv_out)

    k_conv_out = self.k_conv(x_for_qkv)
    k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
    k_conv_out = self.k_lif(k_conv_out)

    v_conv_out = self.v_conv(x_for_qkv)
    v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
    v_conv_out = self.v_lif(v_conv_out)

    # ---- COUNT matmul operands (these are "input activations" to matmul) ----
    cnt = getattr(self, "_spike_counter", None)
    pfx = getattr(self, "_spike_prefix", None)
    if (cnt is not None) and (pfx is not None):
        cnt.count(f"{pfx}.matmul1_k", k_conv_out)  # k^T @ v
        cnt.count(f"{pfx}.matmul1_v", v_conv_out)
        cnt.count(f"{pfx}.matmul2_q", q_conv_out)  # q @ (k^T@v)

    q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
    k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()
    v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

    x = k.transpose(-2, -1) @ v
    x = (q @ x) * self.scale

    x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
    x = self.attn_lif(x)
    x = x.flatten(0, 1)
    x = self.proj_conv(x)
    x = self.proj_bn(x).reshape(T, B, C, H, W).contiguous()
    x = self.proj_lif(x)
    return x, v

# Apply monkey patch if SSA exists
if hasattr(spk_model, "SSA"):
    setattr(spk_model.SSA, "attach_counter", _ssa_attach_counter)
    setattr(spk_model.SSA, "forward", _ssa_forward_with_count)


# ---------------------------
# ---------------------------
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset / Model parameters
parser.add_argument('-data-dir', '--data-dir', metavar='DIR', default='', help='path to the dataset root (required; set it here or in the config file)')
parser.add_argument('--dataset', '-d', metavar='NAME', default='torch/cifar10',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
parser.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
parser.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('-vb', '--val-batch-size', type=int, default=16, metavar='N',
                    help='input val batch size for training (default: 32)')

# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.0001,
                    help='weight decay (default: 0.0001)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')

# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
parser.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
parser.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
parser.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
parser.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)')
parser.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
parser.add_argument('--jsd', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
parser.add_argument('--remode', type=str, default='const',
                    help='Random erase mode (default: "const")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
parser.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
parser.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
parser.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
parser.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model EMA
parser.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 1)')
parser.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
parser.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
parser.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')

parser.add_argument('--fusion', type=str, default='xor', choices=['add', 'xor', 'or', 'or_clamp', 'ms'],
                    help="residual fusion operator: add = x+z (multi-bit), "
                         "xor = x+z-2xz (binary, paper Eq. (5)), "
                         "or = x+z-xz (binary, clipped ADD = min(1,x+z)). Default: xor")

# device selection for single GPU
parser.add_argument('--device', default='cuda:0', type=str,
                    help='device for single-process eval, e.g. cuda:1')


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.world_size = 1
    args.rank = 0
    # ensure output_dir attribute exists for all ranks (rank0 will fill)
    args.output_dir = ""

    if args.distributed:
        args.device = f'cuda:{args.local_rank}'
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info(
            f'Training in distributed mode with multiple processes, 1 GPU per process. '
            f'Process {args.rank}, total {args.world_size}.'
        )
    else:
        # make sure the "current cuda device" matches args.device (important for timm prefetcher)
        if args.device.startswith('cuda'):
            idx = int(args.device.split(':')[1]) if ':' in args.device else 0
            torch.cuda.set_device(idx)
        _logger.info(f'Training with a single process on 1 GPU. device={args.device}')

    assert args.rank >= 0

    # resolve AMP arguments
    use_amp = None
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32.")

    random_seed(args.seed, args.rank)

    net = create_model(
        'spikformer',
        pretrained=False,
        drop_rate=0.,
        drop_path_rate=0.2,
        drop_block_rate=None,
        fusion=args.fusion,
    )
    print("Creating model")
    _logger.info(f"residual fusion = {args.fusion}")
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    if args.num_classes is None:
        assert hasattr(net, 'num_classes')
        args.num_classes = net.num_classes

    if args.local_rank == 0:
        _logger.info(f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in net.parameters()])}')

    data_config = resolve_data_config(vars(args), model=net, verbose=args.local_rank == 0)

    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1
        num_aug_splits = args.aug_splits

    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        net = convert_splitbn_model(net, max(num_aug_splits, 2))

    # move model to GPU
    net.cuda()
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last)

    # sync BN for distributed
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp != 'native':
            net = convert_syncbn_model(net)
        else:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if args.torchscript:
        assert not use_amp == 'apex'
        assert not args.sync_bn
        net = torch.jit.script(net)

    optimizer = create_optimizer_v2(net, **optimizer_kwargs(cfg=args))

    amp_autocast = suppress
    loss_scaler = None
    if use_amp == 'apex':
        net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. float32.')

    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            net, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0
        )

    model_ema = None
    if args.model_ema:
        model_ema = ModelEmaV2(net, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    if args.distributed:
        if has_apex and use_amp != 'native':
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DDP.")
            net = ApexDDP(net, delay_allreduce=True, find_unused_parameters=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DDP.")
            net = NativeDDP(net, device_ids=[args.local_rank], find_unused_parameters=True)

    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info(f'Scheduled epochs: {num_epochs}')

    # ---------------------------
    # Datasets / Loader
    # ---------------------------
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False, batch_size=args.batch_size
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.val_batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # output dir (rank0 only)
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = get_outdir(args.output if args.output else './output/train', exp_name)
        args.output_dir = output_dir  # <-- IMPORTANT: used for CSV output
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # ---------------------------
    # Register spike counter AFTER DDP wrap (works for both)
    # ---------------------------
    spike_counter = SpikeInputCounter(
        include_types=(nn.Conv2d, nn.Conv1d, nn.Linear),
        exclude_name_suffixes=("patch_embed.proj_conv",),
    ).register(net)

    try:
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            if args.local_rank == 0:
                _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(net, args.world_size, args.dist_bn == 'reduce')

        eval_metrics = validate(net, loader_eval, validate_loss_fn, args,
                                amp_autocast=amp_autocast, spike_counter=spike_counter)
        print('The test metrics is', eval_metrics)

        if model_ema is not None and not args.model_ema_force_cpu:
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

    except KeyboardInterrupt:
        pass
    finally:
        spike_counter.remove()


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', spike_counter=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    if spike_counter is not None:
        spike_counter.reset()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx

            if not args.prefetcher:
                input = input.cuda()
                target = target.cuda()

            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)

            if isinstance(output, (tuple, list)):
                output = output[0]

            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)

            # IMPORTANT: reset spiking states
            functional.reset_net(model)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()

            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m)
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    # print spike/nonzero summary + write CSV to output_dir
    if spike_counter is not None and args.local_rank == 0:
        rows = spike_counter.summary(sort_by="rate")
        _logger.info("==== nonzero count on INPUT activations (Conv/Linear + SSA matmul operands) ====")
        for name, ones, total, rate, shape, dtype in rows:
            _logger.info(f"{name:60s} nz={ones:12d}  total={total:12d}  rate={rate:.6f}  shape={shape}  dtype={dtype}")

        # ---- CSV output (ONLY this, no per-batch loss csv) ----
        try:
            outdir = args.output_dir if getattr(args, "output_dir", "") else "."
            os.makedirs(outdir, exist_ok=True)
            nz_csv_path = os.path.join(outdir, "act_nz_summary.csv")
            with open(nz_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["name", "nz", "total", "rate", "shape", "dtype"])
                for name, ones, total, rate, shape, dtype in rows:
                    w.writerow([name, int(ones), int(total), float(rate), str(shape), str(dtype)])
            _logger.info(f"[CSV] wrote activation nz summary to: {nz_csv_path}")
        except Exception as e:
            _logger.warning(f"Failed to write act_nz_summary.csv: {e}")

    return metrics


if __name__ == '__main__':
    main()