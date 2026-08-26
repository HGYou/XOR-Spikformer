# XOR-Spikformer

Code for the TCAS-II brief *"XOR-Spikformer: A Resource-Efficient FPGA Accelerator for Spiking Transformers With XOR Residuals and Overlap Regularization"*.

This repository contains the training and evaluation code, experiment configurations, and measurement scripts used for the software experiments. FPGA results are reported in the paper.

```text
cifar/                          CIFAR-10/100 training and evaluation
tiny_imagenet/                  Tiny-ImageNet training and evaluation
imagenet/                       ImageNet training and evaluation
configs/<dataset>.yml           training configurations
results/args/<dataset>.yaml     settings of reported runs
tools/measure.py                firing rate, overlap, bit-width, and operation counts
scripts/prepare_tiny_imagenet.py
                                Tiny-ImageNet conversion script
```

## Setup

```bash
conda create -n xor_spikformer python=3.11 -y
conda activate xor_spikformer
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements.txt
```

The extra index is required for the `+cu118` PyTorch wheels.

The code uses the legacy `spikingjelly.clock_driven` API and the data pipeline from `timm==0.5.4`. Newer versions are not compatible without code changes.

## Datasets

Expected directory layout:

```text
<data_root>/cifar10/cifar-10-batches-py/
<data_root>/cifar100/cifar-100-python/
<data_root>/tiny-imagenet-200-if/{train,val}/<wnid>/*.JPEG
<data_root>/imagenet/{train,val}/<wnid>/*.JPEG
```

CIFAR datasets can be downloaded by torchvision.

Tiny-ImageNet must be converted to an ImageFolder-compatible layout before training:

```bash
python scripts/prepare_tiny_imagenet.py <data_root>/tiny-imagenet-200 \
    --out <data_root>/tiny-imagenet-200-if
```

Use `--link` to create hard links instead of copying the dataset.

## Training

Residual variants are selected with `--fusion`. The overlap-regularization weight is set by `--lam-ov`, and `--ov-warmup-epochs` controls its linear warm-up.

| operator | `--fusion` | forward |
| --- | --- | --- |
| ADD | `add` | `x + z` |
| XOR | `xor` | `x + z - 2xz` |
| OR | `or` | `x + z - xz` |
| clipped ADD | `or_clamp` | `min(1, x + z)` |
| membrane shortcut | `ms` | `x + F(LIF(x))` |

`or` and `or_clamp` have the same binary forward operation but different backward behavior.

The membrane-shortcut variant is implemented for CIFAR and Tiny-ImageNet. Overlap regularization is not applied to this variant because its residual operands are not binary.

### Reported XOR+OV runs

CIFAR-10:

```bash
cd cifar
python train.py -c ../configs/cifar10.yml \
    --data-dir <data_root>/cifar10 \
    --fusion xor \
    --lam-ov 10 \
    --ov-warmup-epochs 0 \
    --seed 42 \
    --gpu 0 \
    --output ./output
```

CIFAR-100:

```bash
cd cifar
python train.py -c ../configs/cifar100.yml \
    --data-dir <data_root>/cifar100 \
    --fusion xor \
    --lam-ov 10 \
    --ov-warmup-epochs 20 \
    --seed 42 \
    --gpu 0
```

Tiny-ImageNet:

```bash
cd tiny_imagenet
python train.py -c ../configs/tiny_imagenet.yml \
    --data-dir <data_root>/tiny-imagenet-200-if \
    --fusion xor \
    --lam-ov 25 \
    --ov-warmup-epochs 0 \
    --seed 42 \
    --gpu 0
```

ImageNet:

```bash
cd imagenet
python train.py -c ../configs/imagenet.yml \
    --data-dir <data_root>/imagenet \
    --fusion xor \
    --lam-ov 10 \
    --ov-warmup-epochs 20 \
    --seed 42
```

For the ADD and OR baselines, use:

```text
--fusion add --lam-ov 0
--fusion or  --lam-ov 0
```

`--data-dir` has no default value.

For CIFAR, architecture parameters such as `dim`, `layer`, `num_heads`, `patch_size`, and `time_step` are read from the configuration file.

For Tiny-ImageNet and ImageNet, the architecture is fixed in `model.py`: patch size 8 / 16, dimension 512, 8 heads, 8 blocks, and `T = 4`.

## Evaluation

Example for CIFAR-10:

```bash
cd cifar
CUDA_VISIBLE_DEVICES=0 python test.py -c ../configs/cifar10.yml \
    --data-dir <data_root>/cifar10 \
    --fusion xor \
    --initial-checkpoint <path>/model_best.pth.tar
```

`test.py` does not provide a `--gpu` option. Select the device with `CUDA_VISIBLE_DEVICES`.

The `--fusion` setting must match the checkpoint.

## Reported run settings

The complete settings for the reported experiments are stored in:

```text
results/args/<dataset>.yaml
```

Shared parameters are under `common`, while experiment-specific changes are under `experiments`.

For example:

```bash
python -c "import yaml; d=yaml.safe_load(open('results/args/cifar100.yaml')); \
           print({**d['common'], **d['experiments']['xor_ov']})"
```

Experiment keys follow the residual variant:

```text
add
xor
xor_ov*
or
or_ov
or_clamp
ms
```

Environment-specific fields (`gpu`, `output`, `experiment`, and `resume`) are omitted from these records. `data_dir` is stored as a placeholder.

## Measurement

`tools/measure.py` measures the firing rate, residual overlap, residual bit width, and the spike-conditioned operation count used in Eq. (7).

Example:

```bash
python tools/measure.py \
    --dir ./cifar \
    --cfg results/args/cifar10.yaml \
    --experiment xor_ov \
    --data-dir <data_root>/cifar10 \
    --ckpt <path>/model_best.pth.tar \
    --out ./measurements/cifar10_xor_ov
```

`--out` is used as a file prefix and produces:

```text
_layers.csv
_fusion_sites.csv
_summary.json
```

Additional options:

```text
--max-batches       limit the number of evaluated batches
--no-amp            disable AMP and use fp32
```

### Output fields

| field | description |
| --- | --- |
| `op_total` | Eq. (7) spike-conditioned operation count |
| `mean_firing_rate_all_layers` | nonzero fraction over all instrumented layers, including the first convolution |
| `mean_overlap_rate` | residual operand co-activation rate |

NormComp is computed by normalizing `op_total` to the ADD run of the same dataset.

For ADD, `mean_overlap_rate` is not defined; `support_coactivation_rate` is reported instead.

## Environment

| component | version |
| --- | --- |
| Python | 3.11 |
| PyTorch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| timm | 0.5.4 |
| spikingjelly | 0.0.0.0.12 |
| cupy | cupy-cuda11x 13.6.0 |

The neuron model is:

```text
MultiStepLIFNode(
    tau=2.0,
    detach_reset=True,
    backend='cupy'
)
```

The surrogate gradient is SpikingJelly's default `Sigmoid(alpha=4.0)`.

The threshold is 1.0 except for `attn_lif`, which uses 0.5 following the original Spikformer. All experiments use `T = 4`.

### FPGA environment

| item | setting |
| --- | --- |
| Vivado | 2025.1 |
| Vitis | 2025.1 |
| compiler | `aarch64-none-elf-gcc` 13.3.0 |
| board | AMD Versal VPK120 |
| device | `xcvp1202-vsva2785-2MP-e-S` |
| PL clock | 250 MHz |

## License

This repository is released under the MIT License. See `LICENSE`.

The implementation builds on [Spikformer](https://github.com/ZK-Zhou/spikformer) (MIT) and [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0).

See `THIRD_PARTY_NOTICES.md` for the origin of reused files and `LICENSES/` for the corresponding license texts.
