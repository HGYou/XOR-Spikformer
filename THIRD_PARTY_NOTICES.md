# Third-party notices

This repository is released under the MIT License (`LICENSE`). It contains code
derived from the two projects below, which stay under their own terms.

## Spikformer

* <https://github.com/ZK-Zhou/spikformer>
* MIT License, Copyright (c) 2022 Zhaokun Zhou — full text in `LICENSES/Spikformer-MIT.txt`
* Derived files: `cifar/model.py`, `tiny_imagenet/model.py`, `imagenet/model.py`
* Modified here: the residual fusion function `fuse()`, the `--fusion` argument that
  selects it, the overlap term collected at each fusion site, and the
  membrane-shortcut (`ms`) path in the CIFAR and Tiny-ImageNet models.

## pytorch-image-models (timm)

* <https://github.com/huggingface/pytorch-image-models>
* Apache License 2.0, Copyright 2019–2020 Ross Wightman — full text in
  `LICENSES/Apache-2.0.txt` (timm's own copy, v0.5.4; that release ships no `NOTICE`
  file)
* Derived files: `cifar/train.py`, `cifar/test.py`, `cifar/loader.py`,
  `cifar/aa_snn.py`, `cifar/transforms_factory.py`, `tiny_imagenet/train.py`,
  `tiny_imagenet/test.py`, `imagenet/train.py`, `imagenet/test.py`
* Modified here: spiking model construction, the `--fusion`, `--lam-ov` and
  `--ov-warmup-epochs` arguments, the overlap-regularization term in the loss, and
  the layer-wise sparsity logging in `test.py`.

Every file listed above carries a source and modification notice at the top.
`cifar/aa_snn.py`, `cifar/loader.py`, `cifar/transforms_factory.py`,
`cifar/train.py`, `cifar/test.py`, `tiny_imagenet/train.py` and `imagenet/train.py`
also retain the upstream attribution line that came with them;
`tiny_imagenet/test.py` and `imagenet/test.py` do not, and carry only the notice
added here.
