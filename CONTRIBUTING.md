# Contributing to TorchTitan on TPU

This repository is Google's distribution of [TorchTitan](https://github.com/pytorch/torchtitan),
adding training on Cloud TPU through [TorchTPU](https://github.com/google-pytorch/torch_tpu). It
tracks upstream closely, and we intend to keep it that way.

## Issues

Bugs and feature requests for TPU support go to
[this repository's issue tracker](https://github.com/google-pytorch/torchtitan/issues). Problems in
shared TorchTitan code that also reproduce off TPU belong
[upstream](https://github.com/pytorch/torchtitan/issues). When in doubt, file here and we will
redirect it. A clear, well-isolated bug report is itself a valuable contribution, and for anything
more than a trivial fix, filing an issue first lets us agree on the shape of the solution before
you write code.

A good report carries what happened against what you expected with the full traceback, the recipe
and exact launch command that reproduces it, and your environment: `torch`, `torch_tpu` and
`libtpu` versions, TPU generation and topology, and whether the run was single-host or multi-host.

## Where your change belongs

**Please keep changes inside `torchtitan/experiments/tpu`.** Everything outside it is upstream code
that we re-sync from `pytorch/torchtitan`, so an edit there is either reverted by the next sync or
becomes a conflict that someone resolves by hand.

* **TPU recipes, kernels, trainer, overrides and tests**: here, under
  `torchtitan/experiments/tpu`.
* **Shared TorchTitan code** (trainer, parallelism, models, components): send it to
  [pytorch/torchtitan](https://github.com/pytorch/torchtitan); it reaches this repository on the
  next sync.
* **Shared code that the TPU lane needs and upstream has not taken yet**: please open an issue here
  first. A small, device-neutral extension point upstream beats a local fork of the file.

## Before you begin

Contributions to this project must be accompanied by a Google Contributor License Agreement (CLA).
You (or your employer) retain the copyright to your contribution; signing simply gives us
permission to redistribute your changes. Visit <https://cla.developers.google.com/> to see your
agreements or to sign a new one. If you have already signed the Google CLA for another open-source
project, you likely do not need to sign it again. A CLA bot checks pull requests automatically and
guides you if action is needed.

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Pull requests

Small and focused beats large and sprawling.

Approved pull requests are copied into Google's internal repository by a tool called Copybara. For
this reason Copybara does not mark a pull request as merged: it closes the pull request and
separately applies a commit to `main` with the same changes. Your change has landed when that
commit appears.

Upstream's [CONTRIBUTING](https://github.com/pytorch/torchtitan/blob/main/CONTRIBUTING.md) covers
the review standards for TorchTitan code, and we do not restate them here. For a change under
`torchtitan/experiments/tpu`, please also show:

* **It trains**: at least 10 steps on the relevant recipe, loss decreasing, no NaN or inf, with
  conclusions drawn from steps past warmup.
* **Its numerics**: bit-identical (step-1 loss and grad_norm unchanged), rounding-order (fp32
  round-off from tiling, fused kernels or reduction order), or semantic. A semantic change is a
  separate, labelled recipe rather than a performance lever.
* **That the mechanism fired**: kernels verified in the HLO dump, since eager log lines do not
  print under `torch.compile`.
* **Performance, and the command behind it**: step time, TPS, MFU and peak HBM on stated hardware
  and topology.
* **A recipe, for a new model**: a per-topology config under
  `torchtitan/experiments/tpu/recipes`, plus a README carrying the launch block.

[`torchtitan/experiments/tpu/README.md`](torchtitan/experiments/tpu/README.md) has the setup and
launch instructions; the
[TorchTPU optimization guide](torchtitan/experiments/tpu/docs/torch-tpu-optimization-guide.md) has the
tuning levers.

## Tests

Put tests in a `tests/` directory next to the code they cover, and let the file name say what
hardware the test needs. Both GitHub CI and Google's internal build read the hardware lane from the
name, so a test that follows the convention runs in both without any CI or build configuration:

| File name                                        | Runs on                                              |
| ------------------------------------------------ | ---------------------------------------------------- |
| `test_foo.py`                                    | CPU                                                  |
| `test_foo_tpu.py`                                | one TPU chip (`--device=tpu`)                        |
| `test_foo_distributed.py`                        | eight TPU chips, multi-process (`--num_devices=8`)   |
| `torchtitan/experiments/tpu/kernels/tests/*.py`  | one TPU chip, whatever the suffix (Pallas kernels)   |

Single-chip tests subclass `BaseAcceleratorDeviceTest`
([`base_device_test.py`](torchtitan/experiments/tpu/testing/base_device_test.py)) and multi-chip tests
subclass `BaseDistributedDeviceTest`
([`base_distributed_device_test.py`](torchtitan/experiments/tpu/testing/base_distributed_device_test.py));
CI checks that a `BaseDistributedDeviceTest` lives in a `*_distributed.py` file. A test may import a
sibling test module for shared fixtures. Anything else (an unusual topology, a custom shard count, a
dependency outside the standard set) is configured by hand on the internal side, so please call it
out in the pull request.

A test that is temporarily broken is listed, with the reason, in
[`.ci/discover_tpu_tests.py`](.ci/discover_tpu_tests.py) rather than deleted or skipped inline.

## AI-generated code

Use of AI assistants is neither encouraged nor discouraged, but **the author is fully responsible
for the code they submit**: you must be able to explain and maintain it, and it must be correct and
tested. Pull requests that appear to be unreviewed automated output will be closed. AI assistants
are not co-authors; if one provided meaningful help, you may add an `Assisted-by:` tag in the
commit message.

## Security

Please report potential security vulnerabilities through <http://g.co/vulnz> rather than in a
public issue; see [SECURITY.md](SECURITY.md). High memory usage, crashes caused by invalid inputs,
and numerical divergence from untrusted models are ordinary bugs rather than security
vulnerabilities: running untrusted model code is equivalent to running untrusted arbitrary Python
code.
