# TorchTitan on TPU

Google's distribution of [TorchTitan](https://github.com/pytorch/torchtitan), adding training on
Cloud TPU through [TorchTPU](https://github.com/google-pytorch/torch_tpu). Upstream TorchTitan
works as documented; TPU support lives in `torchtitan/experiments/tpu` and adds TPU recipes, Pallas
kernels, and TPU-specific trainer and parallelism paths.

## Start here

* **Train on TPU**: [`torchtitan/experiments/tpu/README.md`](torchtitan/experiments/tpu/README.md)
  covers the environment, building the TorchTPU wheel, a single-host TPU VM run, and multi-host
  training on GKE.
* **Pick a recipe**: [`torchtitan/experiments/tpu/recipes`](torchtitan/experiments/tpu/recipes) has
  per-model, per-topology configs, each with its launch block.
* **Tune a model**: the
  [TorchTPU optimization guide](torchtitan/experiments/tpu/docs/torch-tpu-optimization-guide.md).
* **Everything else**: upstream's
  [README](https://github.com/pytorch/torchtitan/blob/main/README.md) and
  [docs](https://github.com/pytorch/torchtitan/tree/main/docs) apply unchanged.

## Issues and contributing

File TPU bugs and feature requests in
[this repository's issue tracker](https://github.com/google-pytorch/torchtitan/issues), and send
TPU changes here, under `torchtitan/experiments/tpu`. Anything in shared TorchTitan code goes
upstream to [pytorch/torchtitan](https://github.com/pytorch/torchtitan), because we keep this
repository in sync with it. See [CONTRIBUTING.md](CONTRIBUTING.md) for the details.

## License

BSD 3-Clause, inherited from upstream TorchTitan. See [LICENSE](LICENSE).
