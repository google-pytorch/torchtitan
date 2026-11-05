# Gemma 4 E2B Post-Training Workshop (HF Accelerate → TorchTitan-TorchTPU + vLLM-TorchTPU)

<!-- disableFinding(LINE_OVER_80) -->
<!-- disableFinding(SNIPPET_EM_DASH) -->

This directory contains the executable scripts and launcher for the 4-lab **Gemma 4 E2B Post-Training Workshop** on **Cloud TPU v6e (`torch_tpu`)**:

## Files in This Directory

- `lab1_hf_accelerate_baseline.py`: Baseline Hugging Face `accelerate` post-training script (supports `--mode cpt` and `--mode sft`) with inline migration annotations contrasting raw `accelerate` against TorchTitan-TorchTPU.
- `prepare_hf_checkpoint_for_vllm.py`: Post-training utility that copies `config.json`, `generation_config.json`, and tokenizer files from `--hf_assets_path` into the TorchTitan `step-<N>` checkpoint folder alongside `model.safetensors` so `vllm-torchtpu` can load it directly.
- `lab4_vllm_torchtpu_inference.py`: Offline batch evaluation and XProf profiling script using `vllm-torchtpu` (`vllm.LLM`) on Python coding prompts.
- `run_workshop.sh`: End-to-end launcher script supporting stages `download`, `cpt`, `sft`, `prepare_vllm`, `infer_offline`, `infer_serve`, and `all`.

## TorchTitan Configs Registered in `torchtitan.experiments.tpu.transformers_modeling_backend`

- `gemma4_e2b_cpt_codeparrot`: **Lab 2** Continuous Pre-Training on `codeparrot/codeparrot-clean-valid` (`ConcatThenSplitPackingConfig` + `splash_tpu` + `block_causal` + `PallasCrossEntropyLoss`).
- `gemma4_e2b_cpt_codeparrot_plain_ce`: Portable `CrossEntropyLoss` variant of Lab 2.
- `gemma4_e2b_sft_alpaca`: **Lab 3** Supervised Fine-Tuning on `iamtarun/python_code_instructions_18k_alpaca` (`FirstFitPackingConfig` + `ChatProcessor` + `HFBackendTokenizer` + `splash_tpu` + `PallasCrossEntropyLoss`).
- `gemma4_e2b_sft_alpaca_plain_ce`: Portable `CrossEntropyLoss` variant of Lab 3.

## Do You Still Need HF `accelerate` with `TorchTitan` + `torch_tpu`? (**No**)

With `torchtitan` + `torch_tpu`, you still use **Hugging Face `transformers`** (`Gemma4ForCausalLM`, `AutoTokenizer`, `.safetensors`) and **Hugging Face `datasets`**, but **TorchTitan completely replaces `accelerate`** as the training engine while **`torch_tpu` replaces `torch_xla`** as a native PyTorch 2 (`PrivateUse1` = `'tpu'`) backend:

| What `accelerate` Used to Do | What Replaces It in `TorchTitan` + `torch_tpu` | Do You Need `accelerate`? |
|---|---|---|
| **Multi-chip process launcher** (`accelerate launch` / `xmp.spawn`) | Standard PyTorch **`PJRT_DEVICE=TPU torchrun --nproc_per_node=4 -m torchtitan.experiments.tpu.train`** | **No** |
| **Model & Tokenizer loading** (`AutoModelForCausalLM`, `AutoTokenizer`) | TorchTitan's **`transformers_modeling_backend`** (`TPUHFTransformerModel`) loads your HF model (`--hf_model=google/gemma-4-E2B`) directly on the `meta` device (`0 GiB` initial host/HBM copy) | **No** *(uses `transformers` directly)* |
| **Distributed sharding** (`accelerator.prepare` / `XlaFSDP`) | `parallelize_hf_transformers_tpu` shards weights, gradients, and AdamW states across TPU chips using compiler-aware **`SimpleFSDP` (`spmd_types` `DTensor`)** | **No** |
| **Step execution & mixed precision** (`accelerator.backward(loss)`) | Per-decoder-layer **`torch.compile(backend="tpu")`** + automatic swap of HF `sdpa` $\rightarrow$ Pallas **`splash_tpu`** (`block_causal`) | **No** |
| **Output head + Cross-Entropy (262k Vocab)** (`outputs = model(..., labels=labels)`) | Fused Mosaic TPU **`PallasCrossEntropyLoss`** (`b_block_size=2048, h_block_size=512, v_block_size=1024`, auto-mirroring Gemma's `final_logit_softcapping=30.0` in VMEM without materializing `[B*S, 262k]` logits) | **No** |
| **Saving HF checkpoints** (`accelerator.unwrap_model().save_pretrained()`) | TorchTitan's **`CheckpointManager`** (`--checkpoint.initial_load_in_hf --checkpoint.last_save_in_hf`) reads and writes standard Hugging Face `model.safetensors` | **No** |

## Measured Benchmark on Cloud TPU v6e-4 (`4 Chips, 32 GiB HBM/chip`, Gemma 4 E2B)

| Configuration (`lab1_hf_accelerate_baseline.py` vs `TorchTitan`) | Chips Used | Per-Chip Shape `(bs, seq_len)` | Global Tokens / Step | Warmup / Compile Time | Steady-State Step Time | Throughput (`tps/chip` / `global_tps`) | Peak HBM / Chip | Status / Notes |
|---|---|---|---|---|---|---|---|---|
| **1. Default `HF accelerate` (`--strategy accelerate` on `torch_tpu`)** | **1 of 4** (`3 idle`) | `(2, 2048)` / `(1, 512)` | `4,096` / `512` | `151.16s` (`(1, 512)`) | **`1.42s`** (`(1, 512)`) | `360` / **`360 tps`** | **OOM at `(2, 2048)`** (`>32 GiB`) | `accelerate` only detects `torch_xla`; falls back to 1 chip (`tpu:0`) and OOMs above `(1, 512)` |
| **2. 4-Chip PyTorch `DDP` (`--strategy ddp_tpu` on `torch_tpu`)** | **4 of 4** | `(2, 2048)` | `16,384` | — | — | — | **OOM (`>32 GiB`)** | Unsharded `5.1B` weights (`10.2 GB`) + `AdamW` states (`20.5 GB`) + `262k`-vocab logits exceed `32 GiB` |
| **3. Default 4-Chip `HF accelerate` + `torch_xla` (`XlaFSDP` + `AdamW`)** | **4 of 4** | `(2, 2048)` | `16,384` | — | — | — | **OOM (`32.48–39.44 GiB > 31.25 GiB`)** | `XlaFSDP` requires `fp32` primary params (`4.31 GiB`) + `fp32` grads (`4.31 GiB`) + `fp32` `AdamW` states (`8.62 GiB`) + `15.23 GiB` program |
| **4. Tuned 4-Chip `HF accelerate` + `torch_xla` (`--strategy xla_fsdp`, `XlaFSDP` + `Adafactor` + 8 workarounds)** | **4 of 4** | **`(2, 2048)`** | **`16,384`** | **`240.97s`** (Steps 1–2) | **`1.35s – 1.36s`** | **`3,025 tps/chip`** (**`12,100 tps` global**) | **`21.66 GiB`** (`69.3%`) | Requires 8 manual `torch_xla` workarounds (`xmp.spawn(nprocs=None)`, `fp32` primary params, custom PLE wrap, `checkpoint_module`, `shard_param_on_dim_0=True`, `Adafactor`) |
| **5. 4-Chip `TorchTitan` + `torch_tpu` (`FSDP2` + `AdamW`, Labs 2 & 3)** | **4 of 4** | **`(2, 2048)`** | **`16,384`** | **`92.04s`** (Step 1) | **`1.00s – 1.02s`** | **`4,027–4,080 tps/chip`** (**`16,108–16,320 tps` global**) | **`8.75 GiB`** (**`27.34%`**) | **1.35x faster step time**, **2.6x faster compile**, **2.5x lower HBM (`8.75 GiB` vs `21.66 GiB`)** with full **`AdamW`** via native `bfloat16` `FSDP2` (`DTensor`) |

## Quickstart

### 1. Environment Setup (Public PyPI or Public Container Image)

```bash
# Option A: Install public PyPI release (torch-tpu 0.2.0) in a Python 3.12 venv
uv venv --python 3.12 ~/torchtpu_venv
source ~/torchtpu_venv/bin/activate
pip install torch-tpu==0.2.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate grain safetensors tensorboard tyro

# Option B: Use the prebuilt public release container image
# us-docker.pkg.dev/ml-oss-artifacts-published/torch-tpu-public-release-container/torch-tpu:latest
```

### 2. Run Workshop Stages

```bash
# Run individual stages or the full workshop (from the torchtitan repo root):
bash torchtitan/experiments/tpu/workshops/gemma4_post_training/run_workshop.sh download
bash torchtitan/experiments/tpu/workshops/gemma4_post_training/run_workshop.sh cpt
bash torchtitan/experiments/tpu/workshops/gemma4_post_training/run_workshop.sh sft
bash torchtitan/experiments/tpu/workshops/gemma4_post_training/run_workshop.sh prepare_vllm
bash torchtitan/experiments/tpu/workshops/gemma4_post_training/run_workshop.sh infer_offline
```
