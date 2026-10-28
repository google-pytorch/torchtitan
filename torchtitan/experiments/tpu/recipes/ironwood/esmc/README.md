# ESM-C pretraining (`biohub/ESMC-600M`, `biohub/ESMC-6B`) on TPU Ironwood (tpu7x)

Masked-LM pretraining of the ESM Cambrian protein encoders through the TPU
HuggingFace backend (`torchtitan.experiments.tpu.transformers_modeling_backend`),
following "Language Modeling Materializes a World Model of Protein Biology":
MLM with `r_mask = 0.15`, AdamW (lr 3.1e-4, betas 0.9/0.95, eps 1e-8),
2048-token context. The paper's data mix is UniRef + MGnify + JGI; only UniRef
is redistributable, so the recipe trains on UniRef50 (see Data).

## Results

`tpu7x-16` slice (2 hosts x 4 chips, one process per TensorCore), libtpu
0.0.50, torch_tpu 0.2. Per-core throughput is the same on one and two hosts.

| model | hosts | tokens/rank | tokens/step | tok/s per TC | TFLOP/s | MFU | s/step | HBM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESMC-600M | 1 | 32 x 2048 | 524k | 52,568 | 235 | 20.4% | 1.25 | 6.0 GiB |
| ESMC-600M | 2 | 32 x 2048 | 4.19M | 53,318 | 238 | 20.7% | 4.92 | 6.0 GiB |
| ESMC-6B | 1 | 8 x 2048 | 131k | 7,850 | 339 | 29.4% | 2.09 | 3.4 GiB |
| ESMC-6B | 2 | 8 x 2048 | 262k | 7,922 | 342 | 29.6% | 2.07 | 3.4 GiB |
| ESMC-6B | 2 | 8 x 2048 | 4.19M | 8,070 | 348 | 30.2% | 32.5 | 3.4 GiB |

How the ESMC-600M number was reached, one host, ms per 2048 tokens:

| step | tok/s per TC | ms / 2048 tok |
|---|---:|---:|
| 1 x 2048 tokens/rank, 1024 splash blocks | 21,462 | 95.4 |
| 8 x 2048 tokens/rank, batch unfold, 2048 splash blocks | 47,990 | 42.7 |
| + ESMC attention layout (`_EsmcTPUAttention`) | 48,606 | 42.1 |
| + `sa_use_base2_exp`, `sa_use_experimental_scheduler`, no RNG draw at dropout 0 | 49,052 | 41.8 |
| + 32 x 2048 tokens/rank | 52,538 | 39.0 |

For ESMC-6B the same ladder is 4,810 -> 7,654 (8 x 2048, unfold) -> 7,897
(2048 splash blocks) tok/s per TC.

Convergence (MLM loss; the amino-acid unigram entropy is ~2.9 nats, below it
the model uses context):

- ESMC-600M, 1 host, 100 steps of 524k tokens, 20-step warmup:
  4.48 -> 2.97 (step 10) -> 2.89 (step 20) -> 2.71 (step 50) -> 2.63 (step 100;
  steps 81-100 mean 2.63); grad-norm 32 -> 0.4, no spikes.
- ESMC-6B, 1 host, 400 steps of 131k tokens: 4.65 -> 2.89 (step 20) -> 2.59
  (steps 351-400 mean); grad-norm 45 -> 0.8, no spikes.
- ESMC-6B, 2 hosts, 150 steps of 4.19M tokens (the paper's batch), 50-step
  warmup, cosine to 10%: 4.52 -> 2.82 (step 30) -> 2.61 (step 150); grad-norm
  46 -> 0.3, no spikes.

## What the config does

`--config=hf_esmc_mlm_uniref50` is `hf_full_max_plain_ce` (per-layer
`torch.compile(backend="tpu")`, SimpleFSDP, `AttentionSavingSelectiveAC`,
packed-document segment masking, `mixed_precision_reduce=float32`) plus:

- **MLM rows** (`MLMProcessor`, `experiments/tpu/datasets.py`): BERT/ESM
  80/10/10 corruption, labels only at masked positions, no label shift. The
  default next-token processor is trivially solvable by a bidirectional
  encoder (it reads the answer from the token to its right) and must not be
  used to judge convergence.
- **Batch unfold** (`TPUHFTransformerModel.forward`): the trainer folds the
  local batch into one `[1, B*S]` stream; a FullMask encoder cannot skip
  cross-sample attention tiles, so that costs `O((B*S)^2)`. The model runs the
  stream as `[B, S]` and folds back for the head (4 x 2048 tokens/rank on 6B:
  4,374 -> 7,165 tok/s per core).
- **Row-aligned packing** (`RowAlignedPackingConfig`, `pack_rows_to_context`):
  whole proteins per 2048-token row, first-fit across the rows of a batch,
  padding at the end of each row. Plain concat-then-split lets a document
  cross a row boundary, which the unfold would attend as two halves. Costs
  1.6% of tokens as padding on UniRef50 (measured over 30 batches; plain
  concat-then-split cut a document at 31 of 32 row boundaries there).
- **ESMC attention layout** (`_EsmcTPUAttention`, `model.py`): contiguous
  `(B, H, S, D)` q/k/v in forward and backward and `rotate_half` as a `(D, D)`
  matmul, instead of strided transposes and 32-wide sub-tile slices of the
  64-wide head (+11% on 600M).
- **Splash blocks 2048 / 2048 (compute 1024)** for forward and dkv: one kv
  block per q block removes the online-softmax rescale and the unreduced dq
  partial (+3.1% on 6B); `kv_dkv_compute=2048` runs out of VMEM.
  `sa_use_base2_exp` (hardware `exp2`) and `sa_use_experimental_scheduler`
  add +0.9% on 600M.
- **Tokens per rank**: the per-layer fp32 gradient reduce-scatter runs on the
  TensorCore and has to be amortized. 8 x 2048 is the 6B optimum (reduce-scatter
  20% -> 2% of the step; 16 x 2048 does not fit 96 GiB); 600M takes 32 x 2048
  (`--training.num_tokens_per_microbatch_per_dp_rank=65536`, +7% over 8 x 2048).

Tried and neutral or worse on this topology: the `tpu7x` SparseCore collective
offload flag bundle (-5%; the 1-D ring reduce-scatter is 9x slower than on the
TensorCore), grouping several layers per compiled graph (+1% at best),
`compile_optimizer` (within noise at these batch sizes).

## Data

```bash
# ESM-C tokenizer (33 tokens; <cls> 0, <eos> 2, <mask> 32)
mkdir -p $ASSETS/esmc_tokenizer && cd $ASSETS/esmc_tokenizer
for f in tokenizer.json tokenizer_config.json special_tokens_map.json; do
  curl -sL -o $f https://huggingface.co/biohub/ESMC-6B/resolve/main/$f
done

# UniRef50 as parquet with a `text` column (the schema of the HF mirror
# `bloyal/uniref50`; `--dataloader.dataset_path` may also point at that mirror,
# or be omitted to stream it). From the UniProt FTP, first 4M clusters:
python experiments/tpu/recipes/ironwood/esmc/fasta_to_parquet.py $DATA/uniref50 4000000
```

## Training

One host (8 TensorCores). For ESMC-6B drop the `num_tokens_per_microbatch`
flag (the config's default is 8 x 2048):

```bash
TORCH_TPU_TIER2_COMPILATION_CACHE=$USER \
torchrun --nproc_per_node=8 -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.transformers_modeling_backend \
  --config=hf_esmc_mlm_uniref50 \
  --hf_model=biohub/ESMC-600M \
  --hf_assets_path=$ASSETS/esmc_tokenizer \
  --dataloader.dataset_path=$DATA/uniref50 \
  --training.num_tokens_per_microbatch_per_dp_rank=65536 \
  --training.steps=400 \
  --checkpoint.no-enable
```

Full `tpu7x-16` slice (2 hosts x 8 TensorCores). The TPU-VM init only knows
single-host layouts, so a multi-host launch exports the slice geometry from the
VM metadata (`tpu-env`: `CHIPS_PER_HOST_BOUNDS`, `HOST_BOUNDS`,
`ACCELERATOR_TYPE`) and the worker addresses. Run on **each** host with its own
`NODE_RANK` (0 on worker 0, 1 on worker 1):

```bash
export TPU_WORKER_HOSTNAMES=<worker0-ip>,<worker1-ip>
export TPU_CHIPS_PER_HOST_BOUNDS=2,2,1 TPU_HOST_BOUNDS=1,1,2 TPU_ACCELERATOR_TYPE=tpu7x-16
TORCH_TPU_TIER2_COMPILATION_CACHE=$USER \
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=static \
  --rdzv_endpoint=<worker0-ip>:29501 --node_rank=$NODE_RANK \
  -m torchtitan.experiments.tpu.train \
  --module=torchtitan.experiments.tpu.transformers_modeling_backend \
  --config=hf_esmc_mlm_uniref50 \
  --hf_model=biohub/ESMC-600M \
  --hf_assets_path=$ASSETS/esmc_tokenizer \
  --dataloader.dataset_path=$DATA/uniref50 \
  --training.num_tokens_per_microbatch_per_dp_rank=65536 \
  --training.num_tokens_per_train_step=4194304 \
  --training.steps=150 \
  --checkpoint.no-enable
```

`--training.num_tokens_per_train_step=4194304` is the paper's 4.2M-token
optimizer step (gradient accumulation over the 16 ranks' microbatches);
without it the slice runs 16 microbatches per step.
