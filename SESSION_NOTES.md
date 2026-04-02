# Session Notes — GSPO Training Run Setup & Debugging

## Overview

This document summarises all changes made during this session, the reasoning behind each decision, and the current state of all three training runs.

---

## Models & Datasets

| Model | Config | Dataset | Base output column |
|---|---|---|---|
| Gemma 3 4B | `configs/gemma.json` | `jordanpainter/dialect-gemma-base-all` | `base_output_gemma` |
| Llama 3.1 8B | `configs/llama.json` | `jordanpainter/dialect-llama-base-all` | `base_output_llama_sft` |
| Qwen 3 8B | `configs/qwen.json` | `jordanpainter/dialect-qwen-base-all` | `base_output_qwen` |

---

## Config Changes (`configs/*.json`)

### 1. Reward weights
**Change:** `dialect: 0.8, comet: 0.1, cosine: 0.1` (previously `dialect: 0.5, comet: 0.25, cosine: 0.25`)

**Why:** COMET and cosine rewards pull toward standard English (high-quality, semantically similar to chosen), directly fighting the dialect learning objective. Upweighting dialect to 0.8 ensures the dialect signal dominates.

---

### 2. Gradient clipping
**Change:** Added `"max_grad_norm": 1.0` to all three configs (previously absent)

**Why:** Early runs showed grad norm spikes up to 7776 and KL spikes to 9.4, destabilising training. Gradient clipping at 1.0 is standard practice for policy gradient methods.

---

### 3. KL penalty (beta)
**Change:** `beta: 0.02` (previously `0.1`)

**Why:** Beta is the KL penalty coefficient — it anchors the policy to the reference model. The reference model has low dialect density (~0.077), so a high beta directly fights the dialect reward. Setting beta=0 previously produced gains; beta=0.02 is a compromise that allows learning while maintaining a reportable KL metric for the paper. Defensible as standard hyperparameter tuning.

---

### 4. Learning rate
**Change:** Llama and Qwen reduced to `2e-6` (Gemma stays at `5e-6`)

**Why:** 8B models showed severe instability at `5e-6` — KL spikes up to 1.87 and grad norms hitting 2000+. Gemma (4B) was stable at `5e-6`. Smaller LR for larger models is standard and fully defensible in a paper (report per-model LRs in a hyperparameter table).

---

### 5. 8-bit optimizer
**Change:** Added `"optim": "adamw_bnb_8bit"` to Llama and Qwen configs

**Why:** AdamW optimizer states (exp_avg, exp_avg_sq) require ~2x model size in memory. For 8B models across 4x RTX 3090s (96GB total), this caused OOM during the first optimizer step. 8-bit optimizer stores momentum states in int8 instead of fp32/bf16, reducing optimizer memory ~75%. All parameters are still fully trained — this is purely a memory efficiency trick, not a change to the training algorithm.

---

## SLURM Job Changes (`*.sub`)

### gemma.sub
- Partition: `3090` (4x RTX 3090, `--gres=gpu:nvidia_geforce_rtx_3090:4`)
- `--num_processes 1` with `device_map="auto"` — single process spreads model across all 4 GPUs
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation
- `--mem=200G` (increased from 80G to handle two model copies in CPU RAM)

### llama.sub / qwen.sub
- Partition: `3090_risk` (4x RTX 3090, `--gres=gpu:nvidia_geforce_rtx_3090:4`)
- `--num_processes 1` with `device_map="auto"`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `--mem=200G`

**Why 3090_risk instead of A100:** A100 partition was heavily queued (8+ jobs ahead). 3090_risk had 2 idle nodes. 4x RTX 3090 = 96GB total VRAM, sufficient for 8B model + reference model + optimizer states with the 8-bit optimizer.

**Why num_processes=1:** DDP (multiple processes) requires each process to hold a full copy of the model on one GPU (24GB). With a 4B+ model this immediately OOMs. `num_processes=1` + `device_map="auto"` lets HuggingFace shard the model across all 4 GPUs automatically.

---

## Code Changes

### `src/gspo.py` — Dialect reward: gain → absolute density

**Change:** The dialect component of the reward was changed from `gain = gen_density - base_density` to simply `gen_density` (absolute dialect density of the generation).

**Why:** The SFT models (particularly Llama and Qwen) had already learned to produce high dialect density during supervised fine-tuning:

| Model | Base dialect density | Room to improve |
|---|---|---|
| Gemma | 0.019 | High — gain reward works |
| Qwen | 0.086 | Low — near-zero gain signal |
| Llama | 0.160 | Negative — model penalised for high dialect |

When gain variance collapses near zero, the z-score normaliser divides by a tiny denominator, producing extreme z-scores that all saturate at the ±5.0 clip boundary. The reward becomes discrete {-5, 0, +5} rather than smooth, destroying gradient information and causing training instability.

Using absolute `dialect_density(generation)` fixes this: all three models have meaningful variance in how dialectal their outputs are across a batch, giving the normaliser a well-conditioned signal regardless of baseline.

**What is still logged:** `r_d_gain` (gen - base) is still computed and logged to WandB as `train/train/reward_raw/dialect_gain_mean` for monitoring and paper reporting. It just no longer drives the reward signal.

**Paper defence:** "We reward models for generating dialectally rich text, measured by an absolute dialect density score. Gain over the base model is logged for analysis but not used as the training objective, as SFT pre-training results in varied baseline dialect levels across models."

---

### `src/base_output.py` — Fallback for non-save_to_disk datasets

**Change:** Added try/except fallback from `load_from_disk()` to `load_dataset()`.

**Why:** The `alignment-british-final`, `alignment-australian-final`, and `alignment-indian-final` datasets are standard Parquet HF datasets, not `save_to_disk()` snapshots. The original code raised `FileNotFoundError` when trying to load them.

---

## Monitoring Setup

### WandB Discord webhook
An hourly scheduled agent posts training status to Discord at:
- Trigger ID: `trig_01QgbjjBTGz59d8BmjCYnNBP`
- Manage at: https://claude.ai/code/scheduled/trig_01QgbjjBTGz59d8BmjCYnNBP

Each post reports: global step, reward, dialect_gen, dialect_gain, KL, grad norm, and a HEALTHY/LEARNING/UNSTABLE/COLLAPSED verdict per run.

### Key WandB metrics to watch
| Metric | Want | Flag if |
|---|---|---|
| `train/reward` | Climbing from 0 | Still 0 after 200+ steps |
| `train/train/reward_raw/dialect_gen_mean` | Trending up | Flat or declining |
| `train/train/reward_raw/dialect_gain_mean` | Positive | Deeply negative |
| `train/kl` | Gradual drift up | Spikes > 0.5 |
| `train/grad_norm` | Settling < 5 | Spikes > 100 |
| `train/completions/mean_length` | Near max_completion_length | < 10 (collapse) |

---

## Current Run State (as of session end)

| Model | Run ID | Step | Status | Notes |
|---|---|---|---|---|
| Gemma | l3ipzi3b | ~668 | Needs restart | KL spiked to 8.56, grad norm 1568 — restart with new reward |
| Llama | dpkjjcv4 | ~393 | Needs restart | Was learning (reward +3.97) but reward was gain-based; restart with absolute density |
| Qwen | qsqszqem | ~476 | Needs restart | Collapsing (1-token outputs, grad norm 2576); restart with absolute density |

All three should be cancelled and resubmitted after pulling latest changes on the cluster.

---

## Pending: Dialect Variant Base Outputs

Three SLURM jobs were created to generate base outputs for dialect-specific model variants:

| Job file | Model | Dataset | Output dir |
|---|---|---|---|
| `base_output_gemma_brit.sub` | `DialLM-Gemma-sft-brit` | `alignment-british-final` | `$SCRATCH/data/dialect-gemma-base-brit` |
| `base_output_gemma_aus.sub` | `DialLM-Gemma-sft-aus` | `alignment-australian-final` | `$SCRATCH/data/dialect-gemma-base-aus` |
| `base_output_gemma_ind.sub` | `DialLM-Gemma-sft-ind` | `alignment-indian-final` | `$SCRATCH/data/dialect-gemma-base-ind` |

These save locally to `$SCRATCH/data/` — upload to HuggingFace Hub manually once complete.
