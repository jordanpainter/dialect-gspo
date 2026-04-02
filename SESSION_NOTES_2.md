# Session Notes 2 — Debugging, Evaluation & Reward Fixes

## Run Status at Session Start

All three runs (Gemma, Llama, Qwen) were active but had issues discovered during this session.

---

## Issues Found & Fixed

### 1. Qwen3 Thinking Mode (CRITICAL)
**Problem:** Qwen3 generates `<think>...</think>` blocks before responding. During training the model learned to fill the thinking block with garbage tokens (`useruser...` repeated thousands of times), consuming the entire 192-token completion budget and leaving no room for actual responses. This is reward hacking — the thinking section isn't scored so the model exploited it freely.

**Fix:** Added `enable_thinking=False` to `apply_chat_template` in `src/gspo.py`. This suppresses the thinking block entirely. Other models ignore this parameter so it's safe to apply globally.

**Impact:** The completed Qwen run (`gspo_qwen_200`) is corrupted by this — outputs are degenerate. Rerun required.

---

### 2. Dialect Reward: Gain → Absolute Density (CRITICAL)
**Problem:** The dialect reward was `gen_density - base_density` (gain over base model). The SFT models for Llama and Qwen already output high dialect density:

| Model | Base density | Room to improve |
|---|---|---|
| Gemma | 0.019 | High — gain works |
| Qwen | 0.086 | Near zero gain signal |
| Llama | 0.160 | Negative gain — actively penalised |

When gain variance collapses to near-zero, the running z-score normaliser divides by a tiny denominator, producing extreme z-scores that all saturate at the ±5.0 clip boundary. The reward becomes discrete {-5, 0, +5}, destroying gradient information and causing instability.

**Fix:** Changed `src/gspo.py` to use `r_d_gen` (absolute dialect density) instead of `r_d_gain` as the reward signal. `r_d_gain` is still computed and logged to WandB for monitoring.

**Paper defence:** "We reward models for generating dialectally rich text, measured by absolute dialect density. Gain over base is logged for analysis but the absolute signal is used as the reward, as SFT pre-training results in varied baseline dialect levels across models."

---

### 3. 8-bit Optimizer OOM (Llama & Qwen)
**Problem:** AdamW optimizer states (~2x model size) caused CUDA OOM on 4x RTX 3090 (96GB total) for 8B models. The policy model + reference model + COMET + SentenceTransformer left no room for optimizer states.

**Fix:** Added `"optim": "adamw_bnb_8bit"` to `configs/llama.json` and `configs/qwen.json`. Reduces optimizer memory ~75% with no change to training algorithm or model quality. Not needed for Gemma (4B fits easily).

---

### 4. Learning Rate: 8B Models
**Problem:** At `5e-6`, Llama and Qwen showed severe instability — KL spikes to 1.87, grad norms hitting 2000+.

**Fix:** Reduced to `2e-6` for Llama and Qwen. Gemma stays at `5e-6`.

**Paper defence:** Report per-model LRs in a hyperparameter table. Standard practice for multi-model comparisons.

---

### 5. Partition Changes
**Problem:** A100 partition was heavily queued. 3090_risk had idle nodes.

**Resolution:**
- Llama and Qwen moved to `3090_risk` with 4x RTX 3090 and `--num_processes 1` + `device_map="auto"`
- When 3090_risk filled up, Qwen moved back to `a100` with `--gres=gpu:2`
- Gemma remains on `3090` partition

---

## Model Evaluation

### Uploaded models
- `jordanpainter/dialect-llama-gspo-all` — completed 5000-step run
- `jordanpainter/dialect-qwen-gspo-all` — completed 5000-step run (corrupted by thinking mode)

### Evaluation notebook
`notebooks/evaluate_gspo_models.ipynb` — loads each model, runs 8 test prompts, scores dialect density, flags degenerate outputs, saves CSV.

Key config options:
- `LOAD_IN_4BIT = True` — for running on a PC with limited VRAM (~4GB per model)
- Tokenizer loaded from instruct variant for chat template

### Results summary
**Llama GSPO** — outputs coherent, classifier scoring 0.12–0.15 dialect density. Usable. The sanity generation `spep.ComponentPlacement` output was a one-off fluke, not representative.

**Qwen GSPO** — EOS token not stopping generation, prompts repeat at end of outputs. Degenerate outputs confirmed. This run is not usable — being rerun with `enable_thinking=False`.

### Tokenizer investigation
Checked SFT training configs (`dial_post_train/configs/`). Both Llama and Qwen SFT were trained from CPT models (`DiaLLM-Llama3.1-8B-unaligned`, `DiaLLM-Qwen3-8B-unaligned`) with no `base_model_id` set. This means SFT used the TinyLlama-style fallback format:
```
<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n
```
GSPO training uses the same fallback (same tokenizers, same code path). Format is consistent throughout — not a source of issues.

---

## Monitoring

### Hourly Discord webhook
- Trigger ID: `trig_01QgbjjBTGz59d8BmjCYnNBP`
- Manage at: https://claude.ai/code/scheduled/trig_01QgbjjBTGz59d8BmjCYnNBP
- Posts to Discord every hour with per-run verdict (HEALTHY/LEARNING/UNSTABLE/COLLAPSED)

### Updated WandB run IDs
| Model | Run ID |
|---|---|
| gspo gemma | y2b17g6l |
| gspo llama | vushv3gb |
| gspo qwen | latest run on a100 (check WandB) |

---

## Current State

| Model | Status | Notes |
|---|---|---|
| Gemma | Finishing (~5000 steps) | Cleanest run, correct tokenizer, all fixes applied |
| Llama | Completed | Usable, dialectal features confirmed by classifier |
| Qwen | Rerunning on a100 (2 GPUs) | Clean restart with thinking mode disabled, all fixes |

---

## Up Next: Dialect-Specific GSPO

Plan for next session: train dialect-specific GSPO variants (British, Australian, Indian English) using dialect-specific datasets and models. Key consideration: **the reward function needs modifying per dialect** since dialectal features differ significantly between British, Australian and Indian English. The current classifier covers all dialects — for dialect-specific training we should reward features specific to the target dialect rather than generic dialect density.
