# Fine-tuning "this person" on 12 GB VRAM

**Complete experiment record: method, data, code, and every trap encountered. No corpus or weights included.**

English · [中文](README.zh-CN.md)

---

## Overview

Fine-tuned Qwen3-8B with QLoRA on a 12 GB VRAM laptop using single-user message records (~20k messages) from a Chinese group chat to replicate a specific conversational style.

This repository contains **method, code, and complete analysis**. Raw corpus, trained weights, and real conversation examples are not published (see [Privacy](#why-no-corpus)).

---

## Contents

- [Data limitations](#data-limitations)
- [Hardware constraints](#hardware-constraints)
- [Key problems](#key-problems)
- [Processing pipeline](#processing-pipeline)
- [Experimental results](#experimental-results)
- [Validation set problem](#validation-set-problem)
- [Project structure](#project-structure)
- [Reproduction](#reproduction)
- [Privacy](#privacy)

---

## Data limitations

Files exported by QQ Chat Exporter (via NapCat) **contain only the target user's messages**.

First step: quantify actual coverage. Group message `seq` numbers are strictly increasing, so coverage = exported count / (max_seq - min_seq + 1)

| Metric | Value |
|---|---|
| Exported messages | ~20,700 |
| seq range | 105,390 – 228,856 |
| seq span | 123,467 |
| Coverage | 16.70% |
| Missing slots | ~102,900 |

**16.7% coverage means ~100k missing slots are messages from other group members.**

Cause: The export tool calls `getMsgsBySeqAndCount` and `getAioFirstViewLatestMsgs` APIs, which are AIO (conversation view) scoped and structurally limited to returning the personal perspective, excluding other members' messages. Tool logs show `filtered out: 0`, indicating nothing was dropped—this is simply all that was available.

**Key point**: To determine if an export is single-user view, measure seq coverage directly. Near 100% indicates full group conversation; 10-20% indicates single-user view.

### Training impact

The most valuable supervision signal is "person says X → I reply Y". Without other members' messages in the export, this signal can only be extracted from **reply quotes**—reply elements contain the quoted original message and sender.

Result: Only ~1,244 usable pairs, 1,092 after cleaning. **This is the quality ceiling**.

The other signal source is "my message A → my message B" (consecutive self-messages), which is abundant but trains "how to continue my own thoughts" rather than "how to respond to others". This difference is the root cause of all subsequent problems.

---

## Hardware constraints

RTX 4080 Laptop: 12,282 MiB VRAM, 31.7 GB system RAM.

8B model trainable with QLoRA (4-bit NF4 + double quantization) but with zero headroom:

| Item | VRAM usage |
|---|---|
| 4-bit model weights | ~10,020 MB |
| Free | ~2,262 MB |

`embed_tokens` and `lm_head` remain fp16 (~2.5 GB), parameter composition approximately 3.47 GiB uint8 + 1.25 GiB float16.

### VRAM paging problem

Training exhibited significant performance degradation while conventional monitoring metrics appeared normal:

```
GPU utilization: 100%
Power draw: 54-67 W
Time per step: 20-66 s
```

**High utilization with low power draw is the key signal**. 100% utilization with only 55-65 W indicates the GPU is waiting on memory access. Actual cause: VRAM at 97% capacity, Windows WDDM paging VRAM to system memory, GPU compute units idling.

**Diagnostic rule**: High utilization + power far below TDP = VRAM paging. Normal training should consume 110-126 W. Monitoring only nvidia-smi utilization is misleading—power draw must be checked simultaneously.

### Solution

```python
per_device_train_batch_size = 1        # Prevent VRAM paging
gradient_accumulation_steps = 8        # Maintain effective batch size
optim = "paged_adamw_8bit"             # Page optimizer state to system RAM
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

After adjustment, power stabilized at 110-126 W with 10-20 s per step.

### Throughput tradeoff

For the same task, `batch=1 × accum=8` took 11 s/step while `batch=2 × accum=4` took 5.2 s/step (2.1× speedup). However, the latter left only ~155 MB free VRAM, with step time gradually degrading from 5.6 s to 8.4 s—already entering paging territory.

**Conclusion**: VRAM-saturated configurations appear viable but gradually degrade mid-run. Preserving headroom matters more than maximizing peak throughput.

---

## Key problems

### Problem 1: Export placeholders in training data

When the tool cannot parse a reply quote's original message, the `content` field becomes **the literal string `原消息`**.

Result: ~128 reply samples (10% of reply signal) used this literal as the "stimulus". Model learned "how to respond to the words '原消息'"—pure noise occupying the most valuable signal budget.

Similar noise: `[图片]`, `[视频:...]`, bare `@mention`, self-replies (quoted sender is self).

**Solution**: During dataset construction, apply **explicit blacklist validation** to stimulus fields, not just non-empty checks. Placeholders resemble text but carry no meaning.

### Problem 2: Task confusion

Original design used one system prompt and identical user structure for both "reply" and "continue" tasks:

```
system: You are mimicking group member X's speaking style...
user:   X: <my previous line>\nMember: <what they said>
```

Model learned the pattern: "recent text is all my own words".

Problem surfaced during inference: Asked "钟馗?", model replied "What big sword and small sword were you asking about earlier?"—fabricating non-existent context. Because training "context" was almost always self-messages, model learned to "pretend it just said something" rather than "respond to current input".

**Solution**: Different tasks require different system prompts and structures. Cannot expect model to infer task type from content—must distinguish structurally.

### Problem 3: TRL auto-discards labels

When manually constructing `labels`, prompt portions are marked -100 to restrict loss to generation only.

`SFTTrainer`'s `truncate` step **completely discards custom labels**, resulting in training on entire conversations rather than just learning replies.

```python
dataset_kwargs={'skip_prepare_dataset': True}   # Required
```

**Solution**: Write verification script proving prompt is exact prefix of full sequence and tokens are actually supervised (see `src/check_masking.py`). This class of bug raises no exceptions, only mysteriously poor results.

### Problem 4: thinking block boundaries

Qwen3 chat template inserts `<think></think>` blocks in assistant turns. Manual label construction can easily misalign boundaries.

Investigation method: Queried `<think>` token id, searched in supervised span, 0/11606 hits. Briefly suspected template structure changed.

Actually it correctly sat in the prompt region being masked: `add_generation_prompt=True` renders prompts ending with `<|im_start|>assistant\n<think>\n\n</think>\n\n`, with quoted message following.

**Solution**: Verification scripts should print actual text on both sides of boundary (using `repr()`), not just report found/not-found booleans.

### Problem 5: Chinese paths break command-line arguments

Chinese directory names passed to `llama-quantize.exe` get encoding-mangled:

```
中文目录  →  Ĭ�Ϲ�����
```

**Solution**: Create pure-ASCII directory junction, use forward slashes uniformly in scripts:

```cmd
mklink /J C:\gguf D:\path\to\workspace
```

```bash
WS=/c/gguf
```

**Note**: Solution depends on junction existing. Deleting junction causes all scripts to silently fail. Scripts should assert junction exists.

### Ollama 0.40.2 deployment routes blocked

| Attempt | Result |
|---|---|
| `ADAPTER` load LoRA | `LoRA adapters are no longer supported` |
| Import safetensors | `MLX runtime is not available` (MLX macOS-only) |
| `--quantize int4` | `quantize blob lm_head.weight: MLX init failed` |

**Conclusion**: Must convert to GGUF via llama.cpp manually (see `src/export_gguf.sh`).

---

## Processing pipeline

```
QQ export → JSONL cleaning → Build SFT dataset → Verify masking
                                                      ↓
                                                QLoRA tuning
                                                      ↓
                                                 LoRA merge
                                                      ↓
                                            HF → GGUF → Q4_K_M
```

### Key dataset construction decisions

**Decision 1: Task isolation**

| Task | system prompt | user turn | assistant turn |
|---|---|---|---|
| reply | Only respond to this line, don't mention your prior words | Member's original text (their words only) | Reply |
| continue | Continue your own words | Real multi-turn slice, self-messages in assistant turns | Next line |

Key: reply user turns **exclude self-history**, preventing context fabrication at the root.

**Decision 2: Use real multi-turn structure for continuation**

Placing self-messages in `assistant` role and others' in `user` role teaches model "my turn to speak" rather than "these recent characters are all mine".

### LoRA merge: Shard-by-shard processing

Loading full 8B model in bfloat16 requires ~16 GB RAM, which this machine cannot spare (31.7 GB total running other tasks). `src/merge_lora.py` uses **shard-by-shard processing**:

```python
delta = (B @ A) * (lora_alpha / r)      # 32 / 16 = 2.0
sd[key].add_(delta.to(sd[key].dtype))
```

Peak memory equals single shard (~4 GB) + LoRA delta, not full model.

**Note**: Do not hard-code `lora_alpha / r`. Must read from `adapter_config.json`. This experiment happened to be `r=16, alpha=32` (scale 2.0), but hard-coded constants will silently compute wrong merges for other configs.

---

## Experimental results

### Training configuration

| Item | Value |
|---|---|
| Base | Qwen3-8B |
| Method | QLoRA: 4-bit NF4 + double quantization |
| LoRA params | r=16, alpha=32, dropout=0.05, targets q/k/v/o/gate/up/down_proj |
| Trainable params | 43,646,976 (0.5301% of model) |
| Sequence length | 768 |
| Optimizer | paged_adamw_8bit |
| Batch | batch=1 × accum=8 |
| Learning rate | 2e-4, cosine decay, 3% warmup |

### Version comparison

| Metric | v1 (400 steps) | v2 (600 cumulative) |
|---|---|---|
| Train loss | 3.51 | 2.05 |
| Train token accuracy | 45.4% | 64.6% |
| Val loss | 3.91 | 4.32 ⚠️ |
| Val token accuracy | 40.0% | 39.5% |

Gap analysis:

```
              Train loss → Val loss    Gap
v1 (400 steps)   3.51   →   3.91      +0.40 (healthy)
v2 (+200 steps)  2.05   →   4.32      +2.27 (overfitting)
```

200 additional steps increased training accuracy from 45% to 65%, while validation remained unchanged (40.0% → 39.5%). v2 warm-started from v1 (initial train loss ~3.06), collapsing after 200 steps and 0.25 epochs. Overfitting arrives far faster than expected.

### Base vs fine-tuned comparison

| Prompt | Base Qwen3-8B | v1 | v2 |
|---|---|---|---|
| Spank you | Hahaha where? These are golden! | deer head | bring it on |
| You're so bad | I'd give that play full marks | so much chatter | oh I embarrassed myself |
| Lend money | A hundred million? Interest first... | either way, just afraid you can't repay | sure |
| Hiking tomorrow? | Tomorrow? My legs are jelly.jpg—fine, let's go | go | works for me |

Base model exhibits **verbose, explanatory, exclamation-heavy** assistant style; after fine-tuning produces **short fragments** matching corpus **median length of 8 characters**.

**Essence**: Fine-tuning doesn't "teach knowledge" but **transfers output length distribution and tone**.

### Loss vs similarity disconnect

Loss only measures "next token prediction accuracy", **not "sounds like this person"**.

In actual conversation v2 reads more authentic ("spank you" → v2 replies "bring it on", v1 replies "deer head"). Final criterion should be human assessment, not loss values. Both versions retained for A/B comparison.

---

## Validation set problem

**This is the most serious methodological error, more fundamental than overfitting itself.**

v1/v2 validation set composition:

| Component | Count | Share |
|---|---|---|
| Reply samples | 100 | 17% |
| Continue samples | 400 | 83% |

Continue samples' user turns were **randomly stitched from the same message pool**—not real conversation prefixes.

**Result: 83% of validation set is noise, measuring scenarios that don't exist in the real task.**

Consequences:

1. `eval_loss` loses meaning—noise set loss values have no stable relationship to "reply quality"
2. "v2 overfits" conclusion has correct direction but wrong evidence—if validation set itself is noise, cannot separate overfitting from noise contributions in loss increases

### Correct approach

Validation set must match inference task. Model purpose is "member says something → reply", so validation set should **contain only reply samples**:

```python
val_ds = reply_val
val_ds = rv.shuffle(seed=42)
```

Combined with:

- **Early stopping** based on `eval_loss`, not fixed step counts
- **Retain all checkpoints**, select by validation loss, not last step
- **Enable `load_best_model_at_end=True`**

**Key point**: Validation set defines the "good" standard. Metrics mismatched to real tasks cause all subsequent judgments to chase phantoms.

---

## Project structure

```
.
├── README.md              English docs
├── README.zh-CN.md        Chinese docs
├── requirements.txt
├── .gitignore             Critical: hides corpus and weights
├── src/
│   ├── config.py          Env vars for identity/paths (no hard-coding)
│   ├── clean_corpus.py    Corpus cleaning
│   ├── build_sft_dataset.py  Dataset construction
│   ├── check_masking.py   Masking verification
│   ├── train_qlora.py     QLoRA training
│   ├── merge_lora.py      LoRA merging
│   ├── export_gguf.sh     Format conversion
│   └── eval_adapters.py   Multi-adapter comparison
└── docs/
    ├── pitfalls.md        Detailed problems
    └── vram.md            VRAM diagnostics
```

---

## Reproduction

### Environment

```
torch 2.6.0+cu124      transformers 4.51.3   peft 0.15.2
trl 0.15.2             accelerate 1.5.2       bitsandbytes 0.45.3
datasets 3.5.0         numpy 2.2.6            sentencepiece 0.2.2
```

### Parameter configuration

Code contains **no hard-coded uin or nicknames**, all from environment variables:

```bash
export TARGET_UIN="<uin>"
export TARGET_NAME="<group nickname>"
export TARGET_GROUP="<group name>"
export BASE_MODEL="D:/models/Qwen3-8B"
export EXPORTS_DIR="C:/path/to/qq/exports"
```

### Training launch

```bash
export HF_HOME="D:/models/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUT_DIR="./my-lora"
export EPOCHS=1 MAX_LEN=768 BATCH=1 GRAD_ACCUM=8
python -u src/train_qlora.py
```

### Data verification

Must verify data quality before training:

```bash
python src/check_masking.py
```

This script verifies whether prompt is exact prefix of full sequence and how many tokens are actually supervised. This class of bug produces no exceptions, only poor results.

---

## Why no corpus

### Privacy

1. **Others' messages**: Group members' messages lack publication consent. Even though export contains only self-messages, **reply quotes still contain others' original text and nicknames**
2. **Life information leakage**: Group chat contains education, career plans, purchases, relationships. Publishing conversation samples leaks substantial information from context even when redacted
3. **Weights are compression too**: Fine-tuned weights are lossy corpus compression, so weights also remain private

### Published content

What's published is **method**. All problems—VRAM paging, placeholder leakage, validation set misalignment, label discarding—are privacy-agnostic and rarely discussed thoroughly in Chinese literature. This is the part worth preserving.

If you need similar processing, copy `.gitignore` before first commit.

---

## License

Code MIT. Documentation CC BY 4.0.
