# Fitting an 8B QLoRA into 12 GB

Measured on an RTX 4080 Laptop (12,282 MiB VRAM, 31.7 GB system RAM), Windows 11,
CUDA 12.4, Qwen3-8B.

Companion to the [README](../README.md) and [pitfalls](pitfalls.md).

---

## The budget

4-bit NF4 with double quantisation, LoRA rather than full fine-tuning:

| Item | Usage |
|---|---|
| 4-bit quantised weights | ~10,020 MB |
| Free after load | ~2,262 MB |
| `embed_tokens` + `lm_head` (kept fp16) | ~2.5 GB |
| Parameter mix | `3472.9M uint8 + 1245.0M float16` |

Trainable parameters with `r=16` on `q/k/v/o/gate/up/down_proj`:

```
43,646,976 trainable  /  8,234,382,336 total  =  0.5301%
```

The adapter itself is ~175 MB, which is why shipping the LoRA instead of a merged
model is worth it.

---

## What actually costs you VRAM

Ranked by impact in this project:

1. **Activation memory**, driven by sequence length. Dropping `MAX_LEN` 768 -> 512
   was nearly free because the real examples topped out at 443 tokens; the larger
   value only paid for padding.
2. **Optimiser state.** `paged_adamw_8bit` moves it to system RAM, which is the
   single biggest reason an 8B fits at all here.
3. **Batch size**, the obvious one.
4. **Gradient checkpointing** (`GRAD_CKPT=1`), trading compute for memory.

---

## The settings that work

```python
MAX_LEN = 512
BATCH = 1
GRAD_ACCUM = 8                     # effective batch 8
OPTIM = "paged_adamw_8bit"
GRAD_CKPT = 1
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

Result: **10-20 s/step**, power draw steady at **110-126 W**.

---

## The settings that look faster and are not

`BATCH=2, GRAD_ACCUM=4` benchmarks at **5.2 s/step**, 2.1x faster than 11 s/step.

But it leaves only ~**155 MB** of VRAM. Observed step time across 25 steps:

```
step  1- 5 : 5.6 s
step 10    : 5.7 s
step 15    : 5.7 s
step 20-25 : 7.8 - 8.4 s      <- degrading as it slides into paging
```

It never crashed. It just quietly got slower, and would have kept doing so.

**Recommendation**: if you have headroom, use it. If you have 155 MB, you do not
have headroom — you have a delayed failure.

---

## Diagnosing paging

Watch utilisation **and** power together:

```bash
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.free \
           --format=csv
```

| Utilisation | Power | Verdict |
|---|---|---|
| ~100% | ~TDP (110-126 W here) | healthy |
| ~100% | ~55-65 W | **paging — stop and reduce memory** |
| low | low | idle / input-bound |

Sampling repeatedly matters: power fluctuates, so take 3+ readings ~20 s apart.

---

## Things that did *not* help

- **`expandable_segments:True`** — on this Windows build PyTorch reports
  `expandable_segments not supported on this platform`. Keep the setting (some
  platforms honour it, and it is harmless otherwise) but do not count on it.
- **Higher `batch` as a speed strategy** — see above.
- **Loading the model in bfloat16 without quantisation** — an 8B needs ~16 GB.
  Not viable on a 12 GB card, and it is also why `merge_lora.py` works one shard
  at a time for the CPU-side merge.

---

## Quick reference for a 12 GB card

| Lever | Effect | Cost |
|---|---|---|
| `MAX_LEN` down | large activation saving | truncates long examples |
| `BATCH` down | linear activation saving | more steps |
| `GRAD_ACCUM` up | restores effective batch | only wall-clock |
| `paged_adamw_8bit` | moves optimiser state to RAM | slightly slower per step |
| `GRAD_CKPT=1` | big activation saving | ~20-30% slower |
| smaller base model | everything drops | less capacity (often fine for style) |

The last row is worth taking seriously: for mimicking short, stylised chat
messages, a 4B model is frequently sufficient, and it removes the memory pressure
entirely rather than managing it.
