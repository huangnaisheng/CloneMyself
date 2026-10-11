# Fine-tuning a "sounds like this person" model on 12 GB of VRAM

**A full write-up of the method, the numbers, and every trap I fell into. No chat corpus in this repository.**

English · [中文](README.zh-CN.md)

---

## In one line

Using **one person's own messages** from a Chinese group chat (~20k messages),
fine-tune Qwen3-8B with QLoRA on a **12 GB laptop GPU** so it talks like that person.

This repository contains **the method and the code only**. The corpus, the trained
weights, and any real conversation text are deliberately not published — see
[Why no corpus](#why-no-corpus).

---

## Contents

- [1. The problem, and the hard limit of the data](#1-the-problem-and-the-hard-limit-of-the-data)
- [2. Hardware reality: 12 GB of VRAM](#2-hardware-reality-12-gb-of-vram)
- [3. Five traps that are actually worth your time](#3-five-traps-that-are-actually-worth-your-time)
- [4. The pipeline](#4-the-pipeline)
- [5. Results](#5-results)
- [6. The validation-set trap](#6-the-validation-set-trap)
- [7. Repository layout](#7-repository-layout)
- [8. Reproducing this](#8-reproducing-this)
- [9. Why no corpus](#9-why-no-corpus)

---

## 1. The problem, and the hard limit of the data

The exporter (QQ Chat Exporter, running on NapCat) handed me a file containing
**only the target person's own messages**.

Don't start training yet — first measure what you actually got. Message `seq`
numbers in a group are **strictly increasing**, so coverage is just:

```
coverage = records_kept / (max_seq - min_seq + 1)
```

| Metric | Value |
|---|---|
| Records | ~20,700 |
| `seq` range | 105,390 – 228,856 |
| `seq` span | 123,467 |
| **Coverage** | **16.70%** |
| **Missing slots** | **~102,900** |

**16.7% coverage means the ~102,900 missing slots are where everyone else talked.
They never made it into the export at all.**

Here's why: the exporter calls `getMsgsBySeqAndCount` / `getAioFirstViewLatestMsgs`,
which are **AIO (conversation-view) scoped** APIs. They return the *local view of the
conversation from your own perspective*, and are **structurally incapable** of
returning other members' messages. Its own log says `filtered out: 0` — the tool
dropped nothing. That's simply all there was.

> **Takeaway**: to find out whether your export is "self-only", don't guess — measure
> `seq` coverage. Near 100% means the whole group conversation; 15–20% means a
> single-person view.

### What this means for training

The most valuable supervision signal is "**someone said X → I replied Y**". With no
other members' messages in the export, that signal can only be mined from
**reply quotes** — the reply element stores the quoted original message and its sender.

So there were only ~**1,244** usable pairs, **1,092** after cleaning.
**That is the ceiling on quality.**

The other signal path is "**my line → my next line**" (my own consecutive messages).
It's far more plentiful, but it teaches *"how to continue my own thought"* —
**not** *"how to answer someone"*. The difference between these two is the root
cause of every problem that follows.

---

## 2. Hardware reality: 12 GB of VRAM

RTX 4080 Laptop, 12,282 MiB VRAM, 31.7 GB system RAM.

An 8B model **is** trainable with QLoRA (4-bit NF4 + double quantisation), but with
zero headroom:

| Item | Usage |
|---|---|
| 4-bit model weights | ~10,020 MB |
| Free | ~2,262 MB |

`embed_tokens` and `lm_head` stay in fp16 (~2.5 GB). Parameter mix works out to
roughly `3472.9M uint8 + 1245.0M float16`.

### The key symptom: training "hangs" — it's not a hang, it's VRAM paging

This is the single most valuable trap, because **every conventional metric looks fine**:

```
GPU utilisation : 100%        <- looks like it's working hard
Power draw      : 54-67 W     <- THIS is the tell
Seconds/step    : 20-66 s
```

**100% utilisation at only 55–65 W** means the GPU is **waiting on memory**.
The real cause: 12 GB of VRAM was 97% full, and **Windows WDDM was paging VRAM out
to system RAM**, leaving the compute units idle.

> **Diagnostic rule**: `high utilisation + power far below TDP` = VRAM paging,
> not insufficient compute. Healthy training here draws 110–126 W.
> Watching only the utilisation figure will fool you — **always read power draw too**.

### The fix

```python
per_device_train_batch_size = 1        # not wasted VRAM: insurance against paging
gradient_accumulation_steps = 8        # effective batch is still 8
optim = "paged_adamw_8bit"             # page optimizer state to RAM, not VRAM
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

After this, power returned to 110–126 W and it held a steady **10–20 s/step**.

### A throughput lesson on the side

For the same work, `batch=1 × accum=8` ran 11 s/step while `batch=2 × accum=4` ran
**5.2 s/step** — 2.1× faster. But the latter left only ~**155 MB** of VRAM free, and
step time then degraded from 5.6 s to 8.4 s — **it was already sliding into paging**.

> A VRAM-saturated configuration **looks like it works**, then quietly degrades
> mid-run. Leaving headroom matters more than maximising peak throughput.

---

## 3. Five traps that are actually worth your time

This is the most valuable section of this document. The first four are not
"you misconfigured something" problems.

### Trap 1: `原消息` — an export placeholder treated as training material

When the exporter cannot resolve a quoted original message, the reply element's
`content` field is the **literal string `原消息`** (Chinese for "original message").

Result: **~128 reply examples (10% of the reply signal) had that literal string as
their "stimulus".**

The model dutifully learned "how to reply to the words *original message*" — pure
noise, occupying the most valuable signal budget.

Similar noise: `[图片]` (*image*), `[视频:...]` (*video*), bare `@mentions`,
and **"replying to myself"** (where the quoted sender is the target).

> **Lesson**: when building a dataset, apply an **explicit blocklist** to the
> stimulus field — don't just check for non-empty. Placeholders look like ordinary
> text but carry no meaning.

### Trap 2: Sharing one prompt between "reply" and "continue" breaks task separation

The original construction used **one system prompt and one user-turn shape** for
both tasks:

```
system: You are mimicking group member X's way of speaking...
user:   X: <my previous line>\nGroupmate: <what they said>
```

So the model learned the rule: "**the recent text is all things I said**."

The consequence showed up unmistakably at inference. Asked "钟馗?" (*Zhong Kui?*),
the model answered:

> **"What big sword and small sword were you asking about earlier?"**

**It fabricated a prior conversation that never existed.** Because in training the
"context" was almost always its own words, it learned to *pretend it had just said
something* rather than *answer the line in front of it*.

> **Lesson**: if two tasks are **different behaviours at inference time**, they must be
> **structurally distinguishable at training time**. Don't hope the model infers the
> task type from content — give it two different system prompts, and make
> "whose words go in which role" strictly correct.

### Trap 3: TRL silently discards hand-built labels

To make the loss apply only to the sentence the model should generate, the standard
approach is to build `labels` by hand and mark the prompt with `-100`.

**But `SFTTrainer`'s `truncate` step throws your `labels` away entirely** — so you
believe you're "learning only the reply" while actually training on the whole
conversation.

```python
dataset_kwargs={'skip_prepare_dataset': True}   # mandatory
```

> **Lesson**: after building labels by hand, **write a verification script** proving
> "the prompt is an exact prefix of the full sequence" and "some tokens really are
> supervised". See [`src/check_masking.py`](src/check_masking.py), which validates
> every single example. This class of bug never raises — it just makes results
> inexplicably bad.

### Trap 4: which region does the `<think>` block belong to?

Qwen3's chat template inserts a `<think></think>` block into assistant turns.
When building labels by hand it is easy to put it on the wrong side of the boundary.

**This one turned out to be a false alarm, but the method is worth recording.**
I looked up the `<think>` token id, searched for it inside the *supervised* span, and
got 0/11,606 hits — briefly concluding the template had changed.

In fact it wasn't missing: it was **correctly sitting inside the prompt region, masked
out**. The prompt rendered with `add_generation_prompt=True` already ends with
`<|im_start|>assistant\n<think>\n\n</think>\n\n`, and the quoted message sits
**after** it.

> **Lesson**: verification scripts should **print the actual text on both sides of the
> boundary** (with `repr()`), not just report found/not-found. A bare boolean will make
> you misjudge your own code.

### Trap 5: Chinese paths corrupt command-line tool arguments

Passing a Chinese directory name to `llama-quantize.exe` mangles the argument:

```
中文目录  →  Ĭ�Ϲ�����
```

**Fix**: create a pure-ASCII directory junction and use forward slashes everywhere:

```cmd
mklink /J C:\gguf D:\path\to\workspace
```

```bash
WS=/c/gguf
```

> **Lesson**: this path depends on the junction existing. Delete the junction and every
> script fails silently. Assert the junction exists; don't assume it.

### Aside: all three Ollama 0.40.2 deployment paths are dead

Worth listing, because it forces the whole export route:

| Attempt | Error |
|---|---|
| `ADAPTER` to load a LoRA directly | `LoRA adapters are no longer supported` |
| Import a safetensors directory | `MLX runtime is not available` (MLX is Apple-only) |
| `--quantize int4` | `quantize blob lm_head.weight: MLX init failed` |

**Conclusion: convert to GGUF via llama.cpp yourself.** See
[`src/export_gguf.sh`](src/export_gguf.sh).

---

## 4. The pipeline

```
QQ export JSONL
  └─ clean corpus (drop system msgs / media placeholders / bot commands)  src/clean_corpus.py
       └─ build two SFT sets (reply / continue, structurally separated)   src/build_sft_dataset.py
            └─ validate the masking contract (prefix + supervised)        src/check_masking.py
                 └─ QLoRA fine-tune (early stop + task-matched val set)   src/train_qlora.py
                      └─ merge LoRA into the base, shard by shard         src/merge_lora.py
                           └─ HF → GGUF (f16)                             src/export_gguf.sh
                                └─ llama-quantize → Q4_K_M
                                     └─ ollama create
```

### Two key decisions in dataset construction

**Decision 1: three tasks, three different prompts.**

| Task | system prompt | user turn | assistant turn |
|---|---|---|---|
| `reply` | "A groupmate just said something to you… **answer only that line; do not refer to anything you said before**" | the groupmate's quoted text (their words only) | my reply |
| `continue` | "Continue what you were just saying" | a real multi-turn slice, **with my lines in assistant turns** | my next line |

**The critical part**: the `reply` user turn contains **none of my own history**.
That is what kills the fabricated-prior-context failure mode at the root.

**Decision 2: build continuation samples from real multi-turn slices**, not by
concatenating messages into one blob.

Putting "my lines" in the `assistant` role and "their lines" in the `user` role teaches
the model "it's my turn to speak", rather than "all this recent text is mine".

### Merging LoRA: why shard by shard

Loading the whole 8B model in bfloat16 needs ~16 GB of RAM, which this machine
(31.7 GB, also running other things) cannot spare. [`src/merge_lora.py`](src/merge_lora.py)
processes **one safetensors shard at a time**:

```python
delta = (B @ A) * (lora_alpha / r)      # 32 / 16 = 2.0
sd[key].add_(delta.to(sd[key].dtype))
```

Peak memory is about "one shard (~4 GB) + one LoRA delta" instead of the whole model.

> ⚠️ **Never hard-code `lora_alpha / r`.**
> Read it from `adapter_config.json`. This run happened to be `r=16, alpha=32`
> (scale 2.0), but a hard-coded constant will **silently compute the wrong merge** for
> any other config — and the model still runs, it just sounds wrong.

---

## 5. Results

### Training configuration

| Item | Value |
|---|---|
| Base | Qwen3-8B |
| Method | QLoRA: 4-bit NF4 + double quantisation |
| LoRA | `r=16`, `alpha=32`, `dropout=0.05`, targets `q/k/v/o/gate/up/down_proj` |
| Trainable params | 43,646,976 (**0.5301%** of the model) |
| Sequence length | 768 |
| Optimizer | `paged_adamw_8bit` |
| Batch | `batch=1 × accum=8` |
| LR | `2e-4`, cosine, 3% warmup |

### Two versions compared

| Metric | v1 (400 steps) | v2 (600 cumulative) |
|---|---|---|
| Train loss | 3.51 | **2.05** |
| Train token accuracy | 45.4% | 64.6% |
| Val loss | **3.91** | **4.32** ⚠️ |
| Val token accuracy | 40.0% | 39.5% |

Look at the gap:

```
              train loss    val loss     gap
v1 (400 steps)   3.51   →    3.91      +0.40   <- healthy
v2 (+200 steps)  2.05   →    4.32      +2.27   <- memorising
```

200 more steps pushed train token accuracy from 45% to 65%, while the validation set
**did not move at all** (40.0% → 39.5%).

> **v2's `loss 2.05` is a flattering number that lies.**

More notable: v2 was **warm-started from v1**, so its starting train loss was already
~3.06 (≈ v1's converged value). It took only 200 steps and 0.25 of an epoch to break.
**Overfitting arrives much faster than you expect.**

### Base vs fine-tuned

| Prompt | Base Qwen3-8B | v1 | v2 |
|---|---|---|---|
| "I'll spank you" | Hahaha where? These are golden! | deer head | bring on a fight |
| "You're so bad at this" | I'd give that play full marks | so much idle chatter | oh I embarrassed myself |
| "Lend me money" | A hundred million? Interest first… | either way, just afraid you can't repay | sure |
| "Hiking tomorrow?" | Tomorrow? My legs are jelly.jpg — fine, let's go | go | works for me |

The base model is a **verbose, over-explaining assistant with exclamation marks**;
after fine-tuning it produces **two-to-four character fragments**, matching the
corpus's **median length of 8 characters**.

> **This is what the fine-tune is actually doing**: not "teaching it knowledge" but
> **transplanting the output-length distribution and the tone.**

### But loss does not measure "does it sound like them"

**This needs stating plainly**: loss measures "how accurately the next token is
predicted", **not** "does this sound like the person".

In real conversation v2 actually reads *more* like them ("spank you" → v2 replies
"bring on a fight", v1 replies "deer head"). **The final criterion should be whether a
human reads it as convincing, not the loss.** Both versions were kept for A/B testing.

---

## 6. The validation-set trap

**On review, this was the most serious methodological error — more fundamental than
the overfitting itself.**

The v1/v2 validation set composition:

| Component | Count | Share |
|---|---|---|
| Reply samples | 100 | 17% |
| Continuation samples | 400 | **83%** |

And the continuation user turns were **randomly stitched together from the same pool of
messages** — not a conversation prefix that ever actually existed.

**So 83% of that validation set was noise, measuring a scenario that does not exist in
the real task.**

Two consequences, both bad:

1. **`eval_loss` loses its meaning** — whether the loss on a random-noise set is 3.9 or
   4.3 has no stable relationship to "how well the model answers people".
2. **The "v2 overfits" conclusion had the right direction and the wrong evidence.**
   My reasoning rested on "val loss went up = overfitting", but if the validation set is
   itself noise, you cannot separate how much of that rise is overfitting and how much
   is noise.

### The correct approach

**The validation set must match the inference-time task.** This model's job is
"a groupmate says something, how should I answer", so the validation set contains
**reply samples only**:

```python
val_ds = reply_val                      # reply task only
val_ds = rv.shuffle(seed=42)            # never take a sequential slice
```

Combined with:

- **early stopping** on `eval_loss`, instead of blindly training a fixed step count;
- **keeping every checkpoint** and selecting by validation loss, not taking the last step;
- **`load_best_model_at_end=True`**.

> **Lesson**: a validation set is not "just a bit of held-out data".
> **It is your definition of *good*.** Define a metric that doesn't match the real task,
> and every judgement you make afterwards chases a phantom.

---

## 7. Repository layout

```
.
├── README.md              English (this file)
├── README.zh-CN.md        Chinese
├── requirements.txt
├── .gitignore             <- the important one: keeps corpus and weights out
├── src/
│   ├── config.py          identity/paths all from env vars (no hard-coded uin)
│   ├── clean_corpus.py    corpus cleaning
│   ├── build_sft_dataset.py  two-set construction (structurally separated)
│   ├── check_masking.py   masking-contract verification
│   ├── train_qlora.py     QLoRA training (early stop + task-matched val set)
│   ├── merge_lora.py      shard-by-shard LoRA merge
│   ├── export_gguf.sh     HF → GGUF → quantise → ollama create
│   └── eval_adapters.py   one base load, many adapters A/B
└── docs/
    ├── pitfalls.md        pitfall details
    └── vram.md            VRAM diagnostics
```

---

## 8. Reproducing this

### Environment

```
torch 2.6.0+cu124      transformers 4.51.3   peft 0.15.2
trl 0.15.2             accelerate 1.5.2       bitsandbytes 0.45.3
datasets 3.5.0         numpy 2.2.6            sentencepiece 0.2.2
```

### Identity comes from environment variables

There is **no hard-coded uin or nickname anywhere in the code**:

```bash
export TARGET_UIN="<your uin>"
export TARGET_NAME="<your group nickname>"
export TARGET_GROUP="<group name>"
export BASE_MODEL="D:/models/Qwen3-8B"
export EXPORTS_DIR="C:/path/to/qq/exports"
```

### Training

```bash
export HF_HOME="D:/models/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OUT_DIR="./my-lora"
export EPOCHS=1 MAX_LEN=768 BATCH=1 GRAD_ACCUM=8
python -u src/train_qlora.py
```

### First, check whether your data is even enough

```bash
python src/check_masking.py      # verify the labels contract
```

**Run this before training.** It tells you whether the prompt is an exact prefix of the
full sequence and how many tokens are actually supervised. This class of bug doesn't
raise — it just quietly ruins your results.

---

## 9. Why no corpus

This is not an oversight; it's a deliberate decision.

1. **The corpus is other people's private conversation.** Group members never agreed to
   be published. Even though the export contained only my own messages, **the reply
   quotes still contain other people's exact words and nicknames.**
2. **Posting real dialogue as "style examples" means publishing someone's private life.**
   Group chat contains education history, career plans, purchasing decisions, personal
   relationships. Even redacted, the context itself leaks a great deal.
3. **The fine-tuned weights are a lossy compression of the corpus.** So the weights
   stay private too.

**What is published is the method.** Those traps — VRAM paging, placeholder leakage,
validation-set mismatch, discarded labels — have nothing to do with anyone's privacy,
and are rarely explained properly in Chinese-language material. That is the part worth
passing on.

If you attempt something similar, copy the `.gitignore` and do it **before your first
commit**.

---

## License

Code MIT. Documentation CC BY 4.0.
