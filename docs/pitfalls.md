# Pitfalls, in detail

Companion to the [README](../README.md). Each entry is something that cost real
time and that a generic tutorial will not warn you about.

---

## 1. VRAM paging disguises itself as "training is working"

**Symptom**

```
GPU utilisation : 100%
Power draw      : 54-67 W          <- the only clue
Seconds/step    : 20-66 s
```

**Cause** — VRAM was 97% full. Windows WDDM paged VRAM out to system RAM, so the
compute units sat idle waiting on memory. Utilisation stayed pinned at 100%
because the GPU was *busy waiting*, not computing.

**Diagnostic** — always read utilisation and power draw together:

| Utilisation | Power | Meaning |
|---|---|---|
| ~100% | ~TDP | genuinely computing |
| ~100% | far below TDP | **paging / memory-starved** |
| low | low | idle or input-bound |

For this GPU, healthy training drew 110-126 W; paging drew 54-67 W.

**Fix**

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 8
optim = "paged_adamw_8bit"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
```

> Note: on some Windows builds PyTorch warns
> `expandable_segments not supported on this platform`. Keep the setting anyway;
> it is harmless when unsupported, and helps where it is supported.

**Corollary: headroom beats peak throughput.** `batch=2 x accum=4` was 2.1x
faster per step but left ~155 MB free, and step time then degraded 5.6 s -> 8.4 s
over 25 steps. It was already sliding into paging. A VRAM-saturated config looks
fine at the start and rots during the run.

---

## 2. The exporter's `原消息` placeholder is a data leak

When the chat exporter cannot resolve a quoted message, the reply element's
`content` is the **literal string `原消息`**. It looks like normal text.

If you build "reply" examples from quote elements without a blocklist, roughly
**10% of your most valuable signal** becomes "here is how I respond to the words
*original message*".

Other stubs seen in the same export:

| Stub | Count |
|---|---|
| `[图片]` (*image*) | 235 |
| `原消息` (*original message*) | 118 |
| quoting yourself | 7 |
| bare `@someone` | 3 |

**Rule**: validate the *stimulus* against an explicit blocklist, not just
"is it non-empty". Placeholders are syntactically fine and semantically empty.

---

## 3. One prompt for two tasks teaches the model to lie about the past

The original construction used a single system prompt and a single user-turn
shape for both tasks:

```
system: You are mimicking group member X's way of speaking...
user:   X: <my previous line>\nGroupmate: <what they said>
```

Both tasks looked identical to the model. And because the "context" consisted of
the persona's own lines, the model learned: **recent text = things I said**.

At inference, asked a simple question, it replied:

> "What big sword and small sword were you asking about earlier?"

There was no earlier. It had learned to perform having said something, rather
than to answer the line in front of it.

**Fix** — two prompts, two shapes:

| Task | system | user turn | assistant |
|---|---|---|---|
| reply | "answer only that line; do not refer to anything you said before" | the groupmate's words **only** | my reply |
| continue | "continue what you were just saying" | a real slice, my lines in **assistant** turns | my next line |

The critical change is **excluding the persona's own history from the reply user
turn**. That is what removes the incentive to fabricate.

---

## 4. TRL silently discards hand-built labels

To supervise only the generated answer, you mask the prompt with `-100`. But
`SFTTrainer`'s `truncate` step **throws your `labels` away**, so you train on the
whole conversation while believing you do not.

```python
dataset_kwargs={'skip_prepare_dataset': True}   # required
```

This failure is silent. Always verify — [`check_masking.py`](../src/check_masking.py)
checks, on every example:

1. the chat-templated prompt is an **exact token prefix** of the full sequence;
2. at least one token is supervised;
3. reports how many examples are truncated or fully masked.

---

## 5. The `<think>` episode: a verification script that lied

Qwen3's template puts a `<think></think>` block in assistant turns. Building
labels by hand, it is easy to put the boundary in the wrong place.

I probed for the `<think>` token inside the supervised span and got **0/11,606
hits**, and briefly concluded the template had changed.

It had not. The block sits in the **prompt** region, correctly masked, because the
prompt rendered with `add_generation_prompt=True` already ends with:

```
<|im_start|>assistant\n<think>\n\n</think>\n\n
```

and the quoted message comes **after** it.

**Lesson**: print the actual text on both sides of the boundary with `repr()`.
A bare found/not-found boolean will make you misjudge your own code. See the
`prompt region ends : ...` / `supervised text : ...` output in
`check_masking.py`.

---

## 6. Non-ASCII paths corrupt command-line tool arguments

Passing a Chinese directory name to `llama-quantize.exe`:

```
中文目录  ->  Ĭ�Ϲ�����
```

**Fix** — an ASCII junction plus forward slashes:

```cmd
mklink /J C:\gguf D:\path\to\workspace
```

```bash
WS=/c/gguf
```

**This is a dependency, not a convenience.** Delete the junction and every script
fails. `export_gguf.sh` asserts it exists and tells you how to recreate it.

---

## 7. Ollama 0.40.2 rejects every direct LoRA route

| Attempt | Error |
|---|---|
| `ADAPTER` directive with a LoRA | `LoRA adapters are no longer supported` |
| import a safetensors directory | `MLX runtime is not available` (MLX is Apple-only) |
| `--quantize int4` | `quantize blob lm_head.weight: MLX init failed` |

The only working path is **llama.cpp yourself**: merge -> `convert_hf_to_gguf.py`
-> `llama-quantize` -> `ollama create`.

---

## 8. Merging LoRA: two ways to silently corrupt it

### 8a. Never hard-code the scale

```python
delta = (B @ A) * (lora_alpha / r)
```

Read `lora_alpha` and `r` from `adapter_config.json`. A stale constant produces a
model that **still runs** and merely sounds less like the target — the worst kind
of bug. `merge_lora.py` reads the file and refuses to proceed without it.

### 8b. `B @ A` is only valid at small rank

If your adapter rank exceeds the base model's number of key/value heads
(Qwen3-8B: 8), the contraction is **genuinely ambiguous** and the merged weights
are wrong.

This run used `r=16` against 8 KV heads. `merge_lora.py` therefore **refuses**
rather than writing a plausible-looking wrong merge:

```
adapter rank r=16 exceeds the base model's KV head count (8).
`B @ A` is not a correct merge at this rank.
```

If you hit this, either pass the real `KV_HEADS` once you have verified the
architecture, or use a framework merge (`peft`'s `merge_and_unload`, which
handles the reshape) instead of this script.

### 8c. Why shard by shard

Peak RAM matters: loading 8B in bfloat16 needs ~16 GB. Processing one
safetensors shard at a time keeps peak at "one shard (~4 GB) + one delta".

---

## 9. Continuation samples: do not concatenate messages into one blob

Putting the persona's lines into the **user** turn as a multi-line blob teaches
"all this text is mine". Putting them in **assistant** turns and the groupmates'
lines in **user** turns teaches "it is my turn to speak".

Same data, different role assignment, materially different behaviour.
