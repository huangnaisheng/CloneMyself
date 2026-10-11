# -*- coding: utf-8 -*-
"""QLoRA fine-tune of Qwen3-8B on one person's chat style.

Each non-obvious choice here exists because of a measured failure, not taste:

  * reply and continue use different system prompts and different turn shapes
    (see build_sft_dataset.py), so the model can tell "a groupmate just spoke
    to me" from "continue my own monologue".  Sharing one prompt is what
    produced replies that invented prior conversation.
  * eval_dataset is the REPLY split only, because the real use of this model is
    "someone says something, what do I answer".  The earlier setup evaluated
    mostly-random stitched continuation samples, which measured a scenario that
    does not exist at inference time.
  * early stopping + load_best_model_at_end pick the checkpoint by validation
    loss.  The final step is not automatically the best one; a run that added
    200 steps past convergence drove val loss 3.91 -> 4.32 while train loss
    fell to 2.05.
  * MAX_LEN 512 rather than 768: the rebuilt examples top out around 443
    tokens, so the larger value only paid for padding.
  * batch=1 x accum=8 rather than batch=2 x accum=4.  The latter is ~2x faster
    but leaves ~155 MB of VRAM and then degrades mid-run as WDDM starts paging.

Use WARM_START=<adapter dir> to continue from an existing adapter with a fresh
optimizer.  Do NOT use RESUME: paged optimizer state does not restore cleanly.
"""
import collections
import inspect
import os
import random
import sys

sys.stdout.reconfigure(encoding='utf-8')
import torch  # noqa: E402
from datasets import concatenate_datasets, load_dataset  # noqa: E402
from transformers import (AutoTokenizer, AutoModelForCausalLM,  # noqa: E402
                          BitsAndBytesConfig, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments, EarlyStoppingCallback)
from peft import (LoraConfig, get_peft_model,  # noqa: E402
                  prepare_model_for_kbit_training)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('DATASET_OUT', HERE)
OUT = os.environ.get('OUT_DIR', os.path.join(HERE, 'lora-out'))
MAX_LEN = int(os.environ.get('MAX_LEN', '512'))
MAX_STEPS = int(os.environ.get('MAX_STEPS', '300'))
LR = float(os.environ.get('LR', '5e-5'))
BATCH = int(os.environ.get('BATCH', '1'))
GRAD_ACCUM = int(os.environ.get('GRAD_ACCUM', '8'))
REPLY_REPEAT = int(os.environ.get('REPLY_REPEAT', '3'))
MAX_CONTINUE = int(os.environ.get('MAX_CONTINUE', '2500'))
EVAL_STEPS = int(os.environ.get('EVAL_STEPS', '25'))
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', '25'))
PATIENCE = int(os.environ.get('PATIENCE', '5'))
GRAD_CKPT = os.environ.get('GRAD_CKPT', '1') == '1'
WARM_START = os.environ.get('WARM_START', '')
OPTIM = os.environ.get('OPTIM', 'paged_adamw_8bit')
SMOKE = int(os.environ.get('SMOKE_STEPS', '0'))

random.seed(42)
if not C.BASE_MODEL:
    raise SystemExit('set BASE_MODEL')
print(f'=== MAX_LEN={MAX_LEN} MAX_STEPS={MAX_STEPS} LR={LR} '
      f'BATCH={BATCH}x{GRAD_ACCUM} REPLY_REPEAT={REPLY_REPEAT} '
      f'MAX_CONTINUE={MAX_CONTINUE} WARM_START={WARM_START or "(none)"} ===')


def load(name):
    return load_dataset('json', data_files=os.path.join(DATA, name), split='train')


rt = load('v2_reply_train.jsonl')
rv = load('v2_reply_val.jsonl')
ct = load('v2_continue_train.jsonl')
print('reply    train/val:', len(rt), len(rv))
print('continue train    :', len(ct))

if MAX_CONTINUE and len(ct) > MAX_CONTINUE:
    ct = ct.shuffle(seed=42).select(range(MAX_CONTINUE))

train_ds = concatenate_datasets([rt] * max(1, REPLY_REPEAT) + [ct]).shuffle(seed=42)
val_ds = rv.shuffle(seed=42)          # task-matched: reply only
print('final train       :', len(train_ds))

if SMOKE:
    train_ds = train_ds.select(range(min(SMOKE * 2, len(train_ds))))
    val_ds = val_ds.select(range(min(8, len(val_ds))))

tok = AutoTokenizer.from_pretrained(C.BASE_MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_stats = collections.Counter()


def to_tokens(row):
    msgs = row['messages']
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                     add_generation_prompt=True,
                                     enable_thinking=False)
    full = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=False,
                                   enable_thinking=False)
    p_ids = tok(prompt, add_special_tokens=False)['input_ids']
    f_ids = tok(full, add_special_tokens=False)['input_ids']
    n = len(p_ids)
    if f_ids[:n] != p_ids:
        _stats['prompt_not_prefix'] += 1
        n = 0
    f_ids = f_ids[:MAX_LEN]
    n = min(n, len(f_ids))
    labels = [-100] * n + f_ids[n:]
    if all(x == -100 for x in labels):
        _stats['fully_masked'] += 1
        return {'input_ids': [], 'attention_mask': [], 'labels': []}
    _stats['ok'] += 1
    return {'input_ids': f_ids, 'attention_mask': [1] * len(f_ids), 'labels': labels}


train_ds = train_ds.map(to_tokens, remove_columns=train_ds.column_names,
                        desc='tokenizing train')
val_ds = val_ds.map(to_tokens, remove_columns=val_ds.column_names,
                    desc='tokenizing val')
keep = lambda r: len(r['input_ids']) > 0  # noqa: E731
train_ds = train_ds.filter(keep)
val_ds = val_ds.filter(keep)
print('tokenize stats:', dict(_stats))
print('after filter train/val:', len(train_ds), len(val_ds))

for i in range(min(2, len(train_ds))):
    ex = train_ds[i]
    sup = sum(1 for x in ex['labels'] if x != -100)
    print(f'  sample {i}: {len(ex["input_ids"])} tokens, {sup} supervised')

print('=== loading base model in 4-bit ===')
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                         bnb_4bit_compute_dtype=compute_dtype,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    C.BASE_MODEL, quantization_config=bnb, device_map={'': 0},
    trust_remote_code=True, attn_implementation='sdpa', low_cpu_mem_usage=True)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRAD_CKPT)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias='none',
                  task_type='CAUSAL_LM',
                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                  'gate_proj', 'up_proj', 'down_proj'])
if WARM_START:
    from peft import PeftModel
    print('warm start from adapter:', WARM_START)
    model = PeftModel.from_pretrained(model, WARM_START, is_trainable=True)
else:
    model = get_peft_model(model, lora)
model.print_trainable_parameters()

args_kwargs = dict(
    output_dir=OUT,
    max_steps=SMOKE if SMOKE else MAX_STEPS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    logging_steps=5,
    save_strategy='steps',
    save_steps=SAVE_STEPS,
    save_total_limit=6,
    eval_strategy='steps',
    eval_steps=EVAL_STEPS,
    warmup_ratio=0.05,
    lr_scheduler_type='cosine',
    bf16=(compute_dtype == torch.bfloat16),
    fp16=(compute_dtype == torch.float16),
    gradient_checkpointing=GRAD_CKPT,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    optim=OPTIM,
    max_grad_norm=0.3,
    report_to='none',
    dataloader_num_workers=0,
    seed=42,
    remove_unused_columns=False,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)
sig = inspect.signature(TrainingArguments.__init__).parameters
args = TrainingArguments(**{k: v for k, v in args_kwargs.items() if k in sig})

trainer = Trainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tok, padding=True,
                                         label_pad_token_id=-100,
                                         pad_to_multiple_of=8),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE,
                                     early_stopping_threshold=0.01)],
)
print('=== training ===')
trainer.train()

print('=== saving best (by eval_loss) ===')
trainer.save_model(OUT)
tok.save_pretrained(OUT)
print('best_checkpoint :', trainer.state.best_model_checkpoint)
print('best_metric     :', trainer.state.best_metric)
print('saved to', OUT)
