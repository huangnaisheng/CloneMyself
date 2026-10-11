# -*- coding: utf-8 -*-
"""A/B several LoRA adapters (and the base model) on ONE base-model load.

Merging to GGUF + quantising + `ollama create` costs ~40 minutes per candidate,
so compare candidates here first, directly on the adapters.  Only the winner
needs the expensive export.

The holdout is the reply val split: a groupmate says something and the model
must produce the persona's real answer.

Two prompt styles are reported, because they separate two different problems:

  native   the style the adapter was trained with
  new      the corrected reply prompt, applied to every model

If an adapter looks bad under `new` but fine under `native`, it does not
understand the corrected prompt -- it is not necessarily a bad model.

Automatic signals (lower is better except overlap):
  empty      produced nothing
  too_long   exceeded MAX_ANSWER_CHARS: fell back to assistant verbosity
  overlap0   zero character overlap with the stimulus (ignored the input)
  prior_ref  mentions "earlier / a moment ago / what I said" -- the tell for
             fabricated prior conversation

Usage:
  python src/eval_adapters.py --adapter v1=path/to/ckpt --adapter v2=path/to/ckpt \
                              --adapter base
"""
import argparse
import collections
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

MAX_ANSWER_CHARS = int(os.environ.get('MAX_ANSWER_CHARS', '60'))
CJK = re.compile(r'[\u4e00-\u9fff]')
PRIOR_REF = re.compile(r'前面|刚才|我之前|我刚说|我不是说|刚不是|earlier|a moment ago')

SYSTEM_OLD = (
    "你在一个中文QQ群里聊天。你要模仿群成员“{name}”的说话方式："
    "句子通常很短，口语化，常用网络用语，会吐槽、会顺嘴接梗，也可能只是随口一说。"
    "只模仿语气和表达习惯，不要编造他的经历、身份或记忆。"
    "直接输出要发到群里的内容，不要解释，不要加引号，不要输出分析。"
)


def build_messages(stim, style):
    """`stim` is a raw user turn like 'GroupmateName：text'."""
    if style == 'new':
        m = re.match(r'^([^：:]{1,20})[：:](.*)$', stim, re.S)
        neighbour = f'{m.group(1)}：{m.group(2)}' if m else stim
        return [{'role': 'system', 'content': C.reply_prompt()},
                {'role': 'user', 'content': neighbour}]
    return [{'role': 'system', 'content': SYSTEM_OLD.format(name=C.TARGET_NAME)},
            {'role': 'user', 'content': stim}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', default=os.path.join(
        os.environ.get('DATASET_OUT', '.'), 'v2_reply_val.jsonl'))
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--adapter', action='append', default=[],
                    help='NAME=PATH; use NAME alone (no =) for the raw base model')
    ap.add_argument('--out', default='eval_adapters_results.json')
    ap.add_argument('--seed', type=int, default=1234)
    a = ap.parse_args()

    if not a.adapter:
        ap.error('need at least one --adapter NAME=PATH')
    if not C.BASE_MODEL:
        raise SystemExit('set BASE_MODEL')

    models = {}
    for spec in a.adapter:
        name, sep, path = spec.partition('=')
        models[name] = path if sep else None

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    rows = [json.loads(l) for l in open(a.val, encoding='utf-8') if l.strip()]
    step = max(1, len(rows) // a.n)
    rows = rows[::step][:a.n]
    print(f'held-out reply examples: {len(rows)}')

    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    print('=== loading base model once ===')
    model = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL, quantization_config=bnb, device_map={'': 0},
        trust_remote_code=True, attn_implementation='sdpa')

    names = []
    for name, path in models.items():
        if path:
            model = PeftModel.from_pretrained(model, path, adapter_name=name)
            names.append(name)
            print('  attached adapter', name, '<-', path)
    model.eval()

    def generate(messages):
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      enable_thinking=False,
                                      return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=48, do_sample=True,
                                 temperature=0.85, top_p=0.9, top_k=40,
                                 repetition_penalty=1.12,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    torch.manual_seed(a.seed)
    records, stats = [], collections.defaultdict(collections.Counter)

    for i, row in enumerate(rows, 1):
        stim = row['messages'][-2]['content']
        gold = row['messages'][-1]['content']
        print(f'--- [{i}/{len(rows)}] {row.get("time", "")}')
        print(f'  stimulus : {stim}')
        print(f'  real     : {gold}')
        rec = {'time': row.get('time'), 'stimulus': stim, 'gold': gold, 'preds': {}}

        for name, path in models.items():
            styles = ['new'] if path is None else ['native', 'new']
            for style in styles:
                key = f'{name}[{style}]'
                if names and path is not None:
                    model.set_adapter(name)      # also re-enables adapter layers
                elif names:
                    model.disable_adapter()
                try:
                    pred = generate(build_messages(stim, style)).replace('\n', ' / ')
                except Exception as e:           # keep the sweep going
                    pred = f'<ERROR {e}>'
                rec['preds'][key] = pred

                st = stats[key]
                st['n'] += 1
                if not pred or pred.startswith('<ERROR'):
                    st['empty'] += 1
                if len(pred) > MAX_ANSWER_CHARS:
                    st['too_long'] += 1
                if not (set(CJK.findall(pred)) & set(CJK.findall(stim))):
                    st['overlap0'] += 1
                if PRIOR_REF.search(pred):
                    st['prior_ref'] += 1
                print(f'  {key:18s} : {pred}')
        print()
        records.append(rec)

    print('=' * 76)
    print(f'AUTOMATIC SIGNALS (n={len(rows)}; lower is better)')
    print(f'{"model[style]":22s} {"empty":>6s} {"too_long":>9s} {"overlap0":>9s} {"prior_ref":>10s}')
    for key, st in stats.items():
        print(f'{key:22s} {st["empty"]:>6d} {st["too_long"]:>9d} '
              f'{st["overlap0"]:>9d} {st["prior_ref"]:>10d}')

    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump({'stats': {k: dict(v) for k, v in stats.items()},
                   'records': records}, f, ensure_ascii=False, indent=2)
    print(f'\nwrote {a.out}')
    print('\nThese signals are a screen, not a verdict. The final decision is '
          'whether a human reads the replies as sounding like the person.')


if __name__ == '__main__':
    main()
