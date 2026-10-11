# -*- coding: utf-8 -*-
"""Verify the labels contract on every example BEFORE training.

Two things must hold or the training signal is wrong, and neither failure
raises an exception -- they just make results inexplicably bad:

  1. the chat-templated prompt must be an EXACT token prefix of the full
     conversation, otherwise hand-built labels do not line up with the text;
  2. something must actually be supervised, and it must be the persona's own
     message (the template's empty <think> block legitimately stays masked
     inside the prompt).

This script also prints the text on both sides of the boundary with repr(),
because a bare found/not-found boolean is how you end up misjudging your own
code (see the <think> episode in the README).

Usage:  python src/check_masking.py [dataset_dir]
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

MAX_LEN = int(os.environ.get('MAX_LEN', '512'))
FILES = ['v2_reply_train.jsonl', 'v2_reply_val.jsonl', 'v2_continue_train.jsonl']


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('DATASET_OUT', '.')
    if not C.BASE_MODEL:
        raise SystemExit('set BASE_MODEL first')
    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL, trust_remote_code=True)

    for fname in FILES:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f'{fname}: not found, skipped')
            continue

        n = prefix_ok = supervised_ok = fully_masked = truncated = 0
        showed = 0
        for line in open(path, encoding='utf-8'):
            if not line.strip():
                continue
            msgs = json.loads(line)['messages']
            prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)
            full = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False,
                                           enable_thinking=False)
            p = tok(prompt, add_special_tokens=False)['input_ids']
            fid = tok(full, add_special_tokens=False)['input_ids']
            n += 1

            k = len(p) if fid[:len(p)] == p else 0
            prefix_ok += 1 if k else 0
            if len(fid) > MAX_LEN:
                truncated += 1
            fid = fid[:MAX_LEN]
            k = min(k, len(fid))
            labels = [-100] * k + fid[k:]

            if all(x == -100 for x in labels):
                fully_masked += 1
                continue
            supervised_ok += 1

            if showed < 2:
                showed += 1
                sup = len(fid) - k
                print(f'\n  --- {fname} example {showed} ({len(fid)} tokens, '
                      f'{sup} supervised)')
                print(f'      prompt region ends : {tok.decode(fid[:k])[-50:]!r}')
                print(f'      supervised text    : {tok.decode(fid[k:])!r}')

        print(f'\n{fname}: n={n}')
        print(f'   prompt is exact prefix : {prefix_ok}/{n}')
        print(f'   has supervised labels  : {supervised_ok}/{n}')
        print(f'   fully masked (BAD)     : {fully_masked}')
        print(f'   truncated at {MAX_LEN}     : {truncated}')

    print('\nAll checks pass means the labels line up with the text. '
          'If "fully masked" is non-zero, those examples contribute no gradient.')


if __name__ == '__main__':
    main()
