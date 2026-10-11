# -*- coding: utf-8 -*-
"""Merge a LoRA adapter into the base model, one safetensors shard at a time.

Why shard by shard: loading the whole 8B model in bfloat16 needs ~16 GB of RAM.
Peak here is about "one shard (~4 GB) + one LoRA delta".

Two correctness details that are easy to get silently wrong:

  * the scale is read from adapter_config.json (`lora_alpha / r`), never
    hard-coded.  A stale constant still produces a runnable model that simply
    sounds wrong.
  * `delta = B @ A` is only valid when the adapter rank is <= the base model's
    number of key/value heads (Qwen3-8B has 8).  At rank 16 the contraction is
    genuinely ambiguous, so we refuse to guess instead of writing a
    plausible-looking but wrong merge.

Env: ADAPTER_FILE, MERGED_DIR, BASE_MODEL
"""
import json
import os
import shutil
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

PREFIX = 'base_model.model.'
KV_HEADS = int(os.environ.get('KV_HEADS', '8'))  # Qwen3-8B default


def read_scale(adapter_file):
    """Return (scale, r, alpha) from the adapter's own config. Never guess."""
    cfg_path = os.path.join(os.path.dirname(adapter_file), 'adapter_config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
        r = cfg.get('r')
        alpha = cfg.get('lora_alpha')
        if r and alpha:
            return alpha / r, r, alpha
    raise SystemExit(
        f'cannot determine lora_alpha / r: {cfg_path} missing or incomplete. '
        'Refusing to merge with a guessed scale.')


def build_deltas(adapter_file):
    with torch.no_grad():
        ad = load_file(adapter_file)
    pairs = {}
    for k in ad:
        if k.endswith('.lora_A.weight'):
            stem = k[:-len('.lora_A.weight')]
            pairs.setdefault(stem[len(PREFIX):] + '.weight', {})['A'] = ad[k]
        elif k.endswith('.lora_B.weight'):
            stem = k[:-len('.lora_B.weight')]
            pairs.setdefault(stem[len(PREFIX):] + '.weight', {})['B'] = ad[k]
    print(f'adapter modules found: {len(pairs)}')
    return pairs


def delta_for(pair, scale, key):
    A = pair['A'].to(torch.float32)
    B = pair['B'].to(torch.float32)
    r = A.shape[0]
    if r > KV_HEADS:
        raise SystemExit(
            f'{key}: adapter rank r={r} exceeds the base model\'s KV head count '
            f'({KV_HEADS}). `B @ A` is not a correct merge at this rank. '
            'Raise KV_HEADS if you really know the architecture, or use a '
            'framework merge instead of this script.')
    return (B @ A) * scale


def main():
    t0 = time.time()
    adapter_file = os.environ.get('ADAPTER_FILE', '')
    if not adapter_file or not os.path.exists(adapter_file):
        raise SystemExit('set ADAPTER_FILE to an existing adapter_model.safetensors')
    base = C.BASE_MODEL
    if not base:
        raise SystemExit('set BASE_MODEL')
    out = os.environ.get('MERGED_DIR', os.path.join(os.path.dirname(base), 'merged'))

    scale, r, alpha = read_scale(adapter_file)
    print(f'LoRA r={r} alpha={alpha} -> scale={scale:g}')
    pairs = build_deltas(adapter_file)

    with open(os.path.join(base, 'model.safetensors.index.json'), encoding='utf-8') as f:
        index = json.load(f)
    shards = sorted(set(index['weight_map'].values()))
    print('shards to process:', shards)
    os.makedirs(out, exist_ok=True)

    merged = 0
    for si, shard in enumerate(shards, 1):
        print(f'[{si}/{len(shards)}] {shard} ...', flush=True)
        with torch.no_grad():
            sd = load_file(os.path.join(base, shard))
            for key in list(sd.keys()):
                pair = pairs.get(key)
                if not pair or 'A' not in pair or 'B' not in pair:
                    continue
                d = delta_for(pair, scale, key)
                if d.shape != sd[key].shape:
                    print(f'    shape mismatch, skipped: {key} '
                          f'{tuple(d.shape)} vs {tuple(sd[key].shape)}')
                    continue
                sd[key].add_(d.to(sd[key].dtype))
                merged += 1
            save_file(sd, os.path.join(out, shard), metadata={'format': 'pt'})
            del sd
        print(f'    done ({time.time() - t0:.0f}s elapsed)', flush=True)

    for name in os.listdir(base):
        if name.endswith('.safetensors'):
            continue
        src = os.path.join(base, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, name))

    # The base config can point at a machine- or user-specific path. Rewrite it
    # so the merged directory is self-contained and portable.
    cfg_path = os.path.join(out, 'config.json')
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding='utf-8') as f:
            cfg = json.load(f)
        old = cfg.get('_name_or_path')
        cfg['_name_or_path'] = base.replace('\\', '/')
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print(f'config.json _name_or_path: {old!r} -> {cfg["_name_or_path"]!r}')

    print(f'merged modules: {merged} (expected {len(pairs)})')
    print('output:', out)
    print(f'total time: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
