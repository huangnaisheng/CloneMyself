# -*- coding: utf-8 -*-
"""Clean a single-speaker chat export down to usable text messages.

Walks every chunked export of the configured group, drops everything that is
not the persona's own plain text, and writes a normalised JSONL plus a report.

The report matters: the median message length is the number the fine-tune
actually learns to match, so measure it rather than assuming.

Usage:  python src/clean_corpus.py
"""
import collections
import glob
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')

REPLY_PREFIX_RE = re.compile(r'^\[回复消息\]\s*@\S+\s*')
MEDIA_RE = re.compile(r'^\[(?:图片|视频|文件|语音|动画表情|转发|JSON消息|Markdown消息)')


def iter_group_records():
    pattern = os.path.join(C.EXPORTS_DIR, '*_chunked_jsonl', 'chunks', '*.jsonl')
    for jl in glob.glob(pattern):
        manifest = os.path.join(os.path.dirname(os.path.dirname(jl)), 'manifest.json')
        if os.path.exists(manifest):
            with open(manifest, encoding='utf-8') as f:
                group = (json.load(f).get('chatInfo') or {}).get('name')
            if group != C.TARGET_GROUP:
                continue
        with open(jl, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def clean_text(s):
    if not isinstance(s, str):
        return ''
    for ch in ('\u200b', '\u2067', '\u2069'):
        s = s.replace(ch, '')
    s = s.replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', s).strip()


def main():
    C.require_identity()
    out_path = os.environ.get('CLEAN_OUT', os.path.join(
        os.environ.get('DATASET_OUT', '.'), 'clean_messages.jsonl'))

    kept, dropped = [], collections.Counter()
    seen = set()
    dups = 0

    for m in iter_group_records():
        if m.get('system'):
            dropped['system message'] += 1
            continue
        if m.get('type') not in ('text', 'reply'):
            dropped[f"non-text ({m.get('type')})"] += 1
            continue
        s = m.get('sender') or {}
        raw = (m.get('content') or {}).get('text', '')
        body = (REPLY_PREFIX_RE.sub('', clean_text(raw))
                if m.get('type') == 'reply' else clean_text(raw))
        if not body:
            dropped['empty'] += 1
            continue
        if MEDIA_RE.match(body):
            dropped['media placeholder'] += 1
            continue
        if len(body) > C.MAX_TARGET_CHARS:
            dropped['too long'] += 1
            continue
        if body in seen:
            dups += 1
            continue
        seen.add(body)
        kept.append({
            'ts': m.get('timestamp') or 0,
            'time': m.get('time'),
            'name': s.get('name'),
            'uin': s.get('uin'),
            'is_target': s.get('uin') == C.TARGET_UIN and s.get('name') == C.TARGET_NAME,
            'text': body,
        })

    kept.sort(key=lambda r: r['ts'])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    lens = [len(r['text']) for r in kept]
    report = {
        'kept': len(kept),
        'duplicates_removed': dups,
        'dropped': dict(dropped),
        'median_chars': statistics.median(lens) if lens else 0,
        'mean_chars': round(statistics.mean(lens), 1) if lens else 0,
        'p90_chars': sorted(lens)[int(len(lens) * 0.9)] if lens else 0,
        'time_range': [kept[0]['time'], kept[-1]['time']] if kept else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f'\nwrote {out_path}')
    print('\nThe median character count is what the fine-tune will learn to match. '
          'A base instruct model will ignore it and write paragraphs.')


if __name__ == '__main__':
    main()
