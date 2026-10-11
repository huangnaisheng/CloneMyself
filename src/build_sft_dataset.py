# -*- coding: utf-8 -*-
"""Build a two-task SFT dataset from a single-speaker chat export.

Design notes (these are the fixes for the failure modes documented in the
README -- read section 3 before changing anything here):

  reply     A groupmate spoke to me.  The stimulus is the quoted message from
            the reply element, together with at most one preceding groupmate
            line.  The persona's OWN history is deliberately absent from the
            user turn: including it is what taught the model to invent prior
            conversation.

  continue  A real multi-turn slice of the chat: the persona's lines are
            assistant turns, other members' lines are user turns, and the
            target message closes it.

Identity is read from the environment (see config.py); no uin, nickname or
group name is hard-coded.
"""
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')

REPLY_PREFIX_RE = re.compile(r'^\[回复消息\]\s*@\S+\s*')

# The exporter emits these literals when it cannot resolve a quoted message.
# Treating them as real stimuli is the single worst data bug found here: they
# looked like ordinary text, so ~10% of the reply signal was pure noise.
QUOTE_PLACEHOLDERS = {
    '原消息', '原消息：', '@原消息', '消息', '...', '原文',
}


def clean_text(s):
    if not isinstance(s, str):
        return ''
    for ch in ('\u200b', '\u2067', '\u2069'):
        s = s.replace(ch, '')
    s = s.replace('\xa0', ' ')
    return re.sub(r'\s+', ' ', s).strip()


def is_media_placeholder(t):
    """`[图片]`, `[视频:...]`, `[回复消息]...` and friends carry no content."""
    return bool(re.match(
        r'^\[(?:图片|视频|文件|语音|动画表情|转发|JSON消息|Markdown消息|回复消息)', t))


def usable_target(t):
    """A candidate message for the persona to have produced."""
    t = clean_text(t)
    if not t:
        return ''
    if is_media_placeholder(t):
        return ''
    if len(t) > C.MAX_TARGET_CHARS:
        return ''
    return t


def usable_quote(t):
    """A candidate stimulus.  Rejects placeholders and empty stubs."""
    t = usable_target(t)
    if not t:
        return ''
    if t in QUOTE_PLACEHOLDERS:
        return ''
    if len(t) < C.MIN_QUOTE_CHARS:
        return ''
    if REPLY_PREFIX_RE.match(t):
        return ''
    if re.fullmatch(r'@\S{1,30}', t):
        return ''
    if len(t) > C.MAX_QUOTE_CHARS:
        return ''
    return t


def iter_group_records():
    """Yield records from every chunked export of the configured group."""
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


def reply_context(m):
    """Return (sender, content) from the reply element, if present."""
    for e in (m.get('content') or {}).get('elements') or []:
        if e.get('type') == 'reply':
            d = e.get('data') or {}
            return clean_text(d.get('senderName', '')), clean_text(d.get('content', ''))
    return None


SELF_LABEL = None  # set in main() from config


def collect():
    global SELF_LABEL
    C.require_identity()
    SELF_LABEL = C.TARGET_NAME

    msgs, dropped = [], collections.Counter()
    for m in iter_group_records():
        if m.get('system'):
            dropped['system message'] += 1
            continue
        if m.get('type') not in ('text', 'reply'):
            dropped[f"type={m.get('type')}"] += 1
            continue
        s = m.get('sender') or {}
        raw = (m.get('content') or {}).get('text', '')
        rctx = reply_context(m) if m.get('type') == 'reply' else None
        body = (REPLY_PREFIX_RE.sub('', clean_text(raw))
                if m.get('type') == 'reply' else clean_text(raw))
        t = usable_target(body)
        if not t:
            dropped['empty / media / too long'] += 1
            continue
        msgs.append({
            'ts': m.get('timestamp') or 0,
            'time': m.get('time'),
            'name': s.get('name') or '群友',
            'uin': s.get('uin'),
            'text': t,
            'is_target': s.get('uin') == C.TARGET_UIN and s.get('name') == C.TARGET_NAME,
            'reply_ctx': rctx,
        })
    msgs.sort(key=lambda r: r['ts'])
    return msgs, dropped


def recent_window(msgs, i, limit=6):
    m = msgs[i]
    ctx = []
    for j in range(i - 1, -1, -1):
        prev = msgs[j]
        if m['ts'] - prev['ts'] > C.SESSION_GAP_MS:
            break
        ctx.append(prev)
        if len(ctx) >= limit:
            break
    ctx.reverse()
    while ctx and sum(len(c['text']) + len(c['name']) + 1 for c in ctx) > C.MAX_USER_CHARS:
        ctx.pop(0)
    return ctx


def label(r):
    return f"{SELF_LABEL}：{r['text']}" if r['is_target'] else f"{r['name']}：{r['text']}"


def build(msgs):
    reply_rows, cont_rows = [], []
    quote_stats = collections.Counter()

    for i, m in enumerate(msgs):
        if not m['is_target']:
            continue
        window = recent_window(msgs, i)

        # ---------------- reply task ----------------
        if m['reply_ctx']:
            rname, rtext = m['reply_ctx']
            quote = usable_quote(rtext)
            if rname == SELF_LABEL:
                quote_stats['quote from self (skipped)'] += 1
                quote = ''
            elif not quote:
                quote_stats['quote unusable (placeholder / stub)'] += 1
            else:
                quote_stats['quote usable'] += 1

            if quote:
                prior = [c for c in window if not c['is_target']]
                lines = [label(c) for c in prior[-1:]]
                lines.append(f'{rname}：{quote}')
                user = '\n'.join(lines).strip()
                if user:
                    reply_rows.append({
                        'task': 'reply', 'ts': m['ts'], 'time': m['time'],
                        'messages': [
                            {'role': 'system', 'content': C.reply_prompt()},
                            {'role': 'user', 'content': user[:C.MAX_USER_CHARS]},
                            {'role': 'assistant', 'content': m['text']},
                        ],
                    })

        # ----------------- continue task -----------------
        if window:
            turns = window[-4:]
            msgs_list = [{'role': 'system', 'content': C.continue_prompt()}]
            started = False
            for c in turns:
                role = 'assistant' if c['is_target'] else 'user'
                if role == 'assistant':
                    started = True
                if not started:
                    continue
                msgs_list.append({'role': role, 'content': label(c)})
            if len(msgs_list) > 1 and msgs_list[-1]['role'] == 'assistant':
                msgs_list.append({'role': 'user', 'content': '（继续说）'})
                msgs_list.append({'role': 'assistant', 'content': m['text']})
                cont_rows.append({'task': 'continue', 'ts': m['ts'],
                                  'time': m['time'], 'messages': msgs_list})
            elif not started:
                cont_rows.append({
                    'task': 'continue', 'ts': m['ts'], 'time': m['time'],
                    'messages': [
                        {'role': 'system', 'content': C.continue_prompt()},
                        {'role': 'user', 'content': '（继续说）'},
                        {'role': 'assistant', 'content': m['text']},
                    ],
                })
    return reply_rows, cont_rows, quote_stats


def split_by_time(rows, val_frac=0.08):
    rows = sorted(rows, key=lambda r: r['ts'])
    if not rows:
        return [], []
    cut = rows[int(len(rows) * (1 - val_frac))]['ts']
    return [r for r in rows if r['ts'] < cut], [r for r in rows if r['ts'] >= cut]


def write_jsonl(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    msgs, dropped = collect()
    print(f'usable messages      : {len(msgs)}')
    print(f"  of which reply-type: {sum(1 for r in msgs if r['reply_ctx'])}")
    for k, v in dropped.most_common():
        print(f'  dropped {k}: {v}')

    reply_rows, cont_rows, quote_stats = build(msgs)
    print(f'\nreply examples   : {len(reply_rows)}')
    print(f'continue examples: {len(cont_rows)}')
    print('quote handling:')
    for k, v in quote_stats.most_common():
        print(f'   {k}: {v}')

    out = os.environ.get('DATASET_OUT', '.')
    os.makedirs(out, exist_ok=True)
    rt, rv = split_by_time(reply_rows)
    ct, cv = split_by_time(cont_rows)
    write_jsonl(os.path.join(out, 'v2_reply_train.jsonl'), rt)
    write_jsonl(os.path.join(out, 'v2_reply_val.jsonl'), rv)
    write_jsonl(os.path.join(out, 'v2_continue_train.jsonl'), ct)
    write_jsonl(os.path.join(out, 'v2_continue_val.jsonl'), cv)

    report = {
        'usable_messages': len(msgs),
        'reply_examples': len(reply_rows), 'reply_train': len(rt), 'reply_val': len(rv),
        'continue_examples': len(cont_rows), 'continue_train': len(ct),
        'continue_val': len(cv),
        'quotes_usable': quote_stats.get('quote usable', 0),
        'quotes_from_self_skipped': quote_stats.get('quote from self (skipped)', 0),
        'quotes_unusable': quote_stats.get('quote unusable (placeholder / stub)', 0),
    }
    with open(os.path.join(out, 'dataset_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('\n' + json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
