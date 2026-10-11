# -*- coding: utf-8 -*-
"""Central configuration.

Every value that could identify a real person, group, or machine is read from
an environment variable.  Nothing about *whose* chat this is belongs in source
control -- that is the whole point of the repository.

Copy `.env.example` to `.env` (or just export these) before running anything.
"""
import os

# --------------------------------------------------------------------------
# Who we are imitating.  No defaults: fail loudly rather than silently train
# on the wrong person.
# --------------------------------------------------------------------------
TARGET_UIN = os.environ.get('TARGET_UIN', '')
TARGET_NAME = os.environ.get('TARGET_NAME', '')
TARGET_GROUP = os.environ.get('TARGET_GROUP', '')

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
EXPORTS_DIR = os.environ.get(
    'EXPORTS_DIR', os.path.expanduser('~/qq-exports')).replace('\\', '/')
BASE_MODEL = os.environ.get('BASE_MODEL', '').replace('\\', '/')
OUT_DIR = os.environ.get('OUT_DIR', './lora-out')

# --------------------------------------------------------------------------
# Dataset shaping
# --------------------------------------------------------------------------
SESSION_GAP_MS = int(os.environ.get('SESSION_GAP_MS', str(20 * 60 * 1000)))
MAX_TARGET_CHARS = int(os.environ.get('MAX_TARGET_CHARS', '200'))
MAX_QUOTE_CHARS = int(os.environ.get('MAX_QUOTE_CHARS', '200'))
MAX_USER_CHARS = int(os.environ.get('MAX_USER_CHARS', '600'))
MIN_QUOTE_CHARS = int(os.environ.get('MIN_QUOTE_CHARS', '2'))

# --------------------------------------------------------------------------
# Prompts.
#
# Two SEPARATE prompts for two separate behaviours.  This is the single most
# important design decision in the project: sharing one prompt between
# "answer someone" and "continue my own thought" teaches the model that all
# recent text is its own words, which makes it invent prior conversation.
# --------------------------------------------------------------------------
SYSTEM_REPLY = (
    "你在一个中文QQ群里聊天，群友刚对你说了句话。你要模仿群成员“{name}”的说话方式接话："
    "句子通常很短，口语化，常用网络用语，会吐槽、会顺嘴接梗，也可能只是随口一说。"
    "只对着群友刚说的那句话回应，不要提到你自己之前说过什么。"
    "只模仿语气和表达习惯，不要编造他的经历、身份或记忆。"
    "直接输出要发到群里的内容，不要解释，不要加引号，不要输出分析。"
)

SYSTEM_CONTINUE = (
    "你在一个中文QQ群里，接着自己刚才的话往下说。你要模仿群成员“{name}”的说话方式："
    "句子通常很短，口语化，常用网络用语，会吐槽、会顺嘴接梗，也可能只是随口一说。"
    "只模仿语气和表达习惯，不要编造他的经历、身份或记忆。"
    "直接输出要发到群里的内容，不要解释，不要加引号，不要输出分析。"
)


def reply_prompt():
    return SYSTEM_REPLY.format(name=TARGET_NAME or '目标成员')


def continue_prompt():
    return SYSTEM_CONTINUE.format(name=TARGET_NAME or '目标成员')


def require_identity():
    """Abort early if the identity was not supplied."""
    missing = [k for k, v in (('TARGET_UIN', TARGET_UIN),
                              ('TARGET_NAME', TARGET_NAME),
                              ('TARGET_GROUP', TARGET_GROUP)) if not v]
    if missing:
        raise SystemExit(
            'missing required environment variable(s): ' + ', '.join(missing) +
            '\nsee .env.example')
