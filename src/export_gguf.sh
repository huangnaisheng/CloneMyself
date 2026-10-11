#!/usr/bin/env bash
# Export a LoRA adapter into a runnable Ollama model.
#
#   bash src/export_gguf.sh <adapter_model.safetensors> <model-name> [tag] [QUANT]
#
# Stages: merge adapter into base shards -> GGUF (f16) -> quantize -> ollama create
#
# Two hard-won details:
#
#   1. Every path handed to the llama.cpp tools is pure ASCII and uses forward
#      slashes.  Passing a non-ASCII directory name to llama-quantize.exe
#      mangles the argument (observed: 中文目录 -> Ĭ�Ϲ�����).  Make a
#      junction (mklink /J C:\gguf <workspace>) and route through it.
#      WS below is that junction and MUST exist.
#
#   2. Ollama 0.40.2 cannot ingest a LoRA at all (`LoRA adapters are no longer
#      supported`), cannot import a safetensors dir (`MLX runtime is not
#      available`), and its int4 path fails on lm_head.  Converting to GGUF
#      yourself is the only route.
set -euo pipefail

ADAPTER="${1:?usage: export_gguf.sh <adapter.safetensors> <model-name> [tag] [QUANT]}"
NAME="${2:?usage: export_gguf.sh <adapter.safetensors> <model-name> [tag] [QUANT]}"
TAG="${3:-lora}"
QUANT="${4:-Q4_K_M}"

# ASCII junction over the workspace.  Override with WS=... if yours differs.
WS="${WS:-/c/gguf}"
PY="${PY:-$WS/.style-ft-venv/Scripts/python.exe}"
SK="${SK:-$WS/src}"
LLAMA="${LLAMA:-$WS/llamacpp/llama.cpp-master}"
BIN="${BIN:-$WS/llamacpp/bin}"
OLLAMA="${OLLAMA:-$(command -v ollama || echo "$LOCALAPPDATA/Programs/Ollama/ollama.exe")}"
BASE_MODEL_WIN="${BASE_MODEL:-C:/gguf/models/Qwen3-8B}"
MODELS_WIN="${MODELS_WIN:-C:/gguf/models}"

[ -d "$WS" ] || { echo "ERROR: $WS does not exist." >&2
  echo "Create the ASCII junction first:  mklink /J C:\\gguf <workspace>" >&2
  exit 1; }
[ -x "$PY" ] || { echo "ERROR: python not found at $PY" >&2; exit 1; }

ADAPTER_WIN="$(cygpath -m "$ADAPTER" 2>/dev/null || echo "$ADAPTER")"

export PYTHONIOENCODING=utf-8
export HF_HOME="${HF_HOME:-$WS/models/hf_cache}"

echo "=== [1/4] merge adapter into base, shard by shard ==="
ADAPTER_FILE="$ADAPTER_WIN" \
MERGED_DIR="$MODELS_WIN/merged-$TAG" \
BASE_MODEL="$BASE_MODEL_WIN" \
  "$PY" -u "$SK/merge_lora.py"

echo
echo "=== [2/4] convert merged model to GGUF (f16) ==="
cd "$LLAMA"
PYTHONPATH="$LLAMA/gguf-py" "$PY" -u convert_hf_to_gguf.py \
  "$MODELS_WIN/merged-$TAG" \
  --outfile "$MODELS_WIN/$TAG-f16.gguf" \
  --outtype f16 2>&1 | tail -n 3

echo
echo "=== [3/4] quantize to $QUANT ==="
cd "$BIN"
PATH="$BIN:$PATH" ./llama-quantize.exe \
  "$MODELS_WIN/$TAG-f16.gguf" \
  "$MODELS_WIN/$TAG-$QUANT.gguf" "$QUANT" 2>&1 | tail -n 4

echo
echo "=== [4/4] create ollama model: $NAME ==="
cd "$SK"
cat > "Modelfile.$TAG" <<EOF
FROM $MODELS_WIN/$TAG-$QUANT.gguf

PARAMETER temperature 0.85
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.12
PARAMETER num_ctx 4096

SYSTEM """
${SYSTEM_PROMPT:-你在一个中文QQ群里聊天，群友刚对你说了句话。你要模仿群成员的说话方式接话：句子通常很短，口语化，常用网络用语，会吐槽、会顺嘴接梗，也可能只是随口一说。
只对着群友刚说的那句话回应，不要提到你自己之前说过什么。
只模仿语气和表达习惯，不要编造经历、身份或记忆。
直接输出要发到群里的内容，不要解释，不要加引号，不要输出分析。}
"""
EOF
"$OLLAMA" create "$NAME" -f "Modelfile.$TAG"

echo
echo "=== done: $NAME ==="
"$OLLAMA" list | head -6
