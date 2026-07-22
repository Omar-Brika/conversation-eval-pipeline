#!/bin/bash

# installModels.sh — Download GGUF models via huggingface-cli
# Usage: ./installModels.sh --model-dir /path/to/models [--quant Q4_K_M] [--config models.conf]

set -euo pipefail

MODEL_DIR=""
QUANT="Q4_K_M"
CONFIG="models.conf"

# ── CLI arguments ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --quant)     QUANT="$2";     shift 2 ;;
    --config)    CONFIG="$2";    shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [ -z "$MODEL_DIR" ]; then
  echo "Usage: $0 --model-dir /path/to/models [--quant Q4_K_M] [--config models.conf]"
  exit 1
fi

if [ ! -f "$CONFIG" ]; then
  echo "Config file not found: $CONFIG"
  exit 1
fi

# ── Ensure huggingface-cli is available ─────────────────────────
if ! command -v huggingface-cli &>/dev/null; then
  echo "huggingface-cli not found. Installing..."
  pip install -U "huggingface_hub[cli]"
fi

# Enable hf_transfer for faster downloads if available
export HF_HUB_ENABLE_HF_TRANSFER=1

# ── Ensure user is logged in (needed for some gated models) ─────
if ! huggingface-cli whoami &>/dev/null 2>&1; then
  echo "Not logged in to HuggingFace. Running login..."
  huggingface-cli login
fi

# ── Download loop ────────────────────────────────────────────────
FAILED=()
SUCCEEDED=()

while IFS= read -r line; do
  # skip comments and empty lines
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${line// /}" ]] && continue

  # expand ${QUANT} in the line
  line="${line//\$\{QUANT\}/$QUANT}"

  IFS='|' read -r category repo filename <<< "$line"

  dest_dir="${MODEL_DIR}/${category}"
  mkdir -p "$dest_dir"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Model:     ${filename}"
  echo "  Repo:      ${repo}"
  echo "  Category:  ${category}"
  echo "  Dest:      ${dest_dir}/"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [ -f "${dest_dir}/${filename}" ]; then
    echo "  Already exists, skipping."
    SUCCEEDED+=("$filename")
    continue
  fi

  if huggingface-cli download "$repo" "$filename" --local-dir "$dest_dir"; then
    echo "  Downloaded successfully."
    SUCCEEDED+=("$filename")
  else
    echo "  FAILED to download ${filename}"
    FAILED+=("$filename")
  fi

done < "$CONFIG"

# ── Summary ──────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "  Download Summary"
echo "════════════════════════════════════════════════════"
echo "  Succeeded: ${#SUCCEEDED[@]}"
echo "  Failed:    ${#FAILED[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
  echo ""
  echo "  Failed models:"
  for f in "${FAILED[@]}"; do
    echo "    - $f"
  done
  echo ""
  echo "  Re-run this script to retry failed downloads."
  exit 1
fi

echo ""
echo "  All models downloaded to: ${MODEL_DIR}/"
echo "  You can now run: ./run.sh --model-dir ${MODEL_DIR}"
