#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/opd-plus-attention-deepseek-r1-distill-qwen-1.5b-justrl-deepseek-1.5b.sh" "$@"
