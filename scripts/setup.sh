#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/.."
mkdir -p logs outputs
if [[ ! -d LLaMA-Factory/.git ]]; then
  git clone --depth 1 --branch v0.9.5 https://github.com/hiyouga/LLaMA-Factory.git LLaMA-Factory
fi
expected_revision=7af909522a951e3ad9f022ea6f88b6755257eaa5
actual_revision=$(git -C LLaMA-Factory rev-parse HEAD)
if [[ "$actual_revision" != "$expected_revision" ]]; then
  echo "LLaMA-Factory revision differs from the recorded experiment: $actual_revision" >&2
  exit 1
fi
if [[ ! -d .venv ]]; then
  uv venv --python 3.11.17 .venv
fi
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip check --python .venv/bin/python
.venv/bin/python scripts/check_environment.py
.venv/bin/llamafactory-cli version
