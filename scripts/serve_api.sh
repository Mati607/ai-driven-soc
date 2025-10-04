#!/bin/zsh
set -euo pipefail

export PYTHONPATH=$(pwd):${PYTHONPATH:-}
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload


