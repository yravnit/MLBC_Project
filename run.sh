#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip install -q -r requirements.txt
[ -f data/transactions.csv ] || python generate_data.py
[ -f artifacts/model.pkl ] || python train.py
uvicorn app:app --host 0.0.0.0 --port 8000