# Run on Ubuntu

## 1. Unzip
unzip mlbc_fraud.zip -d mlbc_project && cd mlbc_project

## 2. Install Python + pip (if missing)
sudo apt update
sudo apt install -y python3 python3-pip python3-venv unzip

## 3. Create venv & install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## 4. Generate data + train (only once; skip if data/ and artifacts/ exist)
python generate_data.py
python train.py

## 5. Start app
uvicorn app:app --host 0.0.0.0 --port 8000
# or: bash run.sh

## 6. Open
http://localhost:8000
# from another machine: http://<ubuntu-ip>:8000
# health: curl http://localhost:8000/health