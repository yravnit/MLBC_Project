# MLBC Fraud Detection Workbench (ML project)

Machine Learning project for the MLBC (Machine Learning & Blockchain) subject.
ML-only deliverable: credit-card fraud detection model + web UI to score transactions.

## Run
1. Generate data (only once):
   python generate_data.py
2. Train (creates artifacts/):
   python train.py
3. Serve:
   uvicorn app:app --reload
   open http://127.0.0.1:8000

## Structure
- generate_data.py  - synthetic transaction dataset
- train.py          - EDA + model comparison + train + eval + plots + save
- app.py            - FastAPI: /api/predict, /api/meta, /api/samples
- static/index.html - web UI
- artifacts/       - model.pkl, metrics, samples, plots (generated)
- data/            - transactions.csv, schema.json (generated)

## Model
RandomForestClassifier(class_weight='balanced'), threshold tuned to 0.3.
Train/test split stratified 80/20. Metrics in artifacts/metrics.json.