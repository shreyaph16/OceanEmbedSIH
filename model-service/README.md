## Model checkpoints (not in git — download separately)

Place these files in `checkpoints/` before running:
- `varuna_regional_final_30ep.pt`
- `varuna_inference_artifacts.pt`

Place this file in `data/` before running `make_test_payload.py`:
- `surface_025_2022.nc`

## Running locally
cd model-service
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
