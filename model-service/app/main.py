from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np

from .inference.model_loader import get_model_bundle
from .inference.predictor import predict

app = FastAPI(title="OceanEmbed Model Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this once you know your backend's real domain
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    so: List[List[float]]
    uo: List[List[float]]
    vo: List[List[float]]
    zos: List[List[float]]


@app.on_event("startup")
def load_model():
    get_model_bundle()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    bundle = get_model_bundle()
    try:
        so = np.array(req.so, dtype=np.float32)
        uo = np.array(req.uo, dtype=np.float32)
        vo = np.array(req.vo, dtype=np.float32)
        zos = np.array(req.zos, dtype=np.float32)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input arrays: {e}")

    expected_shape = (len(bundle.lat_grid), len(bundle.lon_grid))
    if so.shape != expected_shape:
        raise HTTPException(status_code=400, detail=f"Expected grid shape {expected_shape}, got {so.shape}")

    return predict(bundle, so, uo, vo, zos)