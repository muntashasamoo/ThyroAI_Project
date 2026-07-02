"""
ThyroAI REST API
================
Run locally:   uvicorn api:app --reload --port 8000
Swagger docs:  http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from schemas import (
    DetectionInput,  DetectionOutput,
    StagingInput,    StagingOutput,
    MetastasisInput, MetastasisOutput,
    HealthResponse,
)
import predict

# ── App setup ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ThyroAI — Thyroid Cancer Prediction API",
    description=(
        "Three-module AI system for thyroid cancer analysis:\n\n"
        "- **Module 1** – Benign/Malignant detection from ultrasound + lab features\n"
        "- **Module 2** – TNM-based cancer staging (Stage I–IVB)\n"
        "- **Module 3** – Distant metastasis prediction (M0/M1)\n\n"
        "> For clinical decision support only. Not a substitute for medical advice."
    ),
    version="1.0.0",
    contact={"name": "ThyroAI Team"},
    license_info={"name": "Research Use Only"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (Web UI) if folder exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Health check ──────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health():
    """API health check and version info."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "modules": ["detection", "staging", "metastasis"],
    }

@app.get("/ui", include_in_schema=False)
def serve_ui():
    """Serve the Web UI."""
    ui_path = Path(__file__).parent / "static" / "index.html"
    if not ui_path.exists():
        raise HTTPException(404, "UI not found")
    return FileResponse(str(ui_path))


# ── Module 1 – Detection ──────────────────────────────────────────────────────────

@app.post(
    "/predict/detection",
    response_model=DetectionOutput,
    tags=["Module 1 – Detection"],
    summary="Predict benign vs malignant thyroid nodule",
)
def detect(body: DetectionInput):
    """
    Classifies a thyroid nodule as **Benign** or **Malignant** using
    ultrasound imaging features and lab biomarkers.

    Returns prediction label, confidence score, and class probabilities.
    """
    try:
        result = predict.predict_detection(
            age=body.age,
            gender=body.gender,
            family_history=body.family_history,
            radiation_exposure=body.radiation_exposure,
            nodule_size_mm=body.nodule_size_mm,
            shape_score=body.shape_score,
            margin_sharpness=body.margin_sharpness,
            echogenicity_index=body.echogenicity_index,
            calcification_presence=body.calcification_presence,
            vascularity_score=body.vascularity_score,
            texture_entropy=body.texture_entropy,
            asymmetry_score=body.asymmetry_score,
            capsular_invasion=body.capsular_invasion,
            tsh_risk_level=body.tsh_risk_level,
            tpoab_probability=body.tpoab_probability,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Module 2 – Staging ────────────────────────────────────────────────────────────

@app.post(
    "/predict/staging",
    response_model=StagingOutput,
    tags=["Module 2 – Staging"],
    summary="Predict cancer stage (I / II / III / IVA / IVB)",
)
def stage(body: StagingInput):
    """
    Predicts the **TNM-based cancer stage** for a thyroid cancer patient
    using clinical and pathological features.

    Returns predicted stage, confidence, and probability distribution.
    """
    try:
        result = predict.predict_staging(
            age=body.age,
            gender=body.gender,
            smoking=body.smoking,
            hx_smoking=body.hx_smoking,
            hx_radiotherapy=body.hx_radiotherapy,
            thyroid_function=body.thyroid_function,
            physical_examination=body.physical_examination,
            adenopathy=body.adenopathy,
            pathology=body.pathology,
            focality=body.focality,
            risk=body.risk,
            t=body.t,
            n=body.n,
            m=body.m,
            response=body.response,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Module 3 – Metastasis ─────────────────────────────────────────────────────────

@app.post(
    "/predict/metastasis",
    response_model=MetastasisOutput,
    tags=["Module 3 – Metastasis"],
    summary="Predict distant metastasis (M0 / M1)",
)
def metastasis(body: MetastasisInput):
    """
    Predicts the presence of **distant metastasis** (M0 = none, M1 = present)
    using clinical features (TNM T/N stages, pathology, risk factors).

    Returns metastasis label, binary flag, confidence, and M0/M1 probabilities.
    """
    try:
        result = predict.predict_metastasis(
            age=body.age,
            gender=body.gender,
            smoking=body.smoking,
            hx_smoking=body.hx_smoking,
            hx_radiotherapy=body.hx_radiotherapy,
            thyroid_function=body.thyroid_function,
            physical_examination=body.physical_examination,
            adenopathy=body.adenopathy,
            pathology=body.pathology,
            focality=body.focality,
            risk=body.risk,
            t=body.t,
            n=body.n,
            response=body.response,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
