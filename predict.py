"""
ThyroAI Prediction Engine
Loads all three modules and provides prediction functions.
"""

import sys, types
import joblib
import numpy as np
from pathlib import Path

# ── patch missing top-level 'LabelEncoder' module (artifact of original pickling) ──
from sklearn.preprocessing import LabelEncoder as _LE
_fake = types.ModuleType("LabelEncoder")
_fake.LabelEncoder = _LE
sys.modules.setdefault("LabelEncoder", _fake)

# ── paths ──────────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"

# ── lazy singletons ────────────────────────────────────────────────────────────────
_models: dict = {}

def _load_all():
    if _models:
        return
    _models["m1"]  = joblib.load(MODEL_DIR / "module1_detection_model.pkl")
    _models["s1"]  = joblib.load(MODEL_DIR / "module1_scaler.pkl")
    _models["m2"]  = joblib.load(MODEL_DIR / "module2_staging_model.pkl")
    _models["s2"]  = joblib.load(MODEL_DIR / "module2_scaler.pkl")
    _models["le2"] = joblib.load(MODEL_DIR / "module2_label_encoder.pkl")
    _models["m3"]  = joblib.load(MODEL_DIR / "module3_metastasis_model.pkl")
    _models["s3"]  = joblib.load(MODEL_DIR / "module3_scaler.pkl")
    _models["le3"] = joblib.load(MODEL_DIR / "module3_label_encoder.pkl")
    _models["enc"] = joblib.load(MODEL_DIR / "clinical_encoders.pkl")


# ── encoder helpers ────────────────────────────────────────────────────────────────

GENDER_MAP   = {"F": 0, "M": 1}
BINARY_MAP   = {"No": 0, "Yes": 1}
THYROID_FN   = {v: i for i, v in enumerate([
    "Clinical Hyperthyroidism", "Clinical Hypothyroidism",
    "Euthyroid", "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"
])}
PHYS_EXAM    = {v: i for i, v in enumerate([
    "Diffuse goiter", "Multinodular goiter", "Normal",
    "Single nodular goiter-left", "Single nodular goiter-right"
])}
ADENOPATHY   = {v: i for i, v in enumerate(["Bilateral","Extensive","Left","No","Posterior","Right"])}
PATHOLOGY    = {v: i for i, v in enumerate(["Follicular","Hurthel cell","Micropapillary","Papillary"])}
FOCALITY     = {"Multi-Focal": 0, "Uni-Focal": 1}
RISK         = {"High": 0, "Intermediate": 1, "Low": 2}
T_STAGE      = {v: i for i, v in enumerate(["T1a","T1b","T2","T3a","T3b","T4a","T4b"])}
N_STAGE      = {"N0": 0, "N1a": 1, "N1b": 2}
M_STAGE      = {"M0": 0, "M1": 1}
RESPONSE     = {v: i for i, v in enumerate([
    "Biochemical Incomplete", "Excellent", "Indeterminate",
    "Structural Incomplete", "Recurred"
])}


# ── Module 1: Thyroid Detection ────────────────────────────────────────────────────

def predict_detection(
    age: float,
    gender: str,              # "M" | "F"
    family_history: int,      # 0/1
    radiation_exposure: int,  # 0/1
    nodule_size_mm: float,
    shape_score: float,
    margin_sharpness: float,
    echogenicity_index: float,
    calcification_presence: int,
    vascularity_score: float,
    texture_entropy: float,
    asymmetry_score: float,
    capsular_invasion: float,
    tsh_risk_level: float,
    tpoab_probability: float,
) -> dict:
    _load_all()
    gender_enc = GENDER_MAP[gender]
    X = np.array([[
        age, gender_enc, family_history, radiation_exposure,
        nodule_size_mm, shape_score, margin_sharpness, echogenicity_index,
        calcification_presence, vascularity_score, texture_entropy,
        asymmetry_score, capsular_invasion, tsh_risk_level, tpoab_probability,
    ]])
    Xs = _models["s1"].transform(X)
    proba = _models["m1"].predict_proba(Xs)[0]
    label = int(np.argmax(proba))
    return {
        "prediction": "Malignant" if label == 1 else "Benign",
        "confidence": float(proba[label]),
        "probability_benign":    float(proba[0]),
        "probability_malignant": float(proba[1]),
    }


# ── Module 2: Cancer Staging ───────────────────────────────────────────────────────

def predict_staging(
    age: float,
    gender: str,
    smoking: str,
    hx_smoking: str,
    hx_radiotherapy: str,
    thyroid_function: str,
    physical_examination: str,
    adenopathy: str,
    pathology: str,
    focality: str,
    risk: str,
    t: str,
    n: str,
    m: str,
    response: str,
) -> dict:
    _load_all()
    X = np.array([[
        age,
        GENDER_MAP[gender],
        BINARY_MAP[smoking],
        BINARY_MAP[hx_smoking],
        BINARY_MAP[hx_radiotherapy],
        THYROID_FN[thyroid_function],
        PHYS_EXAM[physical_examination],
        ADENOPATHY[adenopathy],
        PATHOLOGY[pathology],
        FOCALITY[focality],
        RISK[risk],
        T_STAGE[t],
        N_STAGE[n],
        M_STAGE[m],
        RESPONSE[response],
    ]])
    Xs = _models["s2"].transform(X)
    proba = _models["m2"].predict_proba(Xs)[0]
    label_idx = int(np.argmax(proba))
    stage = _models["le2"].classes_[label_idx]
    return {
        "stage": stage,
        "confidence": float(proba[label_idx]),
        "probabilities": {
            cls: float(p)
            for cls, p in zip(_models["le2"].classes_, proba)
        },
    }


# ── Module 3: Metastasis Prediction ───────────────────────────────────────────────

def predict_metastasis(
    age: float,
    gender: str,
    smoking: str,
    hx_smoking: str,
    hx_radiotherapy: str,
    thyroid_function: str,
    physical_examination: str,
    adenopathy: str,
    pathology: str,
    focality: str,
    risk: str,
    t: str,
    n: str,
    response: str,
) -> dict:
    _load_all()
    X = np.array([[
        age,
        GENDER_MAP[gender],
        BINARY_MAP[smoking],
        BINARY_MAP[hx_smoking],
        BINARY_MAP[hx_radiotherapy],
        THYROID_FN[thyroid_function],
        PHYS_EXAM[physical_examination],
        ADENOPATHY[adenopathy],
        PATHOLOGY[pathology],
        FOCALITY[focality],
        RISK[risk],
        T_STAGE[t],
        N_STAGE[n],
        RESPONSE[response],
    ]])
    Xs = _models["s3"].transform(X)
    proba = _models["m3"].predict_proba(Xs)[0]
    label_idx = int(np.argmax(proba))
    result = _models["le3"].classes_[label_idx]
    return {
        "metastasis": result,
        "has_metastasis": result == "M1",
        "confidence": float(proba[label_idx]),
        "probability_M0": float(proba[0]),
        "probability_M1": float(proba[1]),
    }
