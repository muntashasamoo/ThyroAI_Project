"""
ThyroAI — Input/Output Validation Schemas (Pydantic v2)
"""

from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Shared literals ────────────────────────────────────────────────────────────────

GenderType        = Literal["M", "F"]
BinaryType        = Literal["Yes", "No"]
ThyroidFnType     = Literal[
    "Clinical Hyperthyroidism", "Clinical Hypothyroidism",
    "Euthyroid", "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"
]
PhysExamType      = Literal[
    "Diffuse goiter", "Multinodular goiter", "Normal",
    "Single nodular goiter-left", "Single nodular goiter-right"
]
AdenopathyType    = Literal["Bilateral", "Extensive", "Left", "No", "Posterior", "Right"]
PathologyType     = Literal["Follicular", "Hurthel cell", "Micropapillary", "Papillary"]
FocalityType      = Literal["Multi-Focal", "Uni-Focal"]
RiskType          = Literal["High", "Intermediate", "Low"]
TStageType        = Literal["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"]
NStageType        = Literal["N0", "N1a", "N1b"]
MStageType        = Literal["M0", "M1"]
ResponseType      = Literal[
    "Biochemical Incomplete", "Excellent", "Indeterminate",
    "Structural Incomplete", "Recurred"
]


# ══════════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — Thyroid Detection
# ══════════════════════════════════════════════════════════════════════════════════

class DetectionInput(BaseModel):
    """Ultrasound + lab features for benign/malignant classification."""

    age:                      float = Field(..., ge=0,  le=120, description="Patient age in years")
    gender:                   GenderType
    family_history:           int   = Field(..., ge=0,  le=1,   description="Family history of thyroid cancer (0/1)")
    radiation_exposure:       int   = Field(..., ge=0,  le=1,   description="History of radiation exposure (0/1)")
    nodule_size_mm:           float = Field(..., gt=0,  le=200, description="Nodule size in millimetres")
    shape_score:              float = Field(..., ge=0,  le=1,   description="Ultrasound shape regularity score [0–1]")
    margin_sharpness:         float = Field(..., ge=0,  le=1,   description="Margin sharpness score [0–1]")
    echogenicity_index:       float = Field(..., ge=0,  le=1,   description="Echogenicity index [0–1]")
    calcification_presence:   int   = Field(..., ge=0,  le=1,   description="Calcification present (0/1)")
    vascularity_score:        float = Field(..., ge=0,  le=1,   description="Vascularity score [0–1]")
    texture_entropy:          float = Field(..., ge=0,  le=5,   description="Texture entropy (image analysis)")
    asymmetry_score:          float = Field(..., ge=0,  le=1,   description="Asymmetry score [0–1]")
    capsular_invasion:        float = Field(..., ge=0,  le=1,   description="Capsular invasion indicator [0–1]")
    tsh_risk_level:           float = Field(..., ge=0,           description="TSH risk level (mIU/L)")
    tpoab_probability:        float = Field(..., ge=0,  le=1,   description="TPOAb probability [0–1]")

    @field_validator("shape_score", "margin_sharpness", "echogenicity_index",
                     "vascularity_score", "asymmetry_score", "capsular_invasion",
                     "tpoab_probability", mode="before")
    @classmethod
    def clamp_0_1(cls, v):
        if isinstance(v, (int, float)) and not (0 <= v <= 1):
            raise ValueError("Value must be between 0 and 1")
        return v

    model_config = {"json_schema_extra": {"example": {
        "age": 45, "gender": "F", "family_history": 1,
        "radiation_exposure": 0, "nodule_size_mm": 18.5,
        "shape_score": 0.72, "margin_sharpness": 0.65,
        "echogenicity_index": 0.55, "calcification_presence": 1,
        "vascularity_score": 0.48, "texture_entropy": 2.3,
        "asymmetry_score": 0.35, "capsular_invasion": 0.2,
        "tsh_risk_level": 3.1, "tpoab_probability": 0.25,
    }}}


class DetectionOutput(BaseModel):
    prediction:            Literal["Benign", "Malignant"]
    confidence:            float
    probability_benign:    float
    probability_malignant: float


# ══════════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — Cancer Staging
# ══════════════════════════════════════════════════════════════════════════════════

class StagingInput(BaseModel):
    """Clinical features for TNM cancer staging (I / II / III / IVA / IVB)."""

    age:                  float         = Field(..., ge=0, le=120)
    gender:               GenderType
    smoking:              BinaryType
    hx_smoking:           BinaryType    = Field(..., alias="hx_smoking")
    hx_radiotherapy:      BinaryType
    thyroid_function:     ThyroidFnType
    physical_examination: PhysExamType
    adenopathy:           AdenopathyType
    pathology:            PathologyType
    focality:             FocalityType
    risk:                 RiskType
    t:                    TStageType
    n:                    NStageType
    m:                    MStageType
    response:             ResponseType

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {"example": {
            "age": 52, "gender": "M", "smoking": "No",
            "hx_smoking": "Yes", "hx_radiotherapy": "No",
            "thyroid_function": "Euthyroid",
            "physical_examination": "Single nodular goiter-right",
            "adenopathy": "No", "pathology": "Papillary",
            "focality": "Uni-Focal", "risk": "Intermediate",
            "t": "T2", "n": "N0", "m": "M0", "response": "Excellent",
        }}
    }


class StagingOutput(BaseModel):
    stage:         str
    confidence:    float
    probabilities: dict[str, float]


# ══════════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — Metastasis Prediction
# ══════════════════════════════════════════════════════════════════════════════════

class MetastasisInput(BaseModel):
    """Clinical features for distant metastasis prediction (M0 / M1)."""

    age:                  float         = Field(..., ge=0, le=120)
    gender:               GenderType
    smoking:              BinaryType
    hx_smoking:           BinaryType
    hx_radiotherapy:      BinaryType
    thyroid_function:     ThyroidFnType
    physical_examination: PhysExamType
    adenopathy:           AdenopathyType
    pathology:            PathologyType
    focality:             FocalityType
    risk:                 RiskType
    t:                    TStageType
    n:                    NStageType
    response:             ResponseType  # Note: no M field here

    model_config = {"json_schema_extra": {"example": {
        "age": 61, "gender": "F", "smoking": "Yes",
        "hx_smoking": "No", "hx_radiotherapy": "Yes",
        "thyroid_function": "Clinical Hypothyroidism",
        "physical_examination": "Multinodular goiter",
        "adenopathy": "Bilateral", "pathology": "Follicular",
        "focality": "Multi-Focal", "risk": "High",
        "t": "T4a", "n": "N1b", "response": "Structural Incomplete",
    }}}


class MetastasisOutput(BaseModel):
    metastasis:     str
    has_metastasis: bool
    confidence:     float
    probability_M0: float
    probability_M1: float


# ── Health check ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    modules: list[str]
