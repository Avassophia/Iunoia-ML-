"""
RISK SCORES
Turns model outputs into consistent risk scores + buckets.

Inputs:
  outputs dict from predictor:
    - cortisol_load
    - cycle_variability_days
    - bone_loss_pct_per_month

Outputs:
  same dict +:
    - risk_score (0..1)
    - risk_overall (Low/Moderate/High)
    - risk_breakdown
"""

from __future__ import annotations
from typing import Dict, Any

# -------------------------
# Config
# -------------------------

try:
    from .config_loader import load_constants
except ImportError:
    from config_loader import load_constants

try:
    CONST = load_constants()
except Exception as e:
    print("Failed to load constants in risk_scores:", e)
    CONST = {}

RISK_CFG = CONST.get("risk_buckets", {})

LOW_T = RISK_CFG.get("low_threshold", 0.35)
MOD_T = RISK_CFG.get("moderate_threshold", 0.70)

# Optional: allow configurable weights
ML_CFG = CONST.get("ml", {})
WEIGHTS = ML_CFG.get("risk_weights", {
    "cortisol": 0.45,
    "cycle": 0.30,
    "bone": 0.25
})

# -------------------------
# Helpers
# -------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _bucket(score: float) -> str:
    if score < LOW_T:
        return "Low"
    if score < MOD_T:
        return "Moderate"
    return "High"

# -------------------------
# Individual risks
# -------------------------

def cortisol_risk(cortisol_load: float) -> float:
    return _clamp(float(cortisol_load) / 100.0, 0.0, 1.0)


def cycle_risk(cycle_variability_days: float) -> float:
    return _clamp(float(cycle_variability_days) / 10.0, 0.0, 1.0)


def bone_risk(bone_loss_pct_per_month: float) -> float:
    x = float(bone_loss_pct_per_month)
    return _clamp((x - 0.2) / 2.0, 0.0, 1.0)

# -------------------------
# Composite risk
# -------------------------

def overall_risk_from_outputs(outputs: Dict[str, Any]) -> float:
    c = cortisol_risk(outputs.get("cortisol_load", 0.0))
    y = cycle_risk(outputs.get("cycle_variability_days", 0.0))
    b = bone_risk(outputs.get("bone_loss_pct_per_month", 0.2))

    score = (
        WEIGHTS.get("cortisol", 0.45) * c +
        WEIGHTS.get("cycle", 0.30) * y +
        WEIGHTS.get("bone", 0.25) * b
    )

    return _clamp(score, 0.0, 1.0)

# -------------------------
# Public API
# -------------------------

def attach_risk_breakdown(outputs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(outputs)

    c = cortisol_risk(out.get("cortisol_load", 0.0))
    y = cycle_risk(out.get("cycle_variability_days", 0.0))
    b = bone_risk(out.get("bone_loss_pct_per_month", 0.2))

    risk_score = overall_risk_from_outputs(out)

    out["risk_score"] = round(risk_score, 2)
    out["risk_overall"] = _bucket(risk_score)
    out["risk_breakdown"] = {
        "cortisol": {"score": round(c, 3), "bucket": _bucket(c)},
        "cycle": {"score": round(y, 3), "bucket": _bucket(y)},
        "bone": {"score": round(b, 3), "bucket": _bucket(b)},
    }

    return out

# -------------------------
# Demo
# -------------------------

if __name__ == "__main__":
    sample = {
        "cortisol_load": 34.8,
        "cycle_variability_days": 6.81,
        "bone_loss_pct_per_month": 1.95,
    }

    print(attach_risk_breakdown(sample))