"""
PREDICTOR — maps raw mission inputs and engineered features to mission projections.

MVP "physics-inspired" + evidence-informed simulator:
- Cortisol Load (0–100)
- Cycle Variability (absolute +/- days)
- Bone Loss Rate (% BMD per month)

Designed to be replaced by a trained Azure ML model later.

Recommended run (module mode):
  cd iunoia-core
  python -m model.predictor

Also supports (script mode):
  python model/predictor.py
"""

"""
...docstring...
"""

from dataclasses import dataclass
from typing import Dict, Any
import math
from .config_loader import load_constants

CONST = load_constants()

try:
    from .inference import load_model, predict as ml_predict
    MODEL = load_model()
except:
    MODEL = None

try:
    from .features import build_features
    from .risk_scores import attach_risk_breakdown
except ImportError:
    from features import build_features
    from risk_scores import attach_risk_breakdown


# --- imports that work in BOTH module and script mode ---
try:
    # Module mode: python -m model.predictor
    from .features import build_features
    from .risk_scores import attach_risk_breakdown
except ImportError:
    # Script mode: python model/predictor.py
    from features import build_features
    from risk_scores import attach_risk_breakdown


# -------------------------
# Types
# -------------------------

@dataclass(frozen=True)
class Prediction:
    mission_day: int
    cortisol_load: float               # 0–100 (higher = worse)
    cycle_variability_days: float      # abs deviation from baseline cycle length (days)
    bone_loss_pct_per_month: float     # % BMD loss per month
    drivers: Dict[str, str]            # short explanations of top drivers


# -------------------------
# Helpers
# -------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# -------------------------
# Core simulator
# -------------------------

def predict_from_features(features: Dict[str, float]) -> Prediction:
    """
    Engineered features -> projections.
    Feature keys come from build_features() in features.py
    """

    mission_day = int(features.get("mission_day", 1))

    g = float(features.get("gravity_factor", 0.0))
    r = float(features.get("radiation_factor", 0.6))
    iso = float(features.get("isolation_factor", 0.6))

    stress = float(features.get("stress_score", 0.5))
    sleep_def = float(features.get("sleep_deficit", 0.0))

    irr_hist = float(features.get("cycle_irregularity_history", 0.0))
    bmd_conc = float(features.get("bone_density_concerns", 0.0))
    sleep_dis = float(features.get("sleep_disorders", 0.0))
    repro_flag = float(features.get("pcos_or_endometriosis_flag", 0.0))

    stress_x_sleep = float(features.get("stress_x_sleep", stress * sleep_def))
    rad_x_grav = float(features.get("rad_x_grav", r * g))
    iso_x_stress = float(features.get("iso_x_stress", iso * stress))

    # 1) Cortisol load (0–100)
    c = CONST["cortisol"]

    t = _clamp(mission_day / c["time_scale_days"], 0.0, 1.5)

    time_gain = (
        c["time_gain_base"] +
        c["time_gain_max"] * _sigmoid(3.0 * (t - 0.35))
    )

    base_cort = (
        c["stress_weight"] * stress +
        c["sleep_deficit_weight"] * sleep_def +
        c["isolation_weight"] * iso +
        c["interaction_weight"] * stress_x_sleep
    )

    sensitivity = (
        1.0 +
        c["reproductive_sensitivity"] * repro_flag +
        c["sleep_disorder_sensitivity"] * sleep_dis
    )

    cortisol_score_0_to_1 = _clamp(
        base_cort * time_gain * sensitivity,
        0.0,
        1.0
    )

    cortisol_load = round(100.0 * cortisol_score_0_to_1, 1)

    # 2) Cycle variability (days)
    cv = CONST["cycle_variability"]

    cycle_instability = (
        cv["stress_weight"] * stress +
        cv["sleep_deficit_weight"] * sleep_def +
        cv["isolation_weight"] * iso +
        cv["radiation_weight"] * r +
        cv["history_weight"] * irr_hist +
        cv["reproductive_weight"] * repro_flag
    )

    cycle_instability += (
        cv["stress_sleep_interaction"] * stress_x_sleep +
        cv["isolation_stress_interaction"] * iso_x_stress
    )

    cycle_instability = _clamp(cycle_instability, 0.0, 1.0)

    cycle_variability_days = round(
        cv["baseline_days"] +
        cv["max_additional_days"] * cycle_instability,
        2
    )

    # 3) Bone loss rate (% per month)
    b = CONST["bone_loss"]

    bone_risk_internal = (
        b["gravity_weight"] * g +
        b["radiation_gravity_weight"] * rad_x_grav +
        b["baseline_concern_weight"] * bmd_conc +
        b["sleep_deficit_weight"] * sleep_def +
        b["stress_weight"] * stress
    )

    bone_risk_internal = _clamp(bone_risk_internal, 0.0, 1.0)

    bone_loss_pct_per_month = round(
        b["baseline_pct_per_month"] +
        b["max_additional_pct_per_month"] * bone_risk_internal,
        2
    )

    # Drivers (explainability)
    drivers: Dict[str, str] = {}

    if sleep_def >= stress and sleep_def >= iso:
        drivers["cortisol"] = "Primary driver: cumulative sleep deficit."
    elif stress >= sleep_def and stress >= iso:
        drivers["cortisol"] = "Primary driver: elevated workload/stress."
    else:
        drivers["cortisol"] = "Primary driver: isolation load."

    if stress_x_sleep > 0.25:
        drivers["cycle"] = "Primary driver: stress × sleep interaction."
    elif irr_hist > 0.4 or repro_flag > 0.0:
        drivers["cycle"] = "Primary driver: pre-flight cycle sensitivity."
    else:
        drivers["cycle"] = "Primary driver: circadian disruption proxy."

    if g > 0.5:
        drivers["bone"] = "Primary driver: microgravity exposure duration."
    else:
        drivers["bone"] = "Primary driver: baseline bone density concerns."

    return Prediction(
        mission_day=mission_day,
        cortisol_load=cortisol_load,
        cycle_variability_days=cycle_variability_days,
        bone_loss_pct_per_month=bone_loss_pct_per_month,
        drivers=drivers,
    )


# -------------------------
# Endpoint-style wrappers
# -------------------------

def predict_features(features: Dict[str, float]) -> Dict[str, Any]:
    """
    Uses ML model if available, otherwise fallback to simulator
    """

    if MODEL is not None:
        try:
            preds = ml_predict(MODEL, features)

            outputs = {
                "mission_day": int(features.get("mission_day", 1)),
                "cortisol_load": preds["cortisol"],
                "cycle_variability_days": preds["cycle"],
                "bone_loss_pct_per_month": preds["bone"],
                "drivers": {"model": "ML LinearRegression"}
            }

            return attach_risk_breakdown(outputs)

        except Exception as e:
            print("ML model failed, falling back:", e)

    # fallback
    pred = predict_from_features(features)

    outputs = {
        "mission_day": pred.mission_day,
        "cortisol_load": pred.cortisol_load,
        "cycle_variability_days": pred.cycle_variability_days,
        "bone_loss_pct_per_month": pred.bone_loss_pct_per_month,
        "drivers": pred.drivers,
    }

    return attach_risk_breakdown(outputs)


def predict_raw(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Raw mission inputs -> engineered features -> outputs.
    This is your end-to-end 'online endpoint' function.
    """
    feats = build_features(payload)
    outputs = predict_features(feats)
    return {"inputs": payload, "features": feats, "outputs": outputs}


# -------------------------
# Local demo
# -------------------------

if __name__ == "__main__":
    payload = {
        "mission": {
            "mission_day": 87,
            "gravity": "microgravity",
            "radiation_level": "high",
            "isolation_level": "high",
            "stress_level": 0.72,
            "sleep_hours_last_72h": 16.5
        },
        "history": {
            "cycle_irregularity_history": "moderate",
            "bone_density_concerns": "some",
            "sleep_disorders": "circadian",
            "prior_pcos_or_endometriosis": False
        }
    }

    res = predict_raw(payload)

    print("\n--- FEATURES ---")
    for k, v in res["features"].items():
        print(f"{k:28s} {v}")

    print("\n--- OUTPUTS ---")
    for k, v in res["outputs"].items():
        print(f"{k}: {v}")
