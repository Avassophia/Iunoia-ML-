def build_features(payload: dict) -> dict:
    mission = payload.get("mission", {})
    history = payload.get("history", {})

    gravity_map = {
        "microgravity": 1.0,
        "low gravity": 0.5,
        "normal gravity": 0.0
    }

    radiation_map = {
        "low": 0.2,
        "moderate": 0.5,
        "high": 0.9
    }

    isolation_map = {
        "low": 0.2,
        "moderate": 0.5,
        "high": 0.9
    }

    irregularity_map = {
        "none": 0.0,
        "mild": 0.2,
        "moderate": 0.5,
        "severe": 1.0
    }

    bone_map = {
        "none": 0.0,
        "some": 0.3,
        "moderate": 0.6,
        "significant": 1.0
    }

    sleep_disorder_map = {
        "none": 0.0,
        "mild": 0.3,
        "circadian": 0.6,
        "severe": 1.0
    }

    stress = float(mission.get("stress_level", 0.5))
    sleep_hours = float(mission.get("sleep_hours_last_72h", 18.0))
    sleep_deficit = max(0.0, (21.0 - sleep_hours) / 21.0)

    g   = gravity_map.get(mission.get("gravity", "microgravity"), 1.0)
    r   = radiation_map.get(mission.get("radiation_level", "moderate"), 0.5)
    iso = isolation_map.get(mission.get("isolation_level", "moderate"), 0.5)

    irr  = irregularity_map.get(history.get("cycle_irregularity_history", "none"), 0.0)
    bone = bone_map.get(history.get("bone_density_concerns", "none"), 0.0)
    slpd = sleep_disorder_map.get(history.get("sleep_disorders", "none"), 0.0)
    pcos = 1.0 if history.get("prior_pcos_or_endometriosis", False) else 0.0

    return {
        "mission_day":                int(mission.get("mission_day", 1)),
        "gravity_factor":             g,
        "radiation_factor":           r,
        "isolation_factor":           iso,
        "stress_score":               stress,
        "sleep_deficit":              sleep_deficit,
        "cycle_irregularity_history": irr,
        "bone_density_concerns":      bone,
        "sleep_disorders":            slpd,
        "pcos_or_endometriosis_flag": pcos,
        "stress_x_sleep":             stress * sleep_deficit,
        "rad_x_grav":                 r * g,
        "iso_x_stress":               iso * stress,
    }