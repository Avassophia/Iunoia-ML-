"""
generate_datasets.py
--------------------
Generates multiple synthetic datasets for Iunoia-ML, varying:
  1. Random seed  — same logic, different noise realizations
  2. Scenario config — mission risk profiles (low / moderate / high / worst-case)
  3. Parameter sweep — weight combos from constants.json (stress_weight, etc.)

Output: one CSV per dataset in  data/synthetic/
Usage:
  python generate_datasets.py                    # default: all modes, defaults
  python generate_datasets.py --seeds 3          # 3 seed variants
  python generate_datasets.py --no-params        # skip parameter sweeps
  python generate_datasets.py --out my_dir/      # custom output folder
"""

import argparse
import copy
import csv
import json
import os
import random
import sys
from itertools import product
from pathlib import Path

# ── make model importable from repo root ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from model.features import build_features
from model.predictor import predict_from_features

# ── constants ─────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "mission_day", "gravity", "radiation_level", "isolation_level",
    "stress_score", "sleep_hours_last_72h", "cycle_irregularity_history",
    "bone_density_concerns", "sleep_disorders", "prior_pcos_or_endometriosis",
    "cycle", "bone", "cortisol", "test_group",
]

DEFAULT_CONSTANTS_PATH = Path(__file__).parent / "config" / "constants.json"

# ── helpers ────────────────────────────────────────────────────────────────────

def load_base_constants(path: Path = DEFAULT_CONSTANTS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def run_single(mission_day, gravity, radiation_level, isolation_level,
               stress_level, sleep_hours, cycle_irr, bone_concerns,
               sleep_disorders, pcos, *, constants_override: dict | None = None) -> dict:
    """
    Build features from raw inputs, run the predictor, return a flat row dict.
    constants_override: if provided, temporarily patches the loaded constants
                        (only affects predict_from_features via the CONST global).
    """
    import model.predictor as _pred_mod

    # Patch constants if requested
    original = None
    if constants_override:
        original = copy.deepcopy(_pred_mod.CONST)
        _pred_mod.CONST = constants_override

    payload = {
        "mission": {
            "mission_day": mission_day,
            "gravity": gravity,
            "radiation_level": radiation_level,
            "isolation_level": isolation_level,
            "stress_level": stress_level,
            "sleep_hours_last_72h": sleep_hours,
        },
        "history": {
            "cycle_irregularity_history": cycle_irr,
            "bone_density_concerns": bone_concerns,
            "sleep_disorders": sleep_disorders,
            "prior_pcos_or_endometriosis": pcos,
        },
    }

    feats = build_features(payload)
    pred  = predict_from_features(feats)

    if original is not None:
        _pred_mod.CONST = original

    return {
        "mission_day":                  mission_day,
        "gravity":                      gravity,
        "radiation_level":              radiation_level,
        "isolation_level":              isolation_level,
        "stress_score":                 stress_level,
        "sleep_hours_last_72h":         sleep_hours,
        "cycle_irregularity_history":   cycle_irr,
        "bone_density_concerns":        bone_concerns,
        "sleep_disorders":              sleep_disorders,
        "prior_pcos_or_endometriosis":  pcos,
        "cycle":                        pred.cycle_variability_days,
        "bone":                         pred.bone_loss_pct_per_month,
        "cortisol":                     pred.cortisol_load,
        "test_group":                   "placeholder",
    }


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → wrote {len(rows):>4} rows  {path}")


# ── core row generator (test-plan sections) ────────────────────────────────────

def generate_rows(rng: random.Random,
                  constants_override: dict | None = None) -> list[dict]:
    """
    Runs every section of the test plan and returns all rows.
    rng controls noise; constants_override patches the model weights.
    """
    rows = []

    def r(day, grav, rad, iso, stress, sh, irr, bone, slpd, pcos, group):
        row = run_single(day, grav, rad, iso, stress, sh, irr, bone, slpd, pcos,
                         constants_override=constants_override)
        row["test_group"] = group
        return row

    # ── §2 Baseline sanity ───────────────────────────────────────────────────
    for _ in range(40):
        rows.append(r(30, "microgravity", "moderate", "moderate",
                      0.5, 18.0, "none", "none", "none", False,
                      "baseline_sanity"))

    # ── §3 Single-variable sweeps ────────────────────────────────────────────
    sweep = [0.0, 0.25, 0.5, 0.75, 1.0]

    for v in sweep:
        rows.append(r(30, "microgravity", "moderate", "moderate",
                      v, 18.0, "none", "none", "none", False, "sweep_stress"))

    for v in sweep:
        sh = 21.0 * (1.0 - v)
        rows.append(r(30, "microgravity", "moderate", "moderate",
                      0.5, sh, "none", "none", "none", False, "sweep_sleep_deficit"))

    for iso in ["low", "moderate", "high"]:
        rows.append(r(30, "microgravity", "moderate", iso,
                      0.5, 18.0, "none", "none", "none", False, "sweep_isolation"))

    for rad in ["low", "moderate", "high"]:
        rows.append(r(30, "microgravity", rad, "moderate",
                      0.5, 18.0, "none", "none", "none", False, "sweep_radiation"))

    for grav in ["normal gravity", "low gravity", "microgravity"]:
        rows.append(r(30, grav, "moderate", "moderate",
                      0.5, 18.0, "none", "none", "none", False, "sweep_gravity"))

    for day in [1, 30, 60, 90, 120, 150, 180]:
        rows.append(r(day, "microgravity", "moderate", "moderate",
                      0.5, 18.0, "none", "none", "none", False, "sweep_mission_day"))

    # ── §4 Directionality ────────────────────────────────────────────────────
    for level, sv in [("low", 0.1), ("baseline", 0.5), ("high", 0.9)]:
        for _ in range(3):
            rows.append(r(30, "microgravity", "moderate", "moderate",
                          sv, 18.0, "none", "none", "none", False,
                          f"dir_stress_{level}"))

    for level, sh in [("low", 21.0), ("baseline", 18.0), ("high", 10.5)]:
        for _ in range(3):
            rows.append(r(30, "microgravity", "moderate", "moderate",
                          0.5, sh, "none", "none", "none", False,
                          f"dir_sleep_{level}"))

    # ── §5 Flag isolation ────────────────────────────────────────────────────
    for flag in [False, True]:
        for _ in range(10):
            rows.append(r(30, "microgravity", "moderate", "moderate",
                          0.5, 18.0, "none", "none", "none", flag,
                          f"flag_pcos_{'T' if flag else 'F'}"))

    for irr in ["none", "mild", "moderate", "severe"]:
        for _ in range(5):
            rows.append(r(30, "microgravity", "moderate", "moderate",
                          0.5, 18.0, irr, "none", "none", False,
                          f"flag_cycle_irr_{irr}"))

    # ── §6 Interaction scan ──────────────────────────────────────────────────
    for st, sh, sl_label in [(0.1, 20.5, "low"), (0.9, 20.5, "low"),
                              (0.1, 10.0, "high"), (0.9, 10.0, "high")]:
        for _ in range(5):
            rows.append(r(30, "microgravity", "moderate", "moderate",
                          st, sh, "none", "none", "none", False,
                          f"interact_stress_{st}_sleep_{sl_label}"))

    for st, iso in [(0.1, "low"), (0.9, "low"), (0.1, "high"), (0.9, "high")]:
        for _ in range(5):
            rows.append(r(30, "microgravity", "moderate", iso,
                          st, 18.0, "none", "none", "none", False,
                          f"interact_stress_{st}_iso_{iso}"))

    # ── §7 Noise stability ───────────────────────────────────────────────────
    for _ in range(30):
        ns  = rng.uniform(-0.05, 0.05)
        nsh = rng.uniform(-1.05, 1.05)
        rows.append(r(30, "microgravity", "moderate", "moderate",
                      max(0, min(1, 0.5 + ns)),
                      max(0, 18.0 + nsh),
                      "none", "none", "none", False,
                      "noise_stability"))

    # ── §8 Dead feature detection ────────────────────────────────────────────
    for bone in ["none", "some", "moderate", "significant"]:
        for _ in range(5):
            rows.append(r(30, "microgravity", "moderate", "moderate",
                          0.5, 18.0, "none", bone, "none", False,
                          f"dead_bone_{bone}"))

    for slpd in ["none", "mild", "circadian", "severe"]:
        for _ in range(5):
            rows.append(r(30, "microgravity", "moderate", "moderate",
                          0.5, 18.0, "none", "none", slpd, False,
                          f"dead_sleep_dis_{slpd}"))

    # ── §9 Collapse detection ────────────────────────────────────────────────
    for _ in range(10):
        rows.append(r(1, "normal gravity", "low", "low",
                      0.0, 21.0, "none", "none", "none", False,
                      "collapse_all_min"))

    for _ in range(10):
        rows.append(r(180, "microgravity", "high", "high",
                      1.0, 0.0, "severe", "significant", "severe", True,
                      "collapse_all_max"))

    for _ in range(10):
        rows.append(r(90, "microgravity", "high", "low",
                      0.1, 20.5, "none", "none", "none", False,
                      "collapse_high_grav_low_stress"))

    # ── Realistic longitudinal arcs ──────────────────────────────────────────
    arcs = [
        ("low_risk",    "earth",        "low",    "low",    0.2, 22.0, "none",     "none",        "none",     False),
        ("moderate",    "moon",         "medium", "medium", 0.5, 18.0, "mild",     "some",        "none",     False),
        ("high_risk",   "microgravity", "high",   "high",   0.8, 14.0, "moderate", "some",        "circadian",True),
        ("worst_case",  "microgravity", "high",   "high",   0.95,10.0, "severe",   "significant", "severe",   True),
    ]
    for label, grav, rad, iso, st, sh, irr, bone, slpd, pcos in arcs:
        for day in range(5, 185, 5):
            rows.append(r(day, grav, rad, iso, st, sh, irr, bone, slpd, pcos,
                          f"arc_{label}"))

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — Seed variants
# ══════════════════════════════════════════════════════════════════════════════

def generate_seed_variants(n_seeds: int, out_dir: Path) -> None:
    """
    Same test-plan logic, different noise realizations (noise_stability rows differ).
    """
    print(f"\n[MODE 1] Seed variants  ({n_seeds} datasets)")
    base_seeds = [42, 7, 13, 99, 256, 512, 1024, 2048, 3141, 9999]
    seeds = base_seeds[:n_seeds] if n_seeds <= len(base_seeds) else \
            base_seeds + list(range(10000, 10000 + n_seeds - len(base_seeds)))

    for seed in seeds:
        rng = random.Random(seed)
        rows = generate_rows(rng)
        write_csv(rows, out_dir / f"seed_{seed:05d}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — Scenario configs
# ══════════════════════════════════════════════════════════════════════════════

# Each scenario is a (name, list-of-(group_label, run_kwargs)) pair.
# We generate the full test-plan rows AND add a scenario-specific dense sweep.

SCENARIO_PROFILES = {
    "low_risk": dict(
        gravity="earth", radiation_level="low", isolation_level="low",
        stress_level=0.2, sleep_hours=22.0,
        cycle_irr="none", bone_concerns="none",
        sleep_disorders="none", pcos=False,
    ),
    "moderate_risk": dict(
        gravity="moon", radiation_level="medium", isolation_level="medium",
        stress_level=0.5, sleep_hours=18.0,
        cycle_irr="mild", bone_concerns="some",
        sleep_disorders="none", pcos=False,
    ),
    "high_risk": dict(
        gravity="microgravity", radiation_level="high", isolation_level="high",
        stress_level=0.8, sleep_hours=14.0,
        cycle_irr="moderate", bone_concerns="some",
        sleep_disorders="circadian", pcos=True,
    ),
    "worst_case": dict(
        gravity="microgravity", radiation_level="high", isolation_level="high",
        stress_level=0.95, sleep_hours=10.0,
        cycle_irr="severe", bone_concerns="significant",
        sleep_disorders="severe", pcos=True,
    ),
    "pcos_focus": dict(
        gravity="microgravity", radiation_level="moderate", isolation_level="moderate",
        stress_level=0.6, sleep_hours=16.0,
        cycle_irr="moderate", bone_concerns="some",
        sleep_disorders="none", pcos=True,
    ),
    "bone_focus": dict(
        gravity="microgravity", radiation_level="high", isolation_level="low",
        stress_level=0.3, sleep_hours=20.0,
        cycle_irr="none", bone_concerns="significant",
        sleep_disorders="none", pcos=False,
    ),
}


def generate_scenario_variants(out_dir: Path) -> None:
    """
    One dataset per scenario profile. Runs the full test plan with the profile's
    values locked in for the longitudinal arc, plus a dense day-by-day sweep.
    """
    print(f"\n[MODE 2] Scenario variants  ({len(SCENARIO_PROFILES)} datasets)")
    rng = random.Random(42)

    for scenario_name, profile in SCENARIO_PROFILES.items():
        rows = generate_rows(rng)   # standard test-plan rows

        # Dense longitudinal sweep specific to this scenario
        for day in range(1, 181):
            row = run_single(
                day,
                profile["gravity"], profile["radiation_level"], profile["isolation_level"],
                profile["stress_level"], profile["sleep_hours"],
                profile["cycle_irr"], profile["bone_concerns"],
                profile["sleep_disorders"], profile["pcos"],
            )
            row["test_group"] = f"scenario_arc_{scenario_name}"
            rows.append(row)

        write_csv(rows, out_dir / f"scenario_{scenario_name}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 — Parameter sweeps (constants.json weight combos)
# ══════════════════════════════════════════════════════════════════════════════

def _make_constants_variant(base: dict, overrides: dict) -> dict:
    """Deep-copy base constants and apply a flat dict of dot-path overrides."""
    import copy
    c = copy.deepcopy(base)
    for dot_path, val in overrides.items():
        keys = dot_path.split(".")
        node = c
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = val
    return c


# Each entry: (label_suffix, {dot.path: value, ...})
# Dot-paths follow the structure of constants.json
PARAM_SWEEP_VARIANTS = [
    # ── cortisol stress weight ──────────────────────────────────────────────
    ("cort_stress_low",   {"cortisol.stress_weight": 0.20,
                           "cortisol.sleep_deficit_weight": 0.55}),
    ("cort_stress_high",  {"cortisol.stress_weight": 0.60,
                           "cortisol.sleep_deficit_weight": 0.15}),

    # ── cortisol isolation weight ────────────────────────────────────────────
    ("cort_iso_heavy",    {"cortisol.isolation_weight": 0.35,
                           "cortisol.interaction_weight": 0.05,
                           "cortisol.stress_weight": 0.30,
                           "cortisol.sleep_deficit_weight": 0.30}),

    # ── bone gravity weight ──────────────────────────────────────────────────
    ("bone_grav_low",     {"bone_loss.gravity_weight": 0.30,
                           "bone_loss.radiation_gravity_weight": 0.40}),
    ("bone_grav_high",    {"bone_loss.gravity_weight": 0.75,
                           "bone_loss.radiation_gravity_weight": 0.10}),

    # ── cycle history sensitivity ────────────────────────────────────────────
    ("cycle_history_high",{"cycle_variability.history_weight": 0.25,
                           "cycle_variability.reproductive_weight": 0.20,
                           "cycle_variability.stress_weight": 0.20,
                           "cycle_variability.sleep_deficit_weight": 0.20,
                           "cycle_variability.isolation_weight": 0.10,
                           "cycle_variability.radiation_weight": 0.05}),

    # ── risk bucket thresholds ───────────────────────────────────────────────
    ("buckets_strict",    {"risk_buckets.low_threshold": 0.20,
                           "risk_buckets.moderate_threshold": 0.55}),
    ("buckets_lenient",   {"risk_buckets.low_threshold": 0.50,
                           "risk_buckets.moderate_threshold": 0.80}),
]


def generate_param_variants(out_dir: Path) -> None:
    """
    One dataset per weight-combo variant. Same test-plan rows, different CONST.
    """
    print(f"\n[MODE 3] Parameter sweep variants  ({len(PARAM_SWEEP_VARIANTS)} datasets)")
    base_constants = load_base_constants()
    rng = random.Random(42)

    for label, overrides in PARAM_SWEEP_VARIANTS:
        constants_variant = _make_constants_variant(base_constants, overrides)
        rows = generate_rows(rng, constants_override=constants_variant)
        write_csv(rows, out_dir / f"params_{label}.csv")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds",      type=int, default=5,
                   help="Number of seed-variant datasets to generate (default: 5)")
    p.add_argument("--no-seeds",   action="store_true", help="Skip seed variants")
    p.add_argument("--no-scenarios", action="store_true", help="Skip scenario variants")
    p.add_argument("--no-params",  action="store_true", help="Skip parameter sweep variants")
    p.add_argument("--out",        type=str, default="data/synthetic",
                   help="Output directory (default: data/synthetic)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(__file__).parent / args.out
    print(f"Output directory: {out_dir}")

    if not args.no_seeds:
        generate_seed_variants(args.seeds, out_dir / "seeds")

    if not args.no_scenarios:
        generate_scenario_variants(out_dir / "scenarios")

    if not args.no_params:
        generate_param_variants(out_dir / "params")

    all_files = list(out_dir.rglob("*.csv"))
    print(f"\nDone. {len(all_files)} CSV files written under {out_dir}/")


if __name__ == "__main__":
    main()
