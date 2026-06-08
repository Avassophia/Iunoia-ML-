import sys
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model.predictor import predict_raw

# create our input, here all inputs are 0.5 and all flags are false
def make_payload(
    stress=0.5,
    sleep_hours=18.0,
    mission_day=90,
    gravity="microgravity",
    radiation="moderate",
    isolation="moderate",
    cycle_irregularity="none",
    bone_concerns="none",
    sleep_disorders="none",
    pcos=False
):
    return {
        "mission": {
            "mission_day": mission_day,
            "gravity": gravity,
            "radiation_level": radiation,
            "isolation_level": isolation,
            "stress_level": stress,
            "sleep_hours_last_72h": sleep_hours
        },
        "history": {
            "cycle_irregularity_history": cycle_irregularity,
            "bone_density_concerns": bone_concerns,
            "sleep_disorders": sleep_disorders,
            "prior_pcos_or_endometriosis": pcos
        }
    }


# # ============================================================
# # SECTION 2 - Baseline sanity check
# # ============================================================

print()
print("SECTION 2: BASELINE SANITY CHECK")
print()

results = []

#run 30 iterations
for i in range(30):
    result = predict_raw(make_payload())
    outputs = result["outputs"]
    results.append({
        "cortisol": outputs["cortisol_load"],
        "cycle":    outputs["cycle_variability_days"],
        "bone":     outputs["bone_loss_pct_per_month"],
        "risk":     outputs["risk_score"]
    })

df_baseline = pd.DataFrame(results)
print("\nBaselineresults across 30 runs:")
print(df_baseline.describe())
print("\nVariance:")
print(df_baseline.var())
print("\nIs output stable (std < 0.01)?", (df_baseline.std() < 0.01).all())

# RESULTS: PASS
# mean values : cortisol=19.5, cycle=3.62, bone=1.59, risk=0.37
# std=0 across all 30 runs, fully deterministic, stable baseline confirmed
# PASS


# SECTION 3 - Single variable responsiveness
print()
# print("SECTION 3: SINGLE VARIABLE RESPONSIVENESS")
print()
levels = [0.0, 0.25, 0.5, 0.75, 1.0]
sleep_levels = [21.0, 17.0, 14.0, 10.5, 7.0]
radiation_levels = ["low", "moderate", "high"]
isolation_levels = ["low", "moderate", "high"]
gravity_levels = ["normal gravity", "low gravity", "microgravity"]
mission_days = [1, 45, 90, 135, 180]
irregularity_levels = ["none", "mild", "moderate", "severe"]
bone_levels = ["none", "some", "moderate", "significant"]
sleep_disorder_levels = ["none", "mild", "circadian", "severe"]

def get_outputs(payload):
    r = predict_raw(payload)["outputs"]
    return r["cortisol_load"], r["cycle_variability_days"], r["bone_loss_pct_per_month"]

# stress
s_c, s_cy, s_b = zip(*[get_outputs(make_payload(stress=v)) for v in levels])

# sleep
sl_c, sl_cy, sl_b = zip(*[get_outputs(make_payload(sleep_hours=v)) for v in sleep_levels])

# radiation
ra_c, ra_cy, ra_b = zip(*[get_outputs(make_payload(radiation=v)) for v in radiation_levels])

# isolation
i_c, i_cy, i_b = zip(*[get_outputs(make_payload(isolation=v)) for v in isolation_levels])

# gravity
g_c, g_cy, g_b = zip(*[get_outputs(make_payload(gravity=v)) for v in gravity_levels])

# mission day
d_c, d_cy, d_b = zip(*[get_outputs(make_payload(mission_day=v)) for v in mission_days])

# cycle irregularity
ci_c, ci_cy, ci_b = zip(*[get_outputs(make_payload(cycle_irregularity=v)) for v in irregularity_levels])

# bone concerns
bc_c, bc_cy, bc_b = zip(*[get_outputs(make_payload(bone_concerns=v)) for v in bone_levels])

# sleep disorders
sd_c, sd_cy, sd_b = zip(*[get_outputs(make_payload(sleep_disorders=v)) for v in sleep_disorder_levels])

# pcos
p_c, p_cy, p_b = zip(*[get_outputs(make_payload(pcos=v)) for v in [False, True]])

fig, axes = plt.subplots(9, 3, figsize=(14, 36))
fig.suptitle("Section 3: Single Variable Responsiveness", fontsize=14)

def plot_row(ax_row, x_labels, c, cy, b, title):
    ax_row[0].plot(range(len(x_labels)), c, marker="o")
    ax_row[0].set_title(f"{title} → Cortisol")
    ax_row[0].set_xticks(range(len(x_labels)))
    ax_row[0].set_xticklabels(x_labels, rotation=15)

    ax_row[1].plot(range(len(x_labels)), cy, marker="o", color="gold")
    ax_row[1].set_title(f"{title} → Cycle")
    ax_row[1].set_xticks(range(len(x_labels)))
    ax_row[1].set_xticklabels(x_labels, rotation=15)

    ax_row[2].plot(range(len(x_labels)), b, marker="o", color="magenta")
    ax_row[2].set_title(f"{title} → Bone Loss")
    ax_row[2].set_xticks(range(len(x_labels)))
    ax_row[2].set_xticklabels(x_labels, rotation=15)

plot_row(axes[0], [str(v) for v in levels],             s_c,  s_cy,  s_b,  "Stress")
plot_row(axes[1], [str(v) for v in sleep_levels],       sl_c, sl_cy, sl_b, "Sleep hours")
plot_row(axes[2], radiation_levels,                     ra_c, ra_cy, ra_b, "Radiation")
plot_row(axes[3], isolation_levels,                     i_c,  i_cy,  i_b,  "Isolation")
plot_row(axes[4], gravity_levels,                       g_c,  g_cy,  g_b,  "Gravity")
plot_row(axes[5], [str(v) for v in mission_days],       d_c,  d_cy,  d_b,  "Mission day")
plot_row(axes[6], irregularity_levels,                  ci_c, ci_cy, ci_b, "Cycle irregularity")
plot_row(axes[7], bone_levels,                          bc_c, bc_cy, bc_b, "Bone concerns")
plot_row(axes[8], ["False", "True"],                    p_c,  p_cy,  p_b,  "PCOS flag")

plt.tight_layout()
plt.show()

print("\nDirectionality summary:")
print("Stress cortisol increasing?",    list(s_c)  == sorted(s_c))
print("Sleep cortisol increasing?",     list(sl_c) == sorted(sl_c))
print("Radiation cortisol increasing?", list(ra_c) == sorted(ra_c))
print("Isolation cortisol increasing?", list(i_c)  == sorted(i_c))
print("Mission day cortisol increasing?",list(d_c) == sorted(d_c))

#RESULTS - PASS, all variables look good, have linear lines, are responding appropriately
#no dead variables 
# Stress cortisol increasing? True
# Sleep cortisol increasing? True
# Radiation cortisol increasing? True
# Isolation cortisol increasing? True
# Mission day cortisol increasing? True

# # SECTION 4 - Directionality check
#comprehensive check for ALL predictors 

checks = {
    "stress_low":           make_payload(stress=0.1),
    "stress_mid":           make_payload(stress=0.5),
    "stress_high":          make_payload(stress=0.9),
    "sleep_low":            make_payload(sleep_hours=21.0),
    "sleep_mid":            make_payload(sleep_hours=14.0),
    "sleep_high":           make_payload(sleep_hours=7.0),
    "radiation_low":        make_payload(radiation="low"),
    "radiation_mid":        make_payload(radiation="moderate"),
    "radiation_high":       make_payload(radiation="high"),
    "isolation_low":        make_payload(isolation="low"),
    "isolation_mid":        make_payload(isolation="moderate"),
    "isolation_high":       make_payload(isolation="high"),
    "gravity_none":         make_payload(gravity="normal gravity"),
    "gravity_low":          make_payload(gravity="low gravity"),
    "gravity_micro":        make_payload(gravity="microgravity"),
    "mission_day_1":        make_payload(mission_day=1),
    "mission_day_90":       make_payload(mission_day=90),
    "mission_day_180":      make_payload(mission_day=180),
    "irregularity_none":    make_payload(cycle_irregularity="none"),
    "irregularity_mild":    make_payload(cycle_irregularity="mild"),
    "irregularity_severe":  make_payload(cycle_irregularity="severe"),
    "bone_none":            make_payload(bone_concerns="none"),
    "bone_some":            make_payload(bone_concerns="some"),
    "bone_significant":     make_payload(bone_concerns="significant"),
    "sleep_dis_none":       make_payload(sleep_disorders="none"),
    "sleep_dis_circadian":  make_payload(sleep_disorders="circadian"),
    "sleep_dis_severe":     make_payload(sleep_disorders="severe"),
    "pcos_off":             make_payload(pcos=False),
    "pcos_on":              make_payload(pcos=True),
}


for label, payload in checks.items():
    r = predict_raw(payload)["outputs"]
    print(f"{label:25s} → cortisol={r['cortisol_load']:6.1f}  "
           f"cycle={r['cycle_variability_days']:5.2f}  "
           f"bone={r['bone_loss_pct_per_month']:5.2f}  "
           f"risk={r['risk_score']:.2f}")


# EXPECTED DIRECTIONS:
# stress has a positive correlation with all outputs
# sleep deprivation has a positive correlation with all outputs
# radiation has a positive correlation with cortisol, cycle, and bone
# isolation has a positive correlation with cortisol and cycle
# gravity has a positive correlation with bone
# mission day has a positive correlation with cortisol
# cycle irregularity has a positive correlation with cycle
# bone concerns has a positive correlation with bone
# sleep disorder has a positive correlation with cortisol
# pcos has a positive correlation with cortisol and cycle

#RESULTS - PASS, everything moves in CORRECT direction.
# NOTE: radiation has no effect on cortisol. 
# Also, sleep disorder effect on cortisol is very small.


# # SECTION 5 - Flag behavior isolation

print("SECTION 5: FLAG BEHAVIOR ISOLATION")
r_pcos_off = predict_raw(make_payload(pcos=False))["outputs"]
r_pcos_on  = predict_raw(make_payload(pcos=True))["outputs"]

print("PCOS OFF → cortisol:", r_pcos_off["cortisol_load"],
      "| cycle:", r_pcos_off["cycle_variability_days"],
      "| risk:", r_pcos_off["risk_score"])
print("PCOS ON  → cortisol:", r_pcos_on["cortisol_load"],
      "| cycle:", r_pcos_on["cycle_variability_days"],
      "| risk:", r_pcos_on["risk_score"])

print("\nDelta cortisol:", round(r_pcos_on["cortisol_load"]          - r_pcos_off["cortisol_load"], 3))
print("Delta cycle:   ", round(r_pcos_on["cycle_variability_days"]   - r_pcos_off["cycle_variability_days"], 3))
print("Delta risk:    ", round(r_pcos_on["risk_score"]               - r_pcos_off["risk_score"], 3))
print("\nFlag is ACTIVE if delta > 0")

print("\nCycle irregularity history:")
for level in ["none", "moderate", "severe"]:
    r = predict_raw(make_payload(cycle_irregularity=level))["outputs"]
    print(f"  {level:10s} → cycle={r['cycle_variability_days']:5.2f}  risk={r['risk_score']:.2f}")

# #RESULTS: PASS, both flags are active and move in the right biological direction.
# # PCOS worsens cortisol and cycle
# # irregularity history worsens cycle variability 


# # # SECTION 6 - Interaction scan
print()
print("SECTION 6: INTERACTION SCAN")
print()

scenarios = {
    "both low    (stress=0.1, sleep=21h)": make_payload(stress=0.1, sleep_hours=21.0),
    "stress high (stress=0.9, sleep=21h)": make_payload(stress=0.9, sleep_hours=21.0),
    "sleep low   (stress=0.1, sleep=7h) ": make_payload(stress=0.1, sleep_hours=7.0),
    "both high   (stress=0.9, sleep=7h) ": make_payload(stress=0.9, sleep_hours=7.0),
}

print("\nStress x Sleep Deficit interaction:")
for label, payload in scenarios.items():
    r = predict_raw(payload)["outputs"]
    print(f"  {label} → cortisol={r['cortisol_load']:6.1f}  risk={r['risk_score']:.2f}")

print("\nExpected: 'both high' should be worse than either alone")
print("If 'both high' ≈ 'one high' → interaction term not working")
#RESULT - PASS! having stress and sleep deprivation both high is worse
#than having only one high. Interaction term is working.


# # SECTION 7 - Stability / noise test
print()
# print("SECTION 7: STABILITY / NOISE TEST")
print()

noisy_results = []

for i in range(20):
    noisy_stress = 0.5 + random.uniform(-0.05, 0.05)
    noisy_sleep  = 18.0 + random.uniform(-0.5, 0.5)

    r = predict_raw(make_payload(
        stress=round(noisy_stress, 3),
        sleep_hours=round(noisy_sleep, 2)
    ))["outputs"]

    noisy_results.append({
        "cortisol": r["cortisol_load"],
        "cycle":    r["cycle_variability_days"],
        "bone":     r["bone_loss_pct_per_month"],
        "risk":     r["risk_score"]
    })

df_noise = pd.DataFrame(noisy_results)
print("\nNoisy run stats:")
print(df_noise.describe())
print("\nStd dev (should be small):")
print(df_noise.std())
print("\nModel stable under small noise?", (df_noise.std() < 5.0).all())
# #RESULTS: PASS, no outputs wildly jump, variables are stable.

# # # SECTION 8 - Dead feature detection
print()
# print("SECTION 8: DEAD FEATURE DETECTION")
print()
def feature_range_impact(feature_name, low_payload, high_payload):
    r_low  = predict_raw(low_payload)["outputs"]
    r_high = predict_raw(high_payload)["outputs"]

    delta_cortisol = abs(r_high["cortisol_load"]            - r_low["cortisol_load"])
    delta_cycle    = abs(r_high["cycle_variability_days"]   - r_low["cycle_variability_days"])
    delta_bone     = abs(r_high["bone_loss_pct_per_month"]  - r_low["bone_loss_pct_per_month"])
    delta_risk     = abs(r_high["risk_score"]               - r_low["risk_score"])

    is_dead = delta_cortisol < 1.0 and delta_cycle < 0.1 and delta_bone < 0.05

    print(f"\n{feature_name}:")
    print(f"  delta cortisol={delta_cortisol:.2f}  delta cycle={delta_cycle:.3f}  "
          f"delta bone={delta_bone:.3f}  delta risk={delta_risk:.3f}")
    print(f"  Status: {'DEAD - candidate for reweighting' if is_dead else 'ACTIVE'}")

feature_range_impact("Stress (0.0 to 1.0)",
    make_payload(stress=0.0), make_payload(stress=1.0))

feature_range_impact("Sleep hours (21h to 7h)",
    make_payload(sleep_hours=21.0), make_payload(sleep_hours=7.0))

feature_range_impact("PCOS flag (False to True)",
    make_payload(pcos=False), make_payload(pcos=True))

feature_range_impact("Cycle irregularity (none to severe)",
    make_payload(cycle_irregularity="none"),
    make_payload(cycle_irregularity="severe"))

feature_range_impact("Radiation (low to high)",
    make_payload(radiation="low"), make_payload(radiation="high"))

feature_range_impact("Isolation (low to high)",
    make_payload(isolation="low"), make_payload(isolation="high"))

feature_range_impact("Gravity (normal to microgravity)",
    make_payload(gravity="normal gravity"), make_payload(gravity="microgravity"))

feature_range_impact("Mission day (1 to 180)",
    make_payload(mission_day=1), make_payload(mission_day=180))

feature_range_impact("Bone concerns (none to significant)",
    make_payload(bone_concerns="none"), make_payload(bone_concerns="significant"))

feature_range_impact("Sleep disorders (none to severe)",
    make_payload(sleep_disorders="none"), make_payload(sleep_disorders="severe"))

#RESULTS: PASS. No dead features - all are active 

# SECTION 9 - Collapse detection
print()
# print("SECTION 9: COLLAPSE DETECTION")
print()

extreme_scenarios = {
    "all minimum": make_payload(stress=0.0, sleep_hours=21.0,
                                radiation="low", isolation="low"),
    "all maximum": make_payload(stress=1.0, sleep_hours=7.0,
                                radiation="high", isolation="high",
                                pcos=True, cycle_irregularity="severe"),
    "baseline":    make_payload()
}
cortisol_values = []
risk_values     = []

for label, payload in extreme_scenarios.items():
    r = predict_raw(payload)["outputs"]
    cortisol_values.append(r["cortisol_load"])
    risk_values.append(r["risk_score"])
    print(f"{label:15s} → cortisol={r['cortisol_load']:6.1f}  "
          f"cycle={r['cycle_variability_days']:5.2f}  "
          f"risk={r['risk_score']:.2f}")

print("\nConstant output? (bad):",
      len(set([round(v, 1) for v in cortisol_values])) == 1)
print("Adequate range (max - min > 10)?",
      max(cortisol_values) - min(cortisol_values) > 10)
print("No saturation (risk values differ)?",
      len(set([round(v, 1) for v in risk_values])) > 1)
#RESULTS: PASS, output changes dynamically, we have adequate range,
#and there is no saturation 


# # SECTION 10 - Minimal acceptance criteria
print()
# print("SECTION 10: MINIMAL ACCEPTANCE CRITERIA")
print()

r_stress_low  = predict_raw(make_payload(stress=0.1))["outputs"]
r_stress_high = predict_raw(make_payload(stress=0.9))["outputs"]
r_sleep_low   = predict_raw(make_payload(sleep_hours=21.0))["outputs"]
r_sleep_high  = predict_raw(make_payload(sleep_hours=7.0))["outputs"]
r_pcos_off    = predict_raw(make_payload(pcos=False))["outputs"]
r_pcos_on     = predict_raw(make_payload(pcos=True))["outputs"]
r_noisy_1     = predict_raw(make_payload(stress=0.52, sleep_hours=18.3))["outputs"]
r_noisy_2     = predict_raw(make_payload(stress=0.48, sleep_hours=17.7))["outputs"]

criteria = {
    "Stress has visible influence on cortisol":
        abs(r_stress_high["cortisol_load"] - r_stress_low["cortisol_load"]) > 5,

    "Sleep has visible influence on cortisol":
        abs(r_sleep_high["cortisol_load"] - r_sleep_low["cortisol_load"]) > 5,

    "PCOS flag produces step change in cycle":
        abs(r_pcos_on["cycle_variability_days"] - r_pcos_off["cycle_variability_days"]) > 0.1,

    "Model stable under small noise":
        abs(r_noisy_1["cortisol_load"] - r_noisy_2["cortisol_load"]) < 10,

    "Stress direction correct (high > low)":
        r_stress_high["cortisol_load"] > r_stress_low["cortisol_load"],

    "Sleep direction correct (less sleep = higher cortisol)":
    #r_sleep_high indicates high sleep deprivation
        r_sleep_high["cortisol_load"] > r_sleep_low["cortisol_load"],
}

all_passed = True
for criterion, passed in criteria.items():
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"{status} - {criterion}")

print("\n" + "=" * 50)
print("OVERALL:", "ALL CRITERIA MET - ready to proceed"
      if all_passed else "FAILURES DETECTED - review predictor.py before proceeding")
print("=" * 50)
#RESULTS: PASS
#Detailed results:
# PASS - Stress has visible influence on cortisol
# PASS - Sleep has visible influence on cortisol
# PASS - PCOS flag produces step change in cycle
# PASS - Model stable under small noise
# PASS - Stress direction correct (high > low)
# PASS - Sleep direction correct (less sleep = higher cortisol)
#All conditions met! 