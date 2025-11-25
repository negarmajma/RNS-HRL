# generate_scenarios.py
"""
Generate synthetic scenarios for training HRL / Q-Learning.

Output: data/scenarios_augmented.csv
Each row: scenario_id, b1..b5, HR, Temp, BP, expert_action, fuzzy_state
"""

import os
import csv
import math
import numpy as np

from data_loader import load_data
from tcpn import TCPN
from fuzzy_system import FuzzySystem


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def generate_physio_profiles(base_hr: float, base_temp: float, base_bp: float):
    """
    Generate three simple physiological profiles around a base value:
      - normal: original values
      - mild: slightly increased
      - high: clearly more risky
    """
    profiles = []

    # Profile 1: normal (original)
    profiles.append(("normal", np.array([base_hr, base_temp, base_bp], dtype=float)))

    # Profile 2: mild stress
    profiles.append(("mild", np.array([
        base_hr + 10.0,
        base_temp + 0.3,
        base_bp + 10.0
    ], dtype=float)))

    # Profile 3: high risk
    profiles.append(("high", np.array([
        base_hr + 20.0,
        base_temp + 0.7,
        base_bp + 20.0
    ], dtype=float)))

    # Clip to physiologically reasonable ranges
    for _, p in profiles:
        p[0] = np.clip(p[0], 40.0, 180.0)   # HR
        p[1] = np.clip(p[1], 35.0, 40.0)    # Temp
        p[2] = np.clip(p[2], 80.0, 200.0)   # BP

    return profiles


def classify_risk(fuzzy_state: str, expert_action: int) -> str:
    """
    Simple risk level classification to control number of samples.
    """
    if fuzzy_state == "S1" and expert_action == 0:
        return "low"
    if fuzzy_state == "S2" and expert_action == 0:
        return "mid"
    # Everything else (S3/S4 or expert=1) is considered high risk
    return "high"


def add_gaussian_noise_to_scenario(base_scenario: dict,
                                   tcpn: TCPN,
                                   fuzzy: FuzzySystem,
                                   brain_sigma_ratio: float = 0.05,
                                   hr_sigma: float = 5.0,
                                   temp_sigma: float = 0.2,
                                   bp_sigma: float = 7.0,
                                   keep_same_fuzzy: bool = True):
    """
    Generate a new noisy scenario with small Gaussian noise.
    If keep_same_fuzzy=True, only keep scenarios whose fuzzy_state
    stays the same as the base scenario.
    """
    b = np.array([base_scenario[f"b{i}"] for i in range(1, 6)], dtype=float)
    hr = float(base_scenario["HR"])
    temp = float(base_scenario["Temp"])
    bp = float(base_scenario["BP"])

    # Noise on 5 brain features
    brain_sigma = np.maximum(np.abs(b) * brain_sigma_ratio, 1e-3)
    b_noisy = b + np.random.normal(loc=0.0, scale=brain_sigma, size=b.shape)

    # Noise on HR / Temp / BP
    hr_noisy = hr + np.random.normal(0.0, hr_sigma)
    temp_noisy = temp + np.random.normal(0.0, temp_sigma)
    bp_noisy = bp + np.random.normal(0.0, bp_sigma)

    # Clip physiology
    hr_noisy = float(np.clip(hr_noisy, 40.0, 180.0))
    temp_noisy = float(np.clip(temp_noisy, 35.0, 40.0))
    bp_noisy = float(np.clip(bp_noisy, 80.0, 200.0))

    # Compute new fuzzy_state from brain features
    tcpn_state = tcpn.step(b_noisy)
    fuzzy_state = fuzzy.get_state(tcpn_state)

    if keep_same_fuzzy and fuzzy_state != base_scenario["fuzzy_state"]:
        return None  # discard this scenario

    new_scenario = {
        "b1": float(b_noisy[0]),
        "b2": float(b_noisy[1]),
        "b3": float(b_noisy[2]),
        "b4": float(b_noisy[3]),
        "b5": float(b_noisy[4]),
        "HR": hr_noisy,
        "Temp": temp_noisy,
        "BP": bp_noisy,
        "expert_action": int(base_scenario["expert_action"]),
        "fuzzy_state": fuzzy_state,
        "risk_level": classify_risk(fuzzy_state, int(base_scenario["expert_action"])),
    }

    return new_scenario


def main(output_path: str = "data/scenarios_augmented.csv",
         seed: int = 42):
    np.random.seed(seed)

    ensure_dir(os.path.dirname(output_path) or ".")

    brain_sensors, physio_sensors, expert_actions = load_data()
    tcpn = TCPN()
    fuzzy = FuzzySystem()

    base_scenarios = []

    # 1) Build base scenarios (brain + three physiological profiles)
    scenario_id = 0
    for i in range(len(brain_sensors)):
        b = brain_sensors[i]
        hr0, temp0, bp0 = physio_sensors[i]
        expert = int(expert_actions[i])

        # fuzzy_state based on brain only
        tcpn_state = tcpn.step(b)
        fuzzy_state = fuzzy.get_state(tcpn_state)

        profiles = generate_physio_profiles(hr0, temp0, bp0)

        for profile_name, phys in profiles:
            hr, temp, bp = phys
            risk_level = classify_risk(fuzzy_state, expert)

            scen = {
                "scenario_id": scenario_id,
                "b1": float(b[0]),
                "b2": float(b[1]),
                "b3": float(b[2]),
                "b4": float(b[3]),
                "b5": float(b[4]),
                "HR": float(hr),
                "Temp": float(temp),
                "BP": float(bp),
                "expert_action": expert,
                "fuzzy_state": fuzzy_state,
                "risk_level": risk_level,
                "profile": profile_name,
            }
            base_scenarios.append(scen)
            scenario_id += 1

    print(f"Number of base scenarios (with physiological profiles): {len(base_scenarios)}")

    # 2) Generate noisy scenarios based on risk level
    augmented_scenarios = list(base_scenarios)  # keep the base ones as well

    # You can tune these counts later if needed
    noisy_counts = {
        "low": 80,
        "mid": 40,
        "high": 20,
    }

    added_noisy = 0
    for base in base_scenarios:
        risk = base["risk_level"]
        n_noisy = noisy_counts.get(risk, 20)

        for _ in range(n_noisy):
            new_scen = add_gaussian_noise_to_scenario(
                base_scenario=base,
                tcpn=tcpn,
                fuzzy=fuzzy,
                keep_same_fuzzy=True,
            )
            if new_scen is not None:
                new_scen["scenario_id"] = scenario_id
                augmented_scenarios.append(new_scen)
                scenario_id += 1
                added_noisy += 1

    print(f"Number of noisy scenarios added: {added_noisy}")

    # 3) In-class interpolation on base scenarios
    from itertools import combinations

    groups = {}
    for scen in base_scenarios:
        key = (scen["fuzzy_state"], int(scen["expert_action"]))
        groups.setdefault(key, []).append(scen)

    added_interp = 0
    lambdas = [0.25, 0.5, 0.75]

    for key, group in groups.items():
        if len(group) < 2:
            continue
        pairs = list(combinations(group, 2))

        for s1, s2 in pairs:
            b1_vec = np.array([s1[f"b{i}"] for i in range(1, 6)], dtype=float)
            b2_vec = np.array([s2[f"b{i}"] for i in range(1, 6)], dtype=float)
            phys1 = np.array([s1["HR"], s1["Temp"], s1["BP"]], dtype=float)
            phys2 = np.array([s2["HR"], s2["Temp"], s2["BP"]], dtype=float)

            expert = int(s1["expert_action"])  # both equal
            base_fuzzy = s1["fuzzy_state"]

            for lam in lambdas:
                b_new = lam * b1_vec + (1.0 - lam) * b2_vec
                phys_new = lam * phys1 + (1.0 - lam) * phys2

                tcpn_state = tcpn.step(b_new)
                fuzzy_state = fuzzy.get_state(tcpn_state)

                # Keep only those that remain in the same fuzzy class
                if fuzzy_state != base_fuzzy:
                    continue

                hr_new = float(np.clip(phys_new[0], 40.0, 180.0))
                temp_new = float(np.clip(phys_new[1], 35.0, 40.0))
                bp_new = float(np.clip(phys_new[2], 80.0, 200.0))

                scen_new = {
                    "scenario_id": scenario_id,
                    "b1": float(b_new[0]),
                    "b2": float(b_new[1]),
                    "b3": float(b_new[2]),
                    "b4": float(b_new[3]),
                    "b5": float(b_new[4]),
                    "HR": hr_new,
                    "Temp": temp_new,
                    "BP": bp_new,
                    "expert_action": expert,
                    "fuzzy_state": fuzzy_state,
                    "risk_level": classify_risk(fuzzy_state, expert),
                }
                augmented_scenarios.append(scen_new)
                scenario_id += 1
                added_interp += 1

    print(f"Number of interpolated scenarios added: {added_interp}")

    # 4) Save to CSV
    fieldnames = [
        "scenario_id",
        "b1", "b2", "b3", "b4", "b5",
        "HR", "Temp", "BP",
        "expert_action",
        "fuzzy_state",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scen in augmented_scenarios:
            row = {k: scen[k] for k in fieldnames}
            writer.writerow(row)

    print(f"Total number of scenarios: {len(augmented_scenarios)}")
    print(f"Augmented scenario file saved to: {output_path}")


if __name__ == "__main__":
    main()
