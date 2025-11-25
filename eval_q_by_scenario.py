# eval_q_by_scenario.py
# ارزیابی baseline Q-Learning روی 10 سناریوی data_loader به‌صورت سناریومحور

import os
import csv
import numpy as np

from data_loader import load_data
from tcpn import TCPN
from fuzzy_system import FuzzySystem


def evaluate_q_by_scenario(
    n_scenarios: int = 10,
    episodes_per_scenario: int = 30,
    q_table_path: str = "models/q_table.npy",
    output_csv: str = "results/eval/q_eval_by_scenario.csv",
):
    """
    برای هر سناریوی 0..n_scenarios-1:
      - episodes_per_scenario بار تصمیم Q-Learning را روی آن سناریو اجرا می‌کند.
      - reward بر اساس همان فرمول main.py (تطابق با expert + جریمه فیزیولوژیک) محاسبه می‌شود.
      - آمار per-scenario در قالب CSV ذخیره می‌گردد.
    """

    # بارگذاری Q-table
    if not os.path.exists(q_table_path):
        raise FileNotFoundError(f"Q-table file not found: {q_table_path}")
    Q = np.load(q_table_path)

    # بارگذاری داده‌ها
    brain_sensors, physio_sensors, expert_actions = load_data()
    tcpn = TCPN()
    fuzzy = FuzzySystem()

    state_map = {"S1": 0, "S2": 1, "S3": 2, "S4": 3}

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    scenario_summaries = []

    for scenario_idx in range(n_scenarios):
        print(f"\n=== Evaluating Q-Learning on scenario {scenario_idx} ===")

        returns = []
        lengths = []
        action_counts_total = np.zeros(2, dtype=int)  # اکشن 0 و 1
        correct_matches_total = 0

        for ep in range(episodes_per_scenario):
            sensors = brain_sensors[scenario_idx]
            hr, temp, bp = physio_sensors[scenario_idx]
            expert = expert_actions[scenario_idx]

            # حالت TCPN و فازی
            tcpn_state = tcpn.step(sensors)
            fuzzy_state = fuzzy.get_state(tcpn_state)
            state_idx = state_map[fuzzy_state]

            # سیاست greedy از روی Q-table
            q_values = Q[state_idx]
            action = int(np.argmax(q_values))  # 0 یا 1

            # پاداش مطابق main.py
            base_reward = 10.0 if action == expert else -5.0

            penalty = 0.0
            if action == 1 and (hr > 100.0 or temp > 38.0 or bp > 140.0):
                penalty -= 10.0

            reward = base_reward + penalty

            returns.append(reward)
            lengths.append(1)  # در این baseline، هر اپیزود فقط یک تصمیم دارد
            action_counts_total[action] += 1

            if action == int(expert):
                correct_matches_total += 1

            print(
                f"Scenario {scenario_idx} | Episode {ep:02d} "
                f"| fuzzy_state={fuzzy_state} | action={action} "
                f"| expert={expert} | reward={reward:.2f}"
            )

        returns = np.array(returns, dtype=float)
        lengths = np.array(lengths, dtype=int)

        total_actions = int(action_counts_total.sum())
        if total_actions > 0:
            p_actions = action_counts_total.astype(float) / total_actions
        else:
            p_actions = np.zeros_like(action_counts_total, dtype=float)

        correct_match_rate = (
            float(correct_matches_total) / total_actions if total_actions > 0 else 0.0
        )

        summary = {
            "scenario": scenario_idx,
            "n_episodes": episodes_per_scenario,
            "fuzzy_state": fuzzy.get_state(tcpn.step(brain_sensors[scenario_idx])),
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "min_return": float(returns.min()),
            "max_return": float(returns.max()),
            "mean_length": float(lengths.mean()),
            "min_length": int(lengths.min()),
            "max_length": int(lengths.max()),
            "total_actions": total_actions,
            "count_action_0": int(action_counts_total[0]),
            "count_action_1": int(action_counts_total[1]),
            "p_action_0": float(p_actions[0]),
            "p_action_1": float(p_actions[1]),
            "correct_match_rate": float(correct_match_rate),
        }

        scenario_summaries.append(summary)

    # ذخیره summary همه‌ی سناریوها در CSV
    if scenario_summaries:
        fieldnames = list(scenario_summaries[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in scenario_summaries:
                writer.writerow(rec)
        print(f"\nScenario-wise Q-Learning evaluation summary saved to: {output_csv}")
    else:
        print("\nNo scenario summaries collected; CSV not written.")

    print("\n=== Q-Learning Scenario-wise Evaluation Summary (stdout) ===")
    for s in scenario_summaries:
        print(
            f"Scenario {s['scenario']} (fuzzy={s['fuzzy_state']}): "
            f"mean_return={s['mean_return']:.2f}, "
            f"p_actions=[a0={s['p_action_0']:.2f}, a1={s['p_action_1']:.2f}], "
            f"correct_match_rate={s['correct_match_rate']:.2f}"
        )

    return scenario_summaries


if __name__ == "__main__":
    evaluate_q_by_scenario(
        n_scenarios=10,
        episodes_per_scenario=30,
        q_table_path="models/q_table.npy",
        output_csv="results/eval/q_eval_by_scenario.csv",
    )
