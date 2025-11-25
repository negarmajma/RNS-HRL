# eval_hrl_by_scenario.py
# ارزیابی HRL (Manager + Worker) به‌صورت سناریومحور روی 10 سناریوی data_loader

import os
import csv
import numpy as np
import torch
import torch.nn as nn

from envs.hierarchical_env import HierarchicalRNSEnv

# همان معماری Manager مورد استفاده در train.py
class Manager(nn.Module):
    """
    ورودی: بردار 5 تایی (tcpn.state)
    خروجی: logits روی 3 اکشن {0: NoOp, 1: DelegateToWorker, 2: StrongInhibit}
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


ACTION_MEANINGS = {
    0: "NoOp (Monitoring)",
    1: "DelegateToWorker (Adaptive Stim/Inhibit)",
    2: "StrongInhibit (Fixed suppression)",
}


def evaluate_hrl_by_scenario(
    n_scenarios: int = 10,
    episodes_per_scenario: int = 30,
    model_path: str = "models/manager.pt",
    output_csv: str = "results/eval/hrl_eval_by_scenario.csv",
):
    """
    برای هر سناریوی 0..n_scenarios-1:
      - episodes_per_scenario اپیزود اجرا می‌شود که در آن reset از آن سناریو شروع می‌شود.
      - آمار Return، طول اپیزود و توزیع اکشن‌های Manager محاسبه می‌شود.
    """

    env = HierarchicalRNSEnv()

    # بارگذاری سیاست Manager
    manager = Manager()
    state_dict = torch.load(model_path, map_location="cpu")
    manager.load_state_dict(state_dict)
    manager.eval()
    print(f"Manager policy loaded from {model_path}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    scenario_summaries = []

    for scenario_idx in range(n_scenarios):
        print(f"\n=== Evaluating scenario {scenario_idx} ===")

        returns = []
        lengths = []
        # شمارش اکشن‌ها روی تمام اپیزودها برای این سناریو
        action_counts_total = np.zeros(3, dtype=int)

        for ep in range(episodes_per_scenario):
            obs, info = env.reset(scenario_idx=scenario_idx)
            done = False
            ep_return = 0.0
            ep_length = 0

            # شمارش اکشن‌ها در این اپیزود
            action_counts_ep = np.zeros(3, dtype=int)

            while not done:
                o = torch.FloatTensor(obs).unsqueeze(0)  # (1,5)
                with torch.no_grad():
                    logits = manager(o)                  # (1,3)
                    action = torch.argmax(logits, dim=-1)
                action_int = int(action.item())

                # یک گام HRL
                next_obs, reward, done, _, next_info = env.step(action_int)

                ep_return += reward
                ep_length += 1
                action_counts_ep[action_int] += 1

                obs = next_obs

            returns.append(ep_return)
            lengths.append(ep_length)
            action_counts_total += action_counts_ep

            print(
                f"Scenario {scenario_idx} | Episode {ep:02d} "
                f"| Return={ep_return:.2f} | Steps={ep_length}"
            )

        returns = np.array(returns, dtype=float)
        lengths = np.array(lengths, dtype=int)

        total_actions = int(action_counts_total.sum())
        if total_actions > 0:
            p_actions = action_counts_total.astype(float) / total_actions
        else:
            p_actions = np.zeros_like(action_counts_total, dtype=float)

        summary = {
            "scenario": scenario_idx,
            "n_episodes": episodes_per_scenario,
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
            "count_action_2": int(action_counts_total[2]),
            "p_action_0": float(p_actions[0]),
            "p_action_1": float(p_actions[1]),
            "p_action_2": float(p_actions[2]),
        }

        scenario_summaries.append(summary)

    # ذخیره‌ی summary همه سناریوها در CSV
    if scenario_summaries:
        fieldnames = list(scenario_summaries[0].keys())
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in scenario_summaries:
                writer.writerow(rec)
        print(f"\nScenario-wise evaluation summary saved to: {output_csv}")
    else:
        print("\nNo scenario summaries collected; CSV not written.")

    print("\n=== Scenario-wise Evaluation Summary ===")
    for s in scenario_summaries:
        print(
            f"Scenario {s['scenario']}: mean_return={s['mean_return']:.2f}, "
            f"mean_length={s['mean_length']:.2f}, "
            f"p_actions="
            f"[NoOp={s['p_action_0']:.2f}, "
            f"Worker={s['p_action_1']:.2f}, "
            f"Inhibit={s['p_action_2']:.2f}]"
        )

    return scenario_summaries


if __name__ == "__main__":
    evaluate_hrl_by_scenario(
        n_scenarios=10,
        episodes_per_scenario=30,
        model_path="models/manager.pt",
        output_csv="results/eval/hrl_eval_by_scenario.csv",
    )
