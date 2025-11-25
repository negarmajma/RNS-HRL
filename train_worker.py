# train_worker.py
# آموزش عامل پایین‌دست (Worker) روی RNSBaseEnv با استفاده از REINFORCE

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt

from envs.base_env import RNSBaseEnv
from worker_policy import WorkerPolicy, action_to_effect, N_ACTIONS

def train_worker(
    n_episodes: int = 1500,
    gamma: float = 0.99,
    lr: float = 3e-4,
):
    # محیط پایه
    env = RNSBaseEnv()

    # سیاست Worker
    policy = WorkerPolicy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # ذخیره reward هر اپیزود برای رسم منحنی
    episode_rewards = []

    os.makedirs("models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    print("آموزش Worker شروع شد...")

    for ep in tqdm(range(n_episodes), desc="Training Worker"):
        obs, info = env.reset()
        done = False

        log_probs = []
        rewards = []

        while not done:
            # obs: numpy array شکل (5,)
            o = torch.FloatTensor(obs).unsqueeze(0)  # (1,5)
            logits = policy(o)                       # (1,N_ACTIONS)
            dist = Categorical(logits=logits)
            action = dist.sample()                   # (1,)

            log_probs.append(dist.log_prob(action))  # log π(a|s)
            a_idx = int(action.item())

            # نگاشت index به الگوی تحریک/مهار روی 5 کانال
            action_effect = action_to_effect(a_idx)

            # یک گام در محیط پایه
            obs, r, done, _, step_info = env.step(action_effect)
            rewards.append(r)

        # محاسبه returns با REINFORCE
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        log_probs = torch.stack(log_probs)  # شکل (T,)

        # برای پایدارتر شدن، returns را نرمالیزه می‌کنیم
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(log_probs * returns).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward = float(np.sum(rewards))
        episode_rewards.append(total_reward)

        # چاپ هر 100 اپیزود
        if (ep + 1) % 100 == 0:
            mean_last_100 = float(np.mean(episode_rewards[-100:]))
            print(
                f"Episode {ep+1} | Reward: {total_reward:.2f} "
                f"| Mean (last 100): {mean_last_100:.2f}"
            )

    # ذخیره مدل Worker
    torch.save(policy.state_dict(), "models/worker.pt")
    print("\nآموزش Worker تمام شد!")
    print("مدل Worker ذخیره شد: models/worker.pt")

    # رسم منحنی reward
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards, linewidth=2, label="Worker Episode Reward")
    if len(episode_rewards) >= 100:
        ma = np.convolve(episode_rewards, np.ones(100)/100, mode="valid")
        plt.plot(
            np.arange(len(ma)) + 99,
            ma,
            linewidth=3,
            label="Moving Average (100 episodes)"
        )
    plt.title("Worker Training Reward Curve", fontsize=16, fontweight="bold")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("results/plots/worker_training_curve.png", dpi=300, bbox_inches="tight")
    plt.savefig("results/plots/worker_training_curve.pdf", bbox_inches="tight")
    plt.close()

    print("منحنی آموزش Worker ذخیره شد: results/plots/worker_training_curve.*")


if __name__ == "__main__":
    # می‌توانی n_episodes را کم/زیاد کنی
    train_worker(n_episodes=1500, gamma=0.99, lr=3e-4)
