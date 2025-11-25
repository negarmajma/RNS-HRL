# eval.py
# ارزیابی نهایی + رسم reward گام به گام (دقیقاً همان چیزی که می‌خوای)

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from envs.hierarchical_env import HierarchicalRNSEnv

class Manager(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = Manager().to(device)

if not os.path.exists("models/manager.pt"):
    print("مدل پیدا نشد! اول train.py رو اجرا کن.")
    exit()

policy.load_state_dict(torch.load("models/manager.pt", map_location=device))
policy.eval()

env = HierarchicalRNSEnv()
n_episodes = 200

# ذخیره تمام rewardهای هر گام در تمام اپیزودها
all_step_rewards = []  # لیست از لیست: هر اپیزود یک لیست از reward گام‌ها
all_stimulations = []  # ذخیره زمان تحریک در هر گام

print(f"Evaluating on {n_episodes} episodes and collecting step-by-step rewards...")

with torch.no_grad():
    for ep in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        done = False
        step_rewards = []
        step_stims = []

        while not done:
            o = torch.FloatTensor(obs).unsqueeze(0).to(device)
            logits = policy(o)
            action = torch.argmax(logits).item()

            obs, r, done, _, info = env.step(action)
            step_rewards.append(r)
            step_stims.append(1 if action == 1 else 0)  # تحریک یا نه

        all_step_rewards.append(step_rewards)
        all_stimulations.append(step_stims)

# --- رسم شکل‌ها ---
os.makedirs("results/plots", exist_ok=True)

# 1. reward گام به گام یک اپیزود نمونه
plt.figure(figsize=(12, 6))
sample_rewards = all_step_rewards[0]
plt.plot(sample_rewards, marker='o', markersize=5, linewidth=2, color='#1f77b4', label='Reward')
plt.plot(np.where(np.array(all_stimulations[0]) == 1)[0], 
         [sample_rewards[i] for i in np.where(np.array(all_stimulations[0]) == 1)[0]], 
         'r^', markersize=12, label='Stimulation')
plt.title('Step-by-Step Reward and Stimulation Timing in One Episode', fontsize=14, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('Instantaneous Reward')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/step_by_step_reward_with_stim.png", dpi=300)
plt.savefig("results/plots/step_by_step_reward_with_stim.pdf")
plt.close()

# 2. میانگین reward در هر گام (روی تمام ۲۰۰ اپیزود)
max_length = max(len(ep) for ep in all_step_rewards)
mean_rewards = []
std_rewards = []

for t in range(max_length):
    rewards_at_t = [ep[t] for ep in all_step_rewards if len(ep) > t]
    mean_rewards.append(np.mean(rewards_at_t))
    std_rewards.append(np.std(rewards_at_t))

plt.figure(figsize=(10, 6))
plt.plot(mean_rewards, color='#d7191c', linewidth=3, label='Mean Reward')
plt.fill_between(range(max_length), 
                 np.array(mean_rewards) - np.array(std_rewards),
                 np.array(mean_rewards) + np.array(std_rewards),
                 color='#d7191c', alpha=0.2, label='±1 Std')
plt.title('Average Instantaneous Reward per Time Step (200 Episodes)', fontsize=14, fontweight='bold')
plt.xlabel('Time Step')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/average_step_reward.png", dpi=300)
plt.savefig("results/plots/average_step_reward.pdf")
plt.close()

# 3. هیستوگرام rewardهای گام
all_rewards_flat = [r for ep in all_step_rewards for r in ep]
plt.figure(figsize=(8, 5))
plt.hist(all_rewards_flat, bins=30, color='#2ca02c', edgecolor='black')
plt.title('Distribution of Instantaneous Rewards Across All Steps', fontsize=14, fontweight='bold')
plt.xlabel('Reward Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/reward_histogram.png", dpi=300)
plt.savefig("results/plots/reward_histogram.pdf")
plt.close()

print("\nشکل‌های گام به گام با موفقیت ذخیره شد:")
print("  • step_by_step_reward_with_stim.png  (reward + زمان تحریک)")
print("  • average_step_reward.png             (میانگین reward هر گام)")
print("  • reward_histogram.png                (توزیع rewardها)")