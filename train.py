# train.py
# Final training of the hierarchical manager + plotting the training curve for the paper

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from envs.hierarchical_env import HierarchicalRNSEnv


class Manager(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 3)  # 0=NoOp, 1=Stimulate, 2=Inhibit
        )

    def forward(self, x):
        return self.net(x)


# Environment and policy
env = HierarchicalRNSEnv()
policy = Manager()
optimizer = optim.Adam(policy.parameters(), lr=3e-4)

# --- Detect and print dataset type (10 scenarios vs augmented ~1770 scenarios) ---
n_scenarios = None
dataset_label = "unknown"

try:
    # Assume RNSBaseEnv has an attribute 'brain' with shape (N, 5)
    base_env = getattr(env, "base_env", None)
    if base_env is not None and hasattr(base_env, "brain"):
        n_scenarios = len(base_env.brain)
        if n_scenarios <= 20:
            dataset_label = "original_10_scenarios"
        else:
            dataset_label = "augmented_dataset"
except Exception:
    pass

print("==========================================")
if n_scenarios is not None:
    print(f"Number of training scenarios: {n_scenarios}  |  dataset_type = {dataset_label}")
    if dataset_label == "original_10_scenarios":
        print(">> Training with the original 10 expert-designed scenarios.")
    else:
        print(">> Training with the augmented dataset (noise + interpolation + physio profiles).")
else:
    print("Warning: could not determine number of scenarios from env.base_env.brain")
print("==========================================\n")
# ------------------------------------------------------------------

# Store episode rewards for plotting
episode_rewards = []

os.makedirs("models", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

print("Training started...")

for ep in tqdm(range(1500), desc="Training"):
    obs, _ = env.reset()
    done = False
    log_probs = []
    rewards_this_episode = []

    while not done:
        o = torch.FloatTensor(obs).unsqueeze(0)
        logits = policy(o)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))

        obs, r, done, _, info = env.step(action.item())
        rewards_this_episode.append(r)

    # Compute returns and update policy
    returns = []
    R = 0
    for r in rewards_this_episode[::-1]:
        R = r + 0.99 * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    log_probs = torch.stack(log_probs)
    loss = -(log_probs * returns).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_reward = sum(rewards_this_episode)
    episode_rewards.append(total_reward)

    # Print every 100 episodes
    if (ep + 1) % 100 == 0:
        mean_last_100 = np.mean(episode_rewards[-100:])
        print(f"Episode {ep+1} | Reward: {total_reward:.2f} | Mean (last 100): {mean_last_100:.2f}")

# Save final model
torch.save(policy.state_dict(), "models/manager.pt")

# Plot and save training curve
plt.figure(figsize=(12, 6))
plt.plot(episode_rewards, linewidth=2, label='Episode Reward')
if len(episode_rewards) >= 100:
    ma = np.convolve(episode_rewards, np.ones(100) / 100, mode='valid')
    plt.plot(ma, linewidth=3, label='Moving Average (100 episodes)')
plt.title('Training Reward Curve', fontsize=16, fontweight='bold')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/training_reward_curve.png", dpi=300, bbox_inches='tight')
plt.savefig("results/plots/training_reward_curve.pdf", bbox_inches='tight')
plt.close()

print("\nTraining finished.")
print("Manager policy saved to: models/manager.pt")
print("Training curve saved to: results/plots/training_reward_curve.png")
