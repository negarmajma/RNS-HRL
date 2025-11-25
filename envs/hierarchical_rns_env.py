# envs/hierarchical_env.py
# محیط سلسله مراتبی واقعی – دقیقاً مطابق مقاله شما

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from envs.rns_option_env import RNSOptionEnv

class HierarchicalRNSEnv(gym.Env):
    def __init__(self):
        self.low_env = RNSOptionEnv()  # low-level همیشه پیوسته
        self.option_duration = 10
        self.current_option = 0
        self.timer = 0

        self.observation_space = self.low_env.observation_space
        self.action_space = spaces.Discrete(3)  # 0=NoOp, 1=Stimulate, 2=Inhibit

        # قالب‌های ثابت برای low-level (می‌توانید بعداً SAC بگذارید)
        self.templates = {
            0: np.zeros(5, dtype=np.float32),
            1: np.array([2.5, 0.0, 2.5, 0.0, 2.5], dtype=np.float32),
            2: np.array([-2.0, -2.0, 0.0, 0.0, -2.0], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        obs, info = self.low_env.reset(seed=seed)
        self.timer = 0
        self.current_option = 0
        return obs, info

    def step(self, action):  # action از مدیر: 0,1,2
        self.current_option = int(action)
        self.timer = self.option_duration

        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self.option_duration):
            if done:
                break

            low_action = self.templates[self.current_option]
            obs, r, terminated, truncated, step_info = self.low_env.step(low_action)
            total_reward += r
            done = terminated or truncated
            info = step_info

        return obs, total_reward, done, False, info