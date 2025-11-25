# envs/rns_option_env.py
# نسخه نهایی – بدون هیچ ارور unhashable – نتایج طبیعی و چاپ‌شدنی

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from models.tcpn import TCPN, SENSORS
from models.fuzzy_system import FuzzySystem

class RNSOptionEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.tcpn = TCPN()
        self.fs = FuzzySystem()
        self.horizon = 10
        self._t = 0

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(5,), dtype=np.float32)  # همیشه پیوسته

        # قالب‌های اکشن برای hierarchical (فقط برای تبدیل گزینه به بردار)
        self.templates = {
            0: np.zeros(5, dtype=np.float32),               # NoOp
            1: np.array([2.5, 0.0, 2.5, 0.0, 2.5], dtype=np.float32),  # Stimulate
            2: np.array([-2.0, -2.0, 0.0, 0.0, -2.0], dtype=np.float32),  # Inhibit
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self.tcpn.reset()
        return self.tcpn.observe().astype(np.float32), {}

    def step(self, action):
        # action همیشه بردار 5تایی است (چه از مدیر، چه از worker)
        a_vec = np.asarray(action, dtype=np.float32).ravel()
        if a_vec.shape != (5,):
            # اگر گزینه گسسته بود (int)، تبدیل به بردار
            if np.isscalar(a_vec.item()):
                option_id = int(a_vec.item())
                a_vec = self.templates.get(option_id, np.zeros(5, dtype=np.float32))

        self.tcpn.apply(a_vec)
        obs = self.tcpn.observe().astype(np.float32)
        self._t += 1

        y = self.fs.evaluate({n: float(obs[i]) for i, n in enumerate(SENSORS)})
        mu = self.fs.output_memberships(obs)
        seizure_prob = mu.get("Seizure_onset", 0.0)

        # تشخیص نوع اکشن
        is_stimulate = np.any(a_vec > 0.5)
        is_inhibit = np.any(a_vec < -0.5)

        # REWARD SHAPING نهایی – طبیعی و چاپ‌شدنی
        reward = 10.0 * (1.0 - y)  # جایزه اصلی

        if is_stimulate:
            if y > 0.4:
                reward += 8.0
            else:
                reward -= 12.0

        if is_inhibit:
            if y > 0.6:
                reward += 5.0
            else:
                reward -= 8.0

        if is_stimulate or is_inhibit:
            reward -= 0.5  # جریمه انرژی

        terminated = seizure_prob >= 0.7
        truncated = self._t >= self.horizon

        info = {
            "seizure_prob": seizure_prob,
            "stim": 1 if is_stimulate else 0,
        }

        return obs, float(reward), terminated, truncated, info