# envs/rns_tcpn_env.py
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, TYPE_CHECKING

from models.safety import project_to_safe_action, constraint_cost

if TYPE_CHECKING:
    from models.tcpn import TCPN

class RNSTCPNEnv(gym.Env):
    """
    محیط کارگر سطح پایین (Low-level environment)
    option_id=1 => Stimulate : فضای اکشن پیوسته ۴بُعدی [V, freq, PW, Dur]
    option_id=0 => Monitor   : فضای اکشن پیوسته ۳بُعدی [d_th1, d_th2, sr]
    """

    metadata = {"render_modes": []}

    def __init__(self, tcpn: "TCPN", option_id: int):
        super().__init__()
        self.tcpn = tcpn
        self.option_id = int(option_id)

        # فضای مشاهده (بردار ۲۰تایی از ویژگی‌های سطح پایین)
        self.observation_space: spaces.Space = spaces.Box(
            low=np.full((20,), -np.inf, dtype=np.float32),
            high=np.full((20,),  np.inf, dtype=np.float32)
        )

        # فضای اکشن بر اساس نوع گزینه
        if self.option_id == 1:
            # Stimulate
            self.action_space: spaces.Space = spaces.Box(
                low=np.array([1.0,  50.0,  50.0, 0.5], dtype=np.float32),
                high=np.array([5.0, 200.0, 100.0, 5.0], dtype=np.float32)
            )
        else:
            # Monitor
            self.action_space: spaces.Space = spaces.Box(
                low=np.array([-0.2, -0.2,  0.5], dtype=np.float32),
                high=np.array([ 0.2,  0.2,  4.0], dtype=np.float32)
            )

    # ------------------ Gymnasium API ------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        obs = self._obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # پروژکشن اکشن به ناحیه ایمن
        safe_action = project_to_safe_action(self.tcpn.current_features(), action, self.option_id)

        # اعمال اکشن به TCPN
        r_env, c_env, terminated = self.tcpn.apply_worker_action(safe_action, self.option_id)

        obs_next = self._obs()
        reward = float(r_env - self.tcpn.lambda_penalty * c_env)
        info = {"cost": float(c_env)}

        # در این نسخه truncated=False است (فقط termination داخلی TCPN)
        return obs_next, reward, bool(terminated), False, info

    # ------------------ کمکی ------------------
    def _obs(self) -> np.ndarray:
        """بازگرداندن بردار ویژگی‌های سطح پایین از TCPN."""
        return self.tcpn.low_level_features_vec()
