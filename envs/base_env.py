# envs/base_env.py
# محیط پایه برای Worker: یک گام از TCPN + Fuzzy + سنسورهای فیزیولوژیک

import numpy as np
from data_loader import load_data, load_augmented_data
from tcpn import TCPN
from fuzzy_system import FuzzySystem

class RNSBaseEnv:
    """
    این محیط نمایانگر «یک لایه پایین» دستگاه است:
    - ورودی: action_effect (بردار 5 تایی اثر تحریک/مهار روی سنسورهای مغزی)
    - دینامیک: TCPN + سیستم فازی + سنسورهای فیزیولوژیک
    - خروجی: حالت جدید، پاداش، done، info
    فعلاً از همین 10 سناریوی ثابت data_loader استفاده می‌شود.
    """

    def __init__(self):
        self.tcpn = TCPN()
        self.fuzzy = FuzzySystem()
        self.brain, self.physio, self.expert = load_augmented_data()


        self.current_scenario = 0
        self.max_steps = 50   # حداکثر گام‌های Worker در هر اپیزود (قابل تنظیم)
        self.steps = 0

    def reset(self, scenario_idx: int | None = None):
        """
        شروع یک اپیزود جدید Worker:
        - انتخاب تصادفی یکی از 10 سناریو
        - ریست شمارنده گام‌ها
        - به‌روزرسانی TCPN و فازی برای حالت اولیه
        """
        if scenario_idx is None:
            self.current_scenario = np.random.randint(0, len(self.brain))
        else:
            # اطمینان از محدوده‌ی معتبر
            scenario_idx = int(scenario_idx)
            if scenario_idx < 0 or scenario_idx >= len(self.brain):
                raise ValueError(f"Invalid scenario_idx={scenario_idx}")
            self.current_scenario = scenario_idx

        self.steps = 0

        sensors = self.brain[self.current_scenario]
        # حالت اولیه، بدون تحریک
        self.tcpn.step(sensors, np.zeros(5))
        fuzzy_state = self.fuzzy.get_state(self.tcpn.state)

        # در فاز 1 برای سادگی obs صرفاً همان tcpn.state است
        obs = self.tcpn.state.copy()
        info = {"fuzzy": fuzzy_state, "scenario": self.current_scenario}
        return obs, info

    def step(self, action_effect):
        """
        یک گام پیشروی:
        - action_effect: بردار 5تایی اثر تحریک/مهار روی TCPN
        - خروجی: obs, reward, done, truncated, info
        """
        self.steps += 1

        sensors = self.brain[self.current_scenario]
        hr, temp, bp = self.physio[self.current_scenario]

        # اعمال تحریک روی TCPN
        self.tcpn.step(sensors, action_effect)
        fuzzy_state = self.fuzzy.get_state(self.tcpn.state)

        # نگاشت fuzzy_state به y (برای reward)
        y = {"S1": 0.0, "S2": 0.3, "S3": 0.6, "S4": 1.0}.get(fuzzy_state, 0.0)

        # پاداش پایه
        reward = 10.0 * (1.0 - y)

        # جریمه وضعیت‌های فیزیولوژیک نامطلوب
        if hr > 100 or temp > 38.0 or bp > 140:
            reward -= 5.0

        # جریمه انرژی تحریک (هر چه تحریک بزرگ‌تر، penalty بیشتر)
        energy = float(np.linalg.norm(action_effect, ord=1))  # L1 norm
        reward -= 0.1 * energy

        # شرط پایان اپیزود برای Worker:
        done = False
        if fuzzy_state == "S4":
            done = True
        if self.steps >= self.max_steps:
            done = True

        obs = self.tcpn.state.copy()
        info = {
            "fuzzy": fuzzy_state,
            "scenario": self.current_scenario,
            "energy": energy,
        }

        return obs, reward, done, False, info
