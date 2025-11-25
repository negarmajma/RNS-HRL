# envs/hierarchical_env.py
# محیط سلسله‌مراتبی – Manager (High-level) که روی BaseEnv + Worker سوار می‌شود

import numpy as np
import torch
from torch.distributions import Categorical

from envs.base_env import RNSBaseEnv
from worker_policy import WorkerPolicy, action_to_effect

class HierarchicalRNSEnv:
    """
    لایه‌ی High-level Manager:
    - در هر گام، یک اکشن گسسته از {0: NoOp, 1: DelegateToWorker, 2: StrongInhibit} می‌گیرد.
    - اگر 1 باشد، تصمیم جزئی تحریک/مهار را به Worker آموزش‌دیده واگذار می‌کند.
    - اگر 2 باشد، یک الگوی مهار ثابت اعمال می‌کند.
    - در هر گام Manager، option_duration گام در BaseEnv اجرا می‌شود.
    """

    def __init__(self):
        self.base_env = RNSBaseEnv()
        self.option_duration = 10

        # Worker آموزش‌دیده (پایین‌دست)
        self.worker = WorkerPolicy()
        try:
            state_dict = torch.load("models/worker.pt", map_location="cpu")
            self.worker.load_state_dict(state_dict)
            self.worker.eval()
            print("Worker policy loaded from models/worker.pt")
        except FileNotFoundError:
            print("WARNING: models/worker.pt not found. Worker will be untrained.")
            self.worker.eval()

        self.current_option = 0
        self.current_scenario = 0

        # شمارنده برای حداکثر تعداد تصمیم‌های Manager در هر اپیزود
        self.max_manager_steps = 20
        self.manager_steps = 0

    def reset(self, scenario_idx: int | None = None):
        """
        ریست اپیزود Manager:
        - اگر scenario_idx=None باشد، BaseEnv سناریو را تصادفی انتخاب می‌کند
          (رفتار قبلی و سازگار با train.py).
        - اگر عدد 0..9 بدهیم، اپیزود دقیقاً از همان سناریو شروع می‌شود
          (برای eval سناریومحور).
        """
        obs, info = self.base_env.reset(scenario_idx=scenario_idx)
        self.manager_steps = 0
        self.current_option = 0
        self.current_scenario = info.get("scenario", 0)
        return obs, info

    def _fixed_inhibit_pattern(self) -> np.ndarray:
        """
        الگوی مهار ثابت برای اکشن StrongInhibit (اکشن 2).
        """
        return np.array([-2.0, -2.0, -2.0, 0.0, -2.0], dtype=float)

    def _worker_action_effect(self) -> np.ndarray:
        """
        گرفتن action_effect از Worker آموزش‌دیده، در وضعیت فعلی BaseEnv.
        - obs Worker در فاز 3 = tcpn.state (همان obs BaseEnv).
        """
        # obs فعلی را از TCPN داخل BaseEnv می‌گیریم
        obs = self.base_env.tcpn.state.copy().astype(np.float32)
        o = torch.from_numpy(obs).unsqueeze(0)  # (1, 5)

        with torch.no_grad():
            logits = self.worker(o)             # (1, N_ACTIONS)
            dist = Categorical(logits=logits)
            action = dist.sample()              # (1,)
            a_idx = int(action.item())

        # نگاشت index به بردار 5تایی تحریک/مهار
        return action_to_effect(a_idx)

    def step(self, manager_action: int):
        """
        یک تصمیم High-level:
        - manager_action: {0, 1, 2}
            0: NoOp (بدون تحریک، action_effect = 0)
            1: DelegateToWorker (Worker تصمیم می‌گیرد چه الگویی اعمال شود)
            2: StrongInhibit (الگوی مهار ثابت)
        - برای option_duration گام، BaseEnv را با action_effect مربوطه جلو می‌بریم.
        - پاداش گزارش‌شده، مجموع پاداش‌های BaseEnv در طول این option است.
        """
        self.current_option = manager_action
        self.manager_steps += 1

        total_reward = 0.0
        done = False
        last_obs = None
        last_info = {}

        for _ in range(self.option_duration):
            if done:
                break

            # تعیین action_effect بر اساس تصمیم Manager
            if manager_action == 0:
                # NoOp → بدون تحریک
                action_effect = np.zeros(5, dtype=float)
            elif manager_action == 1:
                # واگذاری به Worker آموزش‌دیده
                action_effect = self._worker_action_effect()
            else:
                # StrongInhibit → الگوی مهار ثابت
                action_effect = self._fixed_inhibit_pattern()

            # یک گام در BaseEnv
            obs, r, done_base, _, info = self.base_env.step(action_effect)
            total_reward += r

            last_obs = obs
            last_info = info

            # اگر BaseEnv به حالت terminal رسیده، اپیزود Manager هم تمام می‌شود
            if done_base:
                done = True
                break

        # اگر هنوز تمام نشده و تعداد تصمیم‌ها به حد مجاز رسید، اپیزود را ببند
        if not done and self.manager_steps >= self.max_manager_steps:
            done = True

        if last_obs is None:
            # حالت غیرعادی؛ برای ایمنی یک reset انجام می‌دهیم
            last_obs, last_info = self.base_env.reset()

        info_out = {
            "fuzzy": last_info.get("fuzzy", None),
            "scenario": last_info.get("scenario", self.current_scenario),
            "manager_steps": self.manager_steps,
            "option": manager_action,
        }

        return last_obs, total_reward, done, False, info_out
