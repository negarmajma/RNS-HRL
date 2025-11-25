# main.py
# Q-Learning baseline روی 10 سناریوی brain/physio با TCPN + Fuzzy
# این فایل مستقل است و نیازی به rl_agent ندارد.

import os
import numpy as np

from data_loader import load_data
from tcpn import TCPN
from fuzzy_system import FuzzySystem


class QLearningAgent:
    """
    عامل ساده Q-Learning با جدول Q(S,A).
    حالت‌ها: 4 وضعیت فازی S1..S4 → ایندکس 0..3
    اکشن‌ها: 0 = Monitoring (NoStim), 1 = Stimulation
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions), dtype=float)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state_idx: int) -> int:
        """
        انتخاب اکشن به‌صورت ε-greedy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s: int, a: int, r: float, s_next: int):
        """
        به‌روزرسانی Q-table با قاعده Q-Learning.
        """
        best_next = np.max(self.Q[s_next])
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self):
        """
        کاهش تدریجی ε برای کاهش اکتشاف در طول زمان.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def main():
    # بارگذاری داده‌ها
    brain_sensors, physio_sensors, expert_actions = load_data()
    tcpn = TCPN()
    fuzzy = FuzzySystem()

    # نگاشت حالت‌های فازی به ایندکس عددی Q-table
    state_map = {"S1": 0, "S2": 1, "S3": 2, "S4": 3}

    n_states = 4          # S1..S4
    n_actions = 2         # 0 = Monitoring, 1 = Stimulation

    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )

    n_episodes = 1500

    print("آموزش Q-Learning baseline شروع شد...")

    for ep in range(n_episodes):
        # در هر اپیزود، روی 10 سناریو حلقه می‌زنیم
        # می‌توانی برای تصادفی‌سازی، ترتیب سناریوها را shuffle کنی
        indices = np.arange(len(brain_sensors))
        np.random.shuffle(indices)

        ep_total_reward = 0.0

        for i in indices:
            sensors = brain_sensors[i]
            hr, temp, bp = physio_sensors[i]
            expert = expert_actions[i]

            # حالت فعلی TCPN → خروجی S1..S4
            tcpn_state = tcpn.step(sensors)
            fuzzy_state = fuzzy.get_state(tcpn_state)
            state_idx = state_map[fuzzy_state]

            # انتخاب اکشن با ε-greedy
            action = agent.choose_action(state_idx)  # 0 یا 1

            # پاداش پایه: تشویق به تطابق با کارشناس
            # 1=Stimulation, 0=Monitoring
            base_reward = 10.0 if action == expert else -5.0

            # جریمه ایمنی فیزیولوژیک:
            # اگر وضعیت فیزیولوژیک بد باشد و ما تحریک کنیم، جریمه شدید
            penalty = 0.0
            if action == 1 and (hr > 100.0 or temp > 38.0 or bp > 140.0):
                penalty -= 10.0

            reward = base_reward + penalty
            ep_total_reward += reward

            # در این نسخه‌ی baseline، حالت بعدی را همان حالت فعلی در نظر می‌گیریم
            # (مسئله 1-step؛ سناریو ثابت است)
            next_state_idx = state_idx

            # به‌روزرسانی Q-table
            agent.update(state_idx, action, reward, next_state_idx)

        # کاهش ε بعد از هر اپیزود
        agent.decay_epsilon()

        # چاپ هر 100 اپیزود
        if (ep + 1) % 100 == 0:
            print(
                f"Episode {ep+1} | Total Reward over 10 scenarios: {ep_total_reward:.2f} "
                f"| epsilon={agent.epsilon:.3f}"
            )

    print("آموزش Q-Learning baseline تمام شد.")

    # ذخیره‌ی Q-table برای استفاده در ارزیابی بعدی
    os.makedirs("models", exist_ok=True)
    q_table_path = "models/q_table.npy"
    np.save(q_table_path, agent.Q)
    print(f"Q-table ذخیره شد: {q_table_path}")
    print("مقادیر نهایی Q(S,A):")
    print(agent.Q)


if __name__ == "__main__":
    main()
