# data_loader.py
import numpy as np
import os

def load_data():
    # 5 سنسور مغزی – دقیقاً از Appendix مقاله شما
    brain_sensors = np.array([
        [95.0, 285.0, 18.0, 0.18, 44.0],
        [40.0, 80.0, 5.0, 0.0106, 18.0],
        [20.0, 100.0, 5.0, 0.13, 10.0],
        [60.0, 150.0, 8.0, 0.1, 25.0],
        [90.0, 280.0, 18.0, 0.12, 45.0],
        [60.0, 150.0, 8.0, 0.1, 25.0],
        [70.0, 250.0, 15.0, 0.14, 40.0],
        [30.0, 60.0, 3.0, 0.04, 15.0],
        [20.0, 35.0, 1.0, 0.01, 10.0],
        [100.0, 300.0, 20.0, 1.0, 50.0],
    ])

    # 3 سنسور فیزیولوژیکی – دقیقاً از مقاله
    physio_sensors = np.array([
        [130.0, 38.5, 150.0],  # HR, Temp, BP
        [110.0, 38.2, 135.0],
        [90.0, 37.8, 120.0],
        [140.0, 38.5, 150.0],
        [101.0, 37.6, 121.0],
        [90.0, 37.5, 120.0],
        [140.0, 38.7, 160.0],
        [100.0, 38.0, 125.0],
        [90.0, 37.6, 110.0],
        [180.0, 38.5, 150.0],
    ])

    # خروجی مطلوب کارشناس
    expert_actions = [1, 0, 0, 0, 1, 0, 1, 1, 0, 1]  # 1=Stimulation, 0=Monitoring

    return brain_sensors, physio_sensors, expert_actions

def load_augmented_data(path: str = "data/scenarios_augmented.csv"):
    """
    داده‌های سناریوهای افزایش‌یافته را از CSV می‌خواند و
    مشابه load_data سه خروجی برمی‌گرداند:
      brain_sensors: آرایه (N,5)
      physio_sensors: آرایه (N,3)
      expert_actions: لیست طول N
    """
    # ستون‌ها:
    # 0: scenario_id
    # 1..5: b1..b5
    # 6..8: HR, Temp, BP
    # 9: expert_action
    # 10: fuzzy_state (رشته) → در اینجا نمی‌خوانیم

    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(f"Augmented scenario file not found: {path}")

    # فقط ستون‌های عددی 1 تا 9 را می‌خوانیم
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=range(1, 10))

    brain_sensors = data[:, 0:5]        # b1..b5
    physio_sensors = data[:, 5:8]       # HR, Temp, BP
    expert_actions = data[:, 8].astype(int).tolist()

    return brain_sensors, physio_sensors, expert_actions
