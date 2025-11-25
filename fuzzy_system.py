# fuzzy_system.py
# سیستم فازی برای نگاشت 5 سنسور مغزی به چهار سطح S1..S4

import numpy as np

class FuzzySystem:
    def __init__(self):
        """
        توابع عضویت مثلثی برای هر یک از 5 سنسور مغزی.
        بازه‌ها بر اساس محدوده مقادیر موجود در 10 سناریوی فعلی تنظیم شده‌اند
        و بعداً می‌توانند مطابق مقاله/تز شما دقیق‌تر شوند.

        شاخص سنسورها:
        0: Sensor1 (مثلاً ECoG power)
        1: Sensor2 (مثلاً HFO)
        2: Sensor3 (مثلاً IED)
        3: Sensor4 (مثلاً Seizure Probability)
        4: Sensor5 (مثلاً BRFB)
        """

        # پارامترهای (a,b,c) برای MF مثلثی هر سنسور: Low / Medium / High
        self.mf_params = {
            0: {  # سنسور 1
                "L": (0.0, 0.0, 40.0),
                "M": (30.0, 60.0, 90.0),
                "H": (70.0, 120.0, 120.0),
            },
            1: {  # سنسور 2
                "L": (0.0, 0.0, 100.0),
                "M": (80.0, 160.0, 240.0),
                "H": (200.0, 320.0, 320.0),
            },
            2: {  # سنسور 3
                "L": (0.0, 0.0, 5.0),
                "M": (3.0, 10.0, 17.0),
                "H": (15.0, 25.0, 25.0),
            },
            3: {  # سنسور 4 (احتمال تشنج 0..1)
                "L": (0.0, 0.0, 0.15),
                "M": (0.10, 0.30, 0.50),
                "H": (0.40, 1.0, 1.0),
            },
            4: {  # سنسور 5
                "L": (0.0, 0.0, 20.0),
                "M": (15.0, 30.0, 45.0),
                "H": (40.0, 60.0, 60.0),
            },
        }

        # نگاشت سطح فازی به مقدار عددی ریسک
        self.level_values = {"L": 0.0, "M": 0.5, "H": 1.0}

        # وزن هر سنسور در محاسبه ریسک کلی (قابل تنظیم)
        self.sensor_weights = np.array([0.8, 0.7, 0.9, 1.0, 0.8], dtype=float)

    @staticmethod
    def _tri_mf(x, a, b, c):
        """تابع عضویت مثلثی ساده."""
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if a < x < b:
            return (x - a) / (b - a)
        if b < x < c:
            return (c - x) / (c - b)
        return 0.0

    def _fuzzify_sensor(self, value, sensor_index):
        """
        فازی‌سازی یک سنسور:
        خروجی:
            best_label: یکی از L/M/H
            mu_dict: میزان عضویت در هر کلاس
        """
        params = self.mf_params.get(sensor_index)
        if params is None:
            # اگر سنسور اضافه‌ای بود، پیش‌فرض Medium
            return "M", {"L": 0.0, "M": 1.0, "H": 0.0}

        mu_dict = {}
        for label, (a, b, c) in params.items():
            mu = self._tri_mf(value, a, b, c)
            mu_dict[label] = mu

        best_label = max(mu_dict, key=mu_dict.get)
        return best_label, mu_dict

    def _fuzzify_vector(self, x):
        """
        فازی‌سازی یک بردار 5تایی از سنسورها.
        خروجی:
            labels: لیست L/M/H برای هر سنسور
            mus: لیست دیکشنری عضویت‌ها
        """
        x = np.asarray(x, dtype=float).ravel()
        labels = []
        mus = []

        for i, val in enumerate(x):
            label, mu_dict = self._fuzzify_sensor(val, i)
            labels.append(label)
            mus.append(mu_dict)

        return labels, mus

    def _aggregate_risk(self, labels):
        """
        تبدیل L/M/H هر سنسور به یک ریسک کلی نرمال‌شده بین 0 و 1.
        """
        numeric_levels = np.array([self.level_values[lbl] for lbl in labels], dtype=float)
        weighted = numeric_levels * self.sensor_weights
        risk = weighted.sum() / self.sensor_weights.sum()
        return float(risk)

    def _risk_to_state(self, risk):
        """
        نگاشت ریسک [0,1] به چهار سطح S1..S4.
        آستانه‌ها بعداً می‌توانند مطابق جدول مقاله تنظیم شوند.
        """
        if risk < 0.25:
            return "S1"  # حالت نرمال
        elif risk < 0.5:
            return "S2"  # Mid abnormal
        elif risk < 0.75:
            return "S3"  # High abnormal
        else:
            return "S4"  # Seizure / critical

    def get_state(self, tcpn_output):
        """
        ورودی:
          - اگر string مثل 'S1'..'S4' باشد، مستقیم برمی‌گردانیم (سازگاری عقب‌رو).
          - اگر بردار numpy باشد (خروجی عددی TCPN)، فازی‌سازی می‌کنیم و سطح S1..S4 را برمی‌گردانیم.
        """
        # حالت قدیمی: اگر TCPN قبلاً سطح را داده باشد
        if isinstance(tcpn_output, str):
            return tcpn_output

        # حالت جدید: tcpn_output یک بردار 5تایی است
        labels, _ = self._fuzzify_vector(tcpn_output)
        risk = self._aggregate_risk(labels)
        state = self._risk_to_state(risk)
        return state

    # تابع کمکی برای دیباگ
    def debug_fuzzy(self, x):
        """
        جزئیات فازی‌سازی یک بردار x را برمی‌گرداند.
        """
        labels, mus = self._fuzzify_vector(x)
        risk = self._aggregate_risk(labels)
        state = self._risk_to_state(risk)
        return {
            "labels": labels,
            "mus": mus,
            "risk": risk,
            "state": state,
        }
