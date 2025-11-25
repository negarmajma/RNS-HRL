# tcpn.py
import numpy as np

class TCPN:
    def __init__(self):
        self.state = np.zeros(5)

    def step(self, sensors, action_effect=np.zeros(5)):
        self.state = np.array(sensors)
        self.state += action_effect  # اثر تحریک یا مهار
        self.state = np.clip(self.state, 0, None)

        if self.state[3] >= 0.05 and self.state[2] >= 3.0:
            return "S2"
        if self.state[2] >= 10.0 and self.state[1] >= 200.0:
            return "S3"
        if self.state[4] >= 30.0 and self.state[0] >= 50.0:
            return "S4"
        return "S1"