import numpy as np


class PID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        target: float,
        integral_limit: float = 800.0,
        output_limit: float = 100.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral = 0.0
        self.last_error = 0.0

    def clear(self, error: float = 0.0) -> None:
        self.integral = 0.0
        self.last_error = error

    def calc(self, current: float, dt: float) -> float:
        error = self.target - current
        self.integral += error * dt
        self.integral = np.clip(
            self.integral, -self.integral_limit, self.integral_limit
        ).item()
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return np.clip(
            self.kp * error + self.ki * self.integral + self.kd * derivative,
            -self.output_limit,
            self.output_limit,
        ).item()


def move_towards(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    return max(current - max_delta, target)
