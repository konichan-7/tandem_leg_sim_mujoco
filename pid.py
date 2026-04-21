import numpy as np


class PIDControl:
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

    def reset(self, error: float = 0.0) -> None:
        self.integral = 0.0
        self.last_error = error

    def position_pid(self, current: float, dt: float) -> float:
        error = self.target - current
        self.integral += error * dt
        self.integral = float(
            np.clip(self.integral, -self.integral_limit, self.integral_limit)
        )
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return float(
            np.clip(
                self.kp * error + self.ki * self.integral + self.kd * derivative,
                -self.output_limit,
                self.output_limit,
            )
        )


PID_control = PIDControl
