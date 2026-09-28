import math

import numpy as np


def wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return wrap(a - b)


def move_towards(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


def quat_to_euler(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = quat
    pitch = wrap(math.asin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = wrap(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return pitch, -yaw
