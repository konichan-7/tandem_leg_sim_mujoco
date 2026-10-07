import ctypes

KEYBOARD_SPEED = 1.0
KEYBOARD_YAW_RATE = 2.0
LINEAR_ACCELERATION = 1.5
YAW_ACCELERATION = 6.0


def ramp_command(
    current: float, target: float, acceleration: float, dt: float
) -> float:
    step = acceleration * dt
    return current + max(-step, min(step, target - current))


class KeyboardCommand:
    def __init__(self) -> None:
        self._core_graphics = ctypes.CDLL(
            "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
        )
        self._core_graphics.CGEventSourceKeyState.argtypes = [
            ctypes.c_int32,
            ctypes.c_uint16,
        ]
        self._core_graphics.CGEventSourceKeyState.restype = ctypes.c_bool

    def direction(self) -> tuple[float, float]:
        pressed = self._core_graphics.CGEventSourceKeyState
        return (
            float(pressed(0, 18) - pressed(0, 19)),
            float(pressed(0, 21) - pressed(0, 20)),
        )
