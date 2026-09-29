import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from utils.math_tools import LqrDesign


class GimbalController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        design: LqrDesign,
    ) -> None:
        self.model = model
        self.data = data
        self.joints = [model.joint(f"{name}_joint") for name in ("yaw", "pitch")]
        self.qpos = np.array([joint.qposadr[0] for joint in self.joints])
        self.dofs = np.array([joint.dofadr[0] for joint in self.joints])
        self.actuators = np.array(
            [model.actuator(f"{name}_motor").id for name in ("yaw", "pitch")]
        )
        self.pitch_body = model.body("pitch_link").id
        self.home = design.qpos[self.qpos].copy()
        self.gain = design.gain[self.actuators]
        self.limits = design.limits[self.actuators]
        self.reference = self.home.copy()
        self.direction = np.zeros(3)
        self.holding = False

    def reset(self) -> None:
        self.reference[:] = self.home
        self.direction[:] = self.data.xmat[self.pitch_body].reshape(3, 3)[:, 0]
        self.holding = False

    def hold_world(self, enabled: bool) -> None:
        if enabled and not self.holding:
            mujoco.mj_forward(self.model, self.data)
            self.direction[:] = self.data.xmat[self.pitch_body].reshape(3, 3)[:, 0]
            self.reference[:] = self.data.qpos[self.qpos]
        self.holding = enabled

    def update_reference(
        self, base_quaternion: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.holding:
            direction = (
                Rotation.from_quat(base_quaternion, scalar_first=True)
                .inv()
                .apply(self.direction)
            )
            target = np.array(
                [
                    np.arctan2(direction[1], direction[0]) / self.joints[0].axis[2],
                    -np.arctan2(direction[2], np.hypot(direction[0], direction[1]))
                    / self.joints[1].axis[1],
                ]
            )
        else:
            target = self.home
        delta = target - self.reference
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        self.reference += delta
        return self.reference, delta / dt

    def control(self, state_error: np.ndarray, feedforward: np.ndarray) -> None:
        self.data.ctrl[self.actuators] = np.clip(
            feedforward[self.actuators] - self.gain @ state_error,
            -self.limits,
            self.limits,
        )
