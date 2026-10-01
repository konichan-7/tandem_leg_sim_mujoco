import math

import mujoco
import numpy as np


class VMC:
    def __init__(self, l1: float, l2: float, l3: float, l4: float, l5: float) -> None:
        self.l1, self.l2, self.l3, self.l4, self.l5 = l1, l2, l3, l4, l5

    def forward_kinematics(
        self, phi1: float, phi4: float
    ) -> tuple[float, float, float, float]:
        x_b = -self.l5 / 2 + math.cos(phi1) * self.l1
        y_b = math.sin(phi1) * self.l1
        x_d = self.l5 / 2 + math.cos(phi4) * self.l4
        y_d = math.sin(phi4) * self.l4
        a = 2 * self.l2 * (x_d - x_b)
        b = 2 * self.l2 * (y_d - y_b)
        c = self.l2**2 + (x_d - x_b) ** 2 + (y_d - y_b) ** 2 - self.l3**2
        phi2 = 2 * math.atan2(b + math.sqrt(a * a + b * b - c * c), a + c)
        x_c = x_b + self.l2 * math.cos(phi2)
        y_c = y_b + self.l2 * math.sin(phi2)
        phi3 = math.atan2(y_c - y_d, x_c - x_d)
        return phi2, phi3, math.hypot(x_c, y_c), math.atan2(y_c, x_c)

    def inverse_kinematics(self, l0: float, phi0: float) -> tuple[float, float]:
        x_c, y_c = l0 * math.cos(phi0), l0 * math.sin(phi0)
        dist_ac = math.hypot(x_c + self.l5 / 2, y_c)
        alpha = math.acos(
            (self.l1**2 + dist_ac**2 - self.l2**2) / (2 * self.l1 * dist_ac)
        )
        phi1 = math.atan2(y_c, x_c + self.l5 / 2) + alpha
        dist_ec = math.hypot(x_c - self.l5 / 2, y_c)
        beta = math.acos(
            (self.l4**2 + dist_ec**2 - self.l3**2) / (2 * self.l4 * dist_ec)
        )
        phi4 = math.atan2(y_c, x_c - self.l5 / 2) - beta
        return phi1, phi4

    def mat_jrm(
        self, phi0: float, phi1: float, phi2: float, phi3: float, phi4: float, l0: float
    ) -> np.ndarray:
        denom = math.sin(phi3 - phi2)
        return np.array(
            [
                [
                    self.l1 * math.sin(phi0 - phi3) * math.sin(phi1 - phi2) / denom,
                    self.l1
                    * math.sin(phi1 - phi2)
                    * math.cos(phi0 - phi3)
                    / (denom * l0),
                ],
                [
                    self.l4 * math.sin(phi0 - phi2) * math.sin(phi3 - phi4) / denom,
                    self.l4
                    * math.sin(phi3 - phi4)
                    * math.cos(phi0 - phi2)
                    / (denom * l0),
                ],
            ]
        )

    def virtual_force_to_joint_torque(
        self, j_t: np.ndarray, leg_force: float, hip_torque: float
    ) -> np.ndarray:
        return j_t @ np.array([leg_force, hip_torque])

    def joint_torque_to_virtual_force(
        self, j_t: np.ndarray, joint_torque: np.ndarray
    ) -> np.ndarray:
        return np.linalg.solve(j_t, joint_torque)


class Leg:
    def __init__(self, model: mujoco.MjModel, side: str) -> None:
        joints = [model.joint(f"{side}_{part}_joint") for part in ("front", "rear")]
        self.qpos = np.array([joint.qposadr[0] for joint in joints])
        self.dofs = np.array([joint.dofadr[0] for joint in joints])
        self.actuators = np.array(
            [model.actuator(f"{side}_{part}_motor").id for part in ("front", "rear")]
        )
        self.signs = np.array([joint.axis[1] for joint in joints])
        cranks = np.array(
            [
                model.body(f"{side}_{part}_child1_link").pos[[0, 2]]
                for part in ("front", "rear")
            ]
        )
        self.offsets = np.arctan2(-cranks[:, 1], cranks[:, 0])
        upper = np.linalg.norm(cranks[0])
        lower = np.linalg.norm(model.body(f"{side}_wheel_link").pos[[0, 2]])
        self.vmc = VMC(upper, lower, lower, upper, 0.0)
        self.length = self.angle = 0.0
        self.velocity = np.zeros(2)
        self.j_t = np.zeros((2, 2))
        self.force = np.zeros(2)

    def update(self, data: mujoco.MjData) -> None:
        phi1, phi4 = self.offsets + self.signs * data.qpos[self.qpos]
        phi2, phi3, self.length, self.angle = self.vmc.forward_kinematics(phi1, phi4)
        self.j_t[:] = self.signs[:, None] * self.vmc.mat_jrm(
            self.angle, phi1, phi2, phi3, phi4, self.length
        )
        self.velocity[:] = self.j_t.T @ data.qvel[self.dofs]
        self.force[:] = self.vmc.joint_torque_to_virtual_force(
            self.j_t, data.qfrc_actuator[self.dofs]
        )

    def torque(self, force: float, moment: float) -> np.ndarray:
        return self.vmc.virtual_force_to_joint_torque(self.j_t, force, moment)
