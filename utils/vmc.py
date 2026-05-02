import math

import numpy as np

from .math_tools import wrap


class VMC:
    def __init__(
        self,
        l1: float,
        l2: float,
        l3: float,
        l4: float,
        l5: float,
    ) -> None:
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5

    def forward_kinematics(
        self, phi1: float, phi4: float
    ) -> tuple[float, float, float, float]:
        try:
            x_b = -self.l5 / 2 + math.cos(phi1) * self.l1
            y_b = math.sin(phi1) * self.l1
            x_d = self.l5 / 2 + math.cos(phi4) * self.l4
            y_d = math.sin(phi4) * self.l4
            a_0 = 2 * self.l2 * (x_d - x_b)
            b_0 = 2 * self.l2 * (y_d - y_b)
            l_bd = ((x_d - x_b) ** 2 + (y_d - y_b) ** 2) ** 0.5
            c_0 = self.l2**2 + l_bd**2 - self.l3**2

            val = a_0**2 + b_0**2 - c_0**2
            if val < 0:
                val = 0
            phi2 = wrap(2 * math.atan2(b_0 + val**0.5, a_0 + c_0))

            x_c = -self.l5 / 2 + self.l1 * math.cos(phi1) + self.l2 * math.cos(phi2)
            y_c = self.l1 * math.sin(phi1) + self.l2 * math.sin(phi2)
            phi3 = wrap(math.atan2(y_c - y_d, x_c - x_d))
            l_0 = (x_c**2 + y_c**2) ** 0.5
            phi_0 = math.atan2(y_c, x_c)
            return phi2, phi3, l_0, phi_0
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def inverse_kinematics(self, l0: float, phi0: float) -> tuple[float, float]:
        x_c = l0 * math.cos(phi0)
        y_c = l0 * math.sin(phi0)

        dist_ac = math.sqrt((x_c + self.l5 / 2) ** 2 + y_c**2)
        cos_alpha = (self.l1**2 + dist_ac**2 - self.l2**2) / (2 * self.l1 * dist_ac)
        alpha = math.acos(np.clip(cos_alpha, -1, 1))
        angle_ac = math.atan2(y_c, x_c + self.l5 / 2)
        phi1 = angle_ac + alpha

        dist_ec = math.sqrt((x_c - self.l5 / 2) ** 2 + y_c**2)
        cos_beta = (self.l4**2 + dist_ec**2 - self.l3**2) / (2 * self.l4 * dist_ec)
        beta = math.acos(np.clip(cos_beta, -1, 1))
        angle_ec = math.atan2(y_c, x_c - self.l5 / 2)
        phi4 = angle_ec - beta
        return phi1, phi4

    def mat_jrm(
        self,
        phi0: float,
        phi1: float,
        phi2: float,
        phi3: float,
        phi4: float,
        l0: float,
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
            ],
            dtype=np.float64,
        )

    def virtual_force_to_joint_torque(
        self,
        j_t: np.ndarray,
        leg_force: float,
        hip_torque: float,
    ) -> np.ndarray:
        return j_t @ np.array([leg_force, hip_torque], dtype=np.float64)

    def joint_torque_to_virtual_force(
        self,
        j_t: np.ndarray,
        joint_torque: np.ndarray,
    ) -> np.ndarray:
        return np.linalg.solve(j_t, np.asarray(joint_torque, dtype=np.float64))
