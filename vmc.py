import math

import numpy as np

L1 = 0.215
L2 = 0.254
L3 = 0.254
L4 = 0.215
L5 = 0.0


def wrap(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a: float, b: float) -> float:
    return wrap(a - b)


def getPhi(
    phi1: float,
    phi4: float,
    l1: float = L1,
    l2: float = L2,
    l3: float = L3,
    l4: float = L4,
    l5: float = L5,
) -> tuple[float, float, float, float]:
    try:
        x_B = -l5 / 2 + math.cos(phi1) * l1
        y_B = math.sin(phi1) * l1
        x_D = l5 / 2 + math.cos(phi4) * l4
        y_D = math.sin(phi4) * l4
        A_0 = 2 * l2 * (x_D - x_B)
        B_0 = 2 * l2 * (y_D - y_B)
        l_BD = ((x_D - x_B) ** 2 + (y_D - y_B) ** 2) ** 0.5
        C_0 = l2**2 + l_BD**2 - l3**2

        val = A_0**2 + B_0**2 - C_0**2
        if val < 0:
            val = 0
        phi2 = wrap(2 * math.atan2(B_0 + val**0.5, A_0 + C_0))

        x_C = -l5 / 2 + l1 * math.cos(phi1) + l2 * math.cos(phi2)
        y_C = l1 * math.sin(phi1) + l2 * math.sin(phi2)
        phi3 = wrap(math.atan2(y_C - y_D, x_C - x_D))
        l_0 = (x_C**2 + y_C**2) ** 0.5
        phi_0 = math.atan2(y_C, x_C)
        return phi2, phi3, l_0, phi_0
    except Exception:
        return 0.0, 0.0, 0.15, 1.57


def ik(
    L0: float,
    phi0: float,
    l1: float = L1,
    l2: float = L2,
    l3: float = L3,
    l4: float = L4,
    l5: float = L5,
) -> tuple[float, float]:
    xC = L0 * math.cos(phi0)
    yC = L0 * math.sin(phi0)

    dist_AC = math.sqrt((xC + l5 / 2) ** 2 + yC**2)
    cos_alpha = (l1**2 + dist_AC**2 - l2**2) / (2 * l1 * dist_AC)
    alpha = math.acos(np.clip(cos_alpha, -1, 1))
    angle_AC = math.atan2(yC, xC + l5 / 2)
    phi1 = angle_AC + alpha

    dist_EC = math.sqrt((xC - l5 / 2) ** 2 + yC**2)
    cos_beta = (l4**2 + dist_EC**2 - l3**2) / (2 * l4 * dist_EC)
    beta = math.acos(np.clip(cos_beta, -1, 1))
    angle_EC = math.atan2(yC, xC - l5 / 2)
    phi4 = angle_EC - beta
    return phi1, phi4


def Mat_JRM(
    phi0: float,
    phi1: float,
    phi2: float,
    phi3: float,
    phi4: float,
    L0: float,
    l1: float = L1,
    l4: float = L4,
) -> np.ndarray:
    denom = math.sin(phi3 - phi2)
    return np.array(
        [
            [
                l1 * math.sin(phi0 - phi3) * math.sin(phi1 - phi2) / denom,
                l1 * math.sin(phi1 - phi2) * math.cos(phi0 - phi3) / (denom * L0),
            ],
            [
                l4 * math.sin(phi0 - phi2) * math.sin(phi3 - phi4) / denom,
                l4 * math.sin(phi3 - phi4) * math.cos(phi0 - phi2) / (denom * L0),
            ],
        ],
        dtype=np.float64,
    )


def virtual_force_to_joint_torque(
    j_t: np.ndarray,
    leg_force: float,
    hip_torque: float,
) -> np.ndarray:
    return j_t @ np.array([leg_force, hip_torque], dtype=np.float64)
