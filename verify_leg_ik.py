import numpy as np

from demo import VMC_GEOMETRY
from utils.math_tools import angle_diff
from utils.vmc import VMC


def verify_inverse_kinematics() -> tuple[float, float]:
    vmc = VMC(
        VMC_GEOMETRY.l1,
        VMC_GEOMETRY.l2,
        VMC_GEOMETRY.l3,
        VMC_GEOMETRY.l4,
        VMC_GEOMETRY.l5,
    )
    max_length_error = 0.0
    max_angle_error = 0.0

    for target_length in np.linspace(0.05, 0.30, 26):
        for target_angle in np.linspace(0.7, 2.4, 18):
            phi1, phi4 = vmc.inverse_kinematics(target_length, target_angle)
            _, _, actual_length, actual_angle = vmc.forward_kinematics(phi1, phi4)
            max_length_error = max(
                max_length_error,
                abs(actual_length - target_length),
            )
            max_angle_error = max(
                max_angle_error,
                abs(angle_diff(actual_angle, target_angle)),
            )

    tolerance = 1e-10
    if max(max_length_error, max_angle_error) > tolerance:
        raise RuntimeError(
            f"IK/FK mismatch: length={max_length_error:.3e}, "
            f"angle={max_angle_error:.3e}"
        )
    return max_length_error, max_angle_error


def main() -> None:
    length_error, angle_error = verify_inverse_kinematics()
    print(f"max length error: {length_error:.3e}")
    print(f"max angle error: {angle_error:.3e}")


if __name__ == "__main__":
    main()
