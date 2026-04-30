from pathlib import Path
import time

import mujoco
import mujoco.viewer

from utils import VMC

XML_PATH = Path(__file__).parent / "MJCF" / "demo" / "demo.xml"
BASE_QPOS = [0.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0]

LEFT_PHI1_ECD = -2.68
LEFT_PHI4_ECD = -0.62
RIGHT_PHI1_ECD = -2.68
RIGHT_PHI4_ECD = -0.62
LEFT_FRONT_JOINT = "left_front_joint"
LEFT_REAR_JOINT = "left_rear_joint"
RIGHT_FRONT_JOINT = "right_front_joint"
RIGHT_REAR_JOINT = "right_rear_joint"

vmc = VMC(0.16, 0.169, 0.169, 0.16, 0)


def joint_angle(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    return float(data.qpos[model.joint(name).qposadr[0]])


def leg_phi(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    return {
        "left_phi1": joint_angle(model, data, LEFT_FRONT_JOINT) - LEFT_PHI1_ECD,
        "left_phi4": joint_angle(model, data, LEFT_REAR_JOINT) - LEFT_PHI4_ECD,
        "right_phi1": joint_angle(model, data, RIGHT_FRONT_JOINT) - RIGHT_PHI1_ECD,
        "right_phi4": joint_angle(model, data, RIGHT_REAR_JOINT) - RIGHT_PHI4_ECD,
    }


def vmc_state(phi1: float, phi4: float) -> tuple[float, float, float, float]:
    phi2, phi3, l0, phi0 = vmc.forward_kinematics(phi1, phi4)
    return phi2, phi3, l0, phi0


def hang_base(data: mujoco.MjData) -> None:
    data.qpos[:7] = BASE_QPOS
    data.qvel[:6] = 0


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    hang_base(data)
    mujoco.mj_forward(model, data)

    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            data.ctrl[:] = 0
            mujoco.mj_step(model, data)
            hang_base(data)

            if data.time - last_print >= 0.05:
                last_print = data.time
                phi = leg_phi(model, data)
                left_phi2, left_phi3, left_l0, left_phi0 = vmc_state(
                    phi["left_phi1"], phi["left_phi4"]
                )
                right_phi2, right_phi3, right_l0, right_phi0 = vmc_state(
                    phi["right_phi1"], phi["right_phi4"]
                )
                print(
                    f"left phi1={phi['left_phi1']:+.6f} phi4={phi['left_phi4']:+.6f} "
                    f"phi2={left_phi2:+.6f} phi3={left_phi3:+.6f} "
                    f"l0={left_l0:.6f} phi0={left_phi0:+.6f} | "
                    f"right phi1={phi['right_phi1']:+.6f} phi4={phi['right_phi4']:+.6f} "
                    f"phi2={right_phi2:+.6f} phi3={right_phi3:+.6f} "
                    f"l0={right_l0:.6f} phi0={right_phi0:+.6f}"
                )

            viewer.sync()
            sleep_time = model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
