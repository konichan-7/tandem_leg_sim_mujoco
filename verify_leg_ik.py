import mujoco
import mujoco.viewer
import numpy as np
import time
import math

from utils.vmc import DEFAULT_VMC


def main():
    model = mujoco.MjModel.from_xml_path("MJCF/balance_bot.xml")
    data = mujoco.MjData(model)

    l_front_idx = model.joint("Left_front_joint").qposadr[0]
    l_rear_idx = model.joint("Left_rear_joint").qposadr[0]
    r_front_idx = model.joint("Right_front_joint").qposadr[0]
    r_rear_idx = model.joint("Right_rear_joint").qposadr[0]

    l_front_ctrl = model.actuator("Left_front_motor").id
    l_rear_ctrl = model.actuator("Left_rear_motor").id
    r_front_ctrl = model.actuator("Right_front_motor").id
    r_rear_ctrl = model.actuator("Right_rear_motor").id

    target_L0, target_phi0 = 0.25, math.pi / 4
    actual_L0, actual_phi0, current_target_L0 = 0.25, math.pi / 2, 0.25
    paused = False

    print("Starting inverse_kinematics Verification...")
    print("TIP: Use the 'Watch' panel in the MuJoCo GUI (F2) to monitor variables.")
    print("Press Ctrl+C in terminal to exit.")

    base_init_pos = np.array([0, 0, 0.6])
    base_init_quat = np.array([1, 0, 0, 0])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if not paused:
                # Keep suspended
                data.qpos[0:3] = base_init_pos
                data.qpos[3:7] = base_init_quat
                data.qvel[0:6] = 0

                # Command
                current_target_L0 = target_L0 + 0.05 * math.sin(data.time * 2)
                p1, p4 = DEFAULT_VMC.inverse_kinematics(
                    current_target_L0, math.pi - target_phi0
                )

                # inverse_kinematics -> Joint Angles
                q_l_front_target, q_l_rear_target = math.pi - p1, p4
                q_r_front_target, q_r_rear_target = p1 - math.pi, -p4

                print(
                    f"phi1_l: {math.degrees(data.qpos[l_front_idx]):.2f}°, phi4_l: {math.degrees(data.qpos[l_rear_idx]):.2f}° | phi1_r: {math.degrees(data.qpos[r_front_idx]):.2f}°, phi4_r: {math.degrees(data.qpos[r_rear_idx]):.2f}°"
                )

                # PD control
                kp, kd = 1000.0, 50.0
                data.ctrl[l_front_ctrl] = (
                    kp * (q_l_front_target - data.qpos[l_front_idx])
                    - kd * data.qvel[model.joint("Left_front_joint").dofadr[0]]
                )
                data.ctrl[l_rear_ctrl] = (
                    kp * (q_l_rear_target - data.qpos[l_rear_idx])
                    - kd * data.qvel[model.joint("Left_rear_joint").dofadr[0]]
                )
                data.ctrl[r_front_ctrl] = (
                    kp * (q_r_front_target - data.qpos[r_front_idx])
                    - kd * data.qvel[model.joint("Right_front_joint").dofadr[0]]
                )
                data.ctrl[r_rear_ctrl] = (
                    kp * (q_r_rear_target - data.qpos[r_rear_idx])
                    - kd * data.qvel[model.joint("Right_rear_joint").dofadr[0]]
                )

                # FK Verification
                phi_l1, phi_l4 = math.pi - data.qpos[l_front_idx], data.qpos[l_rear_idx]
                _, _, actual_L0, raw_phi0 = DEFAULT_VMC.forward_kinematics(
                    phi_l1, phi_l4
                )
                actual_phi0 = math.pi - raw_phi0
                print(
                    f"Time: {data.time:.2f}s | Target phi0: {target_phi0:.3f} | Actual phi0: {actual_phi0:.3f} | Error: {abs(target_phi0-actual_phi0):.4f}"
                )

                # Log to terminal every 0.5s
                # if int(data.time * 1000) % 500 == 0:
                #     print(f"T: {data.time:.1f}s | Target L0: {current_target_L0:.3f} | Actual L0: {actual_L0:.3f} | Error: {abs(current_target_L0-actual_L0):.4f}")

                mujoco.mj_step(model, data)

            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
