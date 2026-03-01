import mujoco
import mujoco.viewer
import numpy as np
import time
import math

# Leg Kinematics Parameters (from mujoco_controller.py)
L1, L2, L3, L4, L5 = 0.215, 0.254, 0.254, 0.215, 0.0


def getPhi(phi1, phi4, l1, l2, l3, l4, l5):
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
        phi2 = 2 * math.atan2(B_0 + val**0.5, A_0 + C_0)

        x_C = -l5 / 2 + l1 * math.cos(phi1) + l2 * math.cos(phi2)
        y_C = 0 + l1 * math.sin(phi1) + l2 * math.sin(phi2)
        phi3 = math.atan2(y_C - y_D, x_C - x_D)
        l_0 = (x_C**2 + y_C**2) ** 0.5
        phi_0 = math.atan2(y_C, x_C)
        return phi2, phi3, l_0, phi_0
    except Exception as e:
        return 0, 0, 0.15, 1.57


def ik(L0, phi0, l1, l2, l3, l4, l5):
    # IK: (L0, phi0) -> (phi1, phi4)
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

    target_L0, target_phi0 = 0.25, math.pi / 2
    actual_L0, actual_phi0, current_target_L0 = 0.25, math.pi / 2, 0.25
    paused = False

    print("Starting IK Verification...")
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
                p1, p4 = ik(current_target_L0, target_phi0, L1, L2, L3, L4, L5)

                # IK -> Joint Angles
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
                _, _, actual_L0, actual_phi0 = getPhi(
                    phi_l1, phi_l4, L1, L2, L3, L4, L5
                )
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
