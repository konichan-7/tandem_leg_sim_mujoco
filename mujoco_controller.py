import mujoco
import mujoco.viewer
import numpy as np
import time
import math
import scipy.linalg
import sys
import os

# Add sp_lqr to path to import sjtu modules
sys.path.append(os.path.join(os.path.dirname(__file__), "sp_lqr"))
from sjtu.lqr import compute_lqr_controller

# Leg Kinematics (derived from leg.py)
# Note: Using Webots link lengths as they define the intended robot geometry
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
    except:
        return 0, 0, 0.15, 1.57


def Mat_JRM(phi0, phi1, phi2, phi3, phi4, L0, l1, l4):
    denom = math.sin(phi2 - phi3)
    if abs(denom) < 1e-4:
        denom = 1e-4 * (1 if denom >= 0 else -1)
    JRM = np.matrix(
        [
            [
                -l1 * math.sin(phi0 - phi3) * math.sin(phi1 - phi2) / denom,
                -l1 * math.sin(phi1 - phi2) * math.cos(phi0 - phi3) / (L0 * denom),
            ],
            [
                -l4 * math.sin(phi0 - phi2) * math.sin(phi3 - phi4) / denom,
                -l4 * math.sin(phi3 - phi4) * math.cos(phi0 - phi2) / (L0 * denom),
            ],
        ]
    )
    return JRM


def spd(dphi1, dphi4, l1, l2, l3, l4, l5, phi1, phi4):
    # This is a complex derivation from leg.py, using finite diff for L0_speed and phi0_speed is safer
    # unless we exactly replicate the symbolic code. We'll use finite diff for dot_L0 and dot_phi0.
    pass


class PID_control:
    def __init__(self, kp, ki, kd, target):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.target = target
        self.integral = 0
        self.last_error = 0

    def position_pid(self, current, dt):
        error = self.target - current
        self.integral += error * dt
        self.integral = np.clip(self.integral, -100, 100)
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return np.clip(
            self.kp * error + self.ki * self.integral + self.kd * derivative, -20, 20
        )


def quat_to_euler(quat):
    w, x, y, z = quat
    pitch = math.asin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return pitch, yaw


def main():
    model = mujoco.MjModel.from_xml_path("MJCF/balance_bot.xml")
    data = mujoco.MjData(model)

    yaml_path = os.path.join(os.path.dirname(__file__), "sp_lqr", "sjtu.yaml")
    K, _, _, _, _ = compute_lqr_controller(yaml_path, verbose=True)

    target_L0 = 0.15

    F0_l = PID_control(200, 0, 10, target_L0)
    F0_r = PID_control(200, 0, 10, target_L0)

    l_front_idx = model.joint("Left_front_joint").qposadr[0]
    l_rear_idx = model.joint("Left_rear_joint").qposadr[0]
    r_front_idx = model.joint("Right_front_joint").qposadr[0]
    r_rear_idx = model.joint("Right_rear_joint").qposadr[0]

    # Init pose
    data.qpos[2] = 0.25
    data.qpos[l_front_idx] = 1.0
    data.qpos[l_rear_idx] = -1.0
    data.qpos[r_front_idx] = -1.0
    data.qpos[r_rear_idx] = 1.0

    last_theta_ll, last_theta_lr = 0, 0
    last_phi0_l, last_phi0_r = 0, 0
    target_velocity, target_yaw = 0, 0

    # Pre-lookup sensor ID for efficiency
    baselink_quat_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "baselink_quat"
    )

    paused = False

    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == " ":
            paused = not paused

    print("MuJoCo Controller Started. Press SPACE to pause.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            if not paused:
                # State extraction from sensors
                sensor_adr = model.sensor_adr[baselink_quat_id]
                base_quat = data.sensordata[sensor_adr : sensor_adr + 4]

                raw_pitch, yaw = quat_to_euler(base_quat)
                pitch = raw_pitch

                # Print pitch value every 0.1s (assuming 1ms timestep)
                print(f"Time: {data.time:.2f}s | Pitch: {math.degrees(pitch):.2f}°")

                h, dot_pitch, dot_yaw = pitch, -data.qvel[4], data.qvel[5]

                # Phi mapping (Webots logic)
                phi_l1 = math.pi / 2 + data.qpos[l_front_idx]
                phi_l4 = data.qpos[l_rear_idx]
                # print(f"Phi_l1: {math.degrees(phi_l1):.2f}°, Phi_l4: {math.degrees(phi_l4):.2f}°")
                phi_r1 = math.pi / 2 - data.qpos[r_front_idx]
                phi_r4 = -data.qpos[r_rear_idx]
                print(
                    f"Phi_r1: {math.degrees(phi_r1):.2f}°, Phi_r4: {math.degrees(phi_r4):.2f}°"
                )

                # Kinematics
                p2l, p3l, L0_l, phi0_l = getPhi(phi_l1, phi_l4, L1, L2, L3, L4, L5)
                p2r, p3r, L0_r, phi0_r = getPhi(phi_r1, phi_r4, L1, L2, L3, L4, L5)

                theta_ll = -(math.pi / 2 - phi0_l + pitch)
                theta_lr = -(math.pi / 2 - phi0_r + pitch)
                print(
                    f"Theta_ll: {math.degrees(theta_ll):.2f}°, Theta_lr: {math.degrees(theta_lr):.2f}°"
                )
                print(
                    f"phi0_l: {math.degrees(phi0_l):.2f}°, phi0_r: {math.degrees(phi0_r):.2f}°"
                )

                dot_phi0_l = (phi0_l - last_phi0_l) / dt
                dot_phi0_r = (phi0_r - last_phi0_r) / dt
                last_phi0_l, last_phi0_r = phi0_l, phi0_r

                dot_theta_ll = -dot_phi0_l + dot_pitch
                dot_theta_lr = -dot_phi0_r + dot_pitch

                real_state = np.matrix(
                    [
                        [data.qpos[0]],
                        [data.qvel[0]],
                        [yaw],
                        [dot_yaw],
                        [theta_ll],
                        [dot_theta_ll],
                        [theta_lr],
                        [dot_theta_lr],
                        [pitch],
                        [dot_pitch],
                    ]
                )
                print(
                    f"State: s={real_state[0,0]:.3f}, ds={real_state[1,0]:.3f}, phi={math.degrees(real_state[2,0]):.2f}°, dphi={math.degrees(real_state[3,0]):.2f}°"
                )

                expect_state = np.matrix(
                    [
                        [0],
                        [target_velocity],
                        [target_yaw],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                    ]
                )

                # LQR Control
                U = K * (expect_state - real_state)
                T_l, T_r, T_pl, T_pr = (
                    U[0, 0].item(),
                    U[1, 0].item(),
                    U[2, 0].item(),
                    U[3, 0].item(),
                )

                # VMC for length
                dF_0_l = -F0_l.position_pid(L0_l, dt)
                dF_0_r = -F0_r.position_pid(L0_r, dt)
                F_bl = -(dF_0_l - 65 / math.cos(np.clip(theta_ll, -0.5, 0.5)))
                F_br = -(dF_0_r + 65 / math.cos(np.clip(theta_lr, -0.5, 0.5)))

                # Jacobian mapping
                JRM_L = Mat_JRM(phi0_l, phi_l1, p2l, p3l, phi_l4, L0_l, L1, L4)
                JRM_R = Mat_JRM(phi0_r, phi_r1, p2r, p3r, phi_r4, L0_r, L1, L4)
                T_JL = JRM_L * np.matrix([[F_bl], [T_pl]])
                T_JR = JRM_R * np.matrix([[F_br], [T_pr]])

                # Final Actuation mapping (Methodical derived signs)
                data.ctrl[0] = -T_JL[0, 0]
                data.ctrl[1] = T_JL[1, 0]
                data.ctrl[2] = T_JR[0, 0]
                data.ctrl[3] = -T_JR[1, 0]
                data.ctrl[4] = T_l
                data.ctrl[5] = T_r

                mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
