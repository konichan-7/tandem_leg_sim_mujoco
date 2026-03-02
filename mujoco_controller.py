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


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def angle_diff(a, b):
    return wrap(a - b)


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
        phi2 = wrap(2 * math.atan2(B_0 + val**0.5, A_0 + C_0))

        x_C = -l5 / 2 + l1 * math.cos(phi1) + l2 * math.cos(phi2)
        y_C = 0 + l1 * math.sin(phi1) + l2 * math.sin(phi2)
        phi3 = wrap(math.atan2(y_C - y_D, x_C - x_D))
        l_0 = (x_C**2 + y_C**2) ** 0.5
        phi_0 = wrap(math.pi - math.atan2(y_C, x_C))
        return phi2, phi3, l_0, phi_0
    except:
        return 0, 0, 0.15, 1.57


def ik(L0, phi0, l1, l2, l3, l4, l5):
    phi0 = math.pi - phi0
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


def Mat_JRM(phi0, phi1, phi2, phi3, phi4, L0, l1, l4):
    denom = math.sin(phi3 - phi2)
    if abs(denom) < 1e-4:
        denom = 1e-4 * (1 if denom >= 0 else -1)
    JRM = np.matrix(
        [
            [
                l1 * math.sin(phi0 - phi3) * math.sin(phi1 - phi2) / denom,
                l1 * math.sin(phi1 - phi2) * math.cos(phi0 - phi3) / (denom * L0),
            ],
            [
                l4 * math.sin(phi0 - phi2) * math.sin(phi3 - phi4) / denom,
                l4 * math.sin(phi3 - phi4) * math.cos(phi0 - phi2) / (denom * L0),
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
            self.kp * error + self.ki * self.integral + self.kd * derivative, -100, 100
        )


def quat_to_euler(quat):
    w, x, y, z = quat
    pitch = wrap(math.asin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = wrap(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return pitch, yaw


def main():
    model = mujoco.MjModel.from_xml_path("MJCF/balance_bot.xml")
    data = mujoco.MjData(model)

    yaml_path = os.path.join(os.path.dirname(__file__), "sp_lqr", "sjtu.yaml")
    K, _, _, _, _ = compute_lqr_controller(yaml_path, verbose=True)

    target_L0 = 0.15
    target_phi0 = math.pi / 2
    kp, kd = 100.0, 10.0

    F0_l = PID_control(1000, 0, 100, target_L0)
    F0_r = PID_control(1000, 0, 100, target_L0)

    l_front_idx = model.joint("Left_front_joint").qposadr[0]
    l_rear_idx = model.joint("Left_rear_joint").qposadr[0]
    r_front_idx = model.joint("Right_front_joint").qposadr[0]
    r_rear_idx = model.joint("Right_rear_joint").qposadr[0]

    l_front_dof = model.joint("Left_front_joint").dofadr[0]
    l_rear_dof = model.joint("Left_rear_joint").dofadr[0]
    r_front_dof = model.joint("Right_front_joint").dofadr[0]
    r_rear_dof = model.joint("Right_rear_joint").dofadr[0]

    l_front_ctrl = model.actuator("Left_front_motor").id
    l_rear_ctrl = model.actuator("Left_rear_motor").id
    r_front_ctrl = model.actuator("Right_front_motor").id
    r_rear_ctrl = model.actuator("Right_rear_motor").id

    # Init pose
    data.qpos[2] = 0.25
    data.qpos[l_front_idx] = 1.0
    data.qpos[l_rear_idx] = -1.0
    data.qpos[r_front_idx] = -1.0
    data.qpos[r_rear_idx] = 1.0

    last_phi0_l, last_phi0_r = 0, 0
    phi0_initialized = False
    lqr_active = False
    lqr_debug_until = 0.0
    lqr_last_print_time = -1.0
    target_velocity, target_yaw = 0, 0

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
                if data.time < 0.8:
                    data.ctrl[:] = 0
                    lqr_active = False
                elif data.time < 1:
                    p1, p4 = ik(target_L0, target_phi0, L1, L2, L3, L4, L5)

                    q_l_front_target = wrap(math.pi - p1)
                    q_l_rear_target = wrap(p4)
                    q_r_front_target = wrap(p1 - math.pi)
                    q_r_rear_target = wrap(-p4)

                    data.ctrl[l_front_ctrl] = (
                        kp * wrap(q_l_front_target - data.qpos[l_front_idx])
                        - kd * data.qvel[l_front_dof]
                    )
                    data.ctrl[l_rear_ctrl] = (
                        kp * wrap(q_l_rear_target - data.qpos[l_rear_idx])
                        - kd * data.qvel[l_rear_dof]
                    )
                    data.ctrl[r_front_ctrl] = (
                        kp * wrap(q_r_front_target - data.qpos[r_front_idx])
                        - kd * data.qvel[r_front_dof]
                    )
                    data.ctrl[r_rear_ctrl] = (
                        kp * wrap(q_r_rear_target - data.qpos[r_rear_idx])
                        - kd * data.qvel[r_rear_dof]
                    )
                    lqr_active = False
                else:
                    sensor_adr = model.sensor_adr[baselink_quat_id]
                    base_quat = data.sensordata[sensor_adr : sensor_adr + 4]

                    pitch, yaw = quat_to_euler(base_quat)
                    dot_pitch = data.qvel[4]
                    dot_yaw = data.qvel[5]

                    phi_l1 = wrap(math.pi - data.qpos[l_front_idx])
                    phi_l4 = wrap(data.qpos[l_rear_idx])
                    phi_r1 = wrap(math.pi + data.qpos[r_front_idx])
                    phi_r4 = wrap(-data.qpos[r_rear_idx])

                    p2l, p3l, L0_l, phi0_l = getPhi(phi_l1, phi_l4, L1, L2, L3, L4, L5)
                    p2r, p3r, L0_r, phi0_r = getPhi(phi_r1, phi_r4, L1, L2, L3, L4, L5)

                    theta_ll = wrap(-math.pi / 2 + phi0_l + pitch)
                    theta_lr = wrap(-math.pi / 2 + phi0_r + pitch)

                    if not phi0_initialized:
                        last_phi0_l, last_phi0_r = phi0_l, phi0_r
                        phi0_initialized = True
                        dot_phi0_l, dot_phi0_r = 0, 0
                    else:
                        dot_phi0_l = angle_diff(phi0_l, last_phi0_l) / dt
                        dot_phi0_r = angle_diff(phi0_r, last_phi0_r) / dt
                        last_phi0_l, last_phi0_r = phi0_l, phi0_r

                    dot_theta_ll = dot_phi0_l + dot_pitch
                    dot_theta_lr = dot_phi0_r + dot_pitch

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

                    U = K * (expect_state - real_state)
                    T_l, T_r, T_pl, T_pr = (
                        U[0, 0].item(),
                        U[1, 0].item(),
                        U[2, 0].item(),
                        U[3, 0].item(),
                    )

                    F0_l.target = target_L0
                    F0_r.target = target_L0
                    dF_0_l = F0_l.position_pid(L0_l, dt)
                    dF_0_r = F0_r.position_pid(L0_r, dt)
                    gravity_l = 65 / math.cos(theta_ll)
                    gravity_r = 65 / math.cos(theta_lr)
                    F_bl = gravity_l + dF_0_l
                    F_br = gravity_r + dF_0_r
                    F_bl = np.clip(F_bl, -120, 120)
                    F_br = np.clip(F_br, -120, 120)

                    JRM_L = Mat_JRM(phi0_l, phi_l1, p2l, p3l, phi_l4, L0_l, L1, L4)
                    JRM_R = Mat_JRM(phi0_r, phi_r1, p2r, p3r, phi_r4, L0_r, L1, L4)
                    T_JL = JRM_L * np.matrix([[F_bl], [T_pl]])
                    T_JR = JRM_R * np.matrix([[F_br], [T_pr]])

                    data.ctrl[0] = np.clip(T_JL[0, 0], -60, 60)
                    data.ctrl[1] = np.clip(T_JL[1, 0], -60, 60)
                    data.ctrl[2] = np.clip(-T_JR[0, 0], -60, 60)
                    data.ctrl[3] = np.clip(T_JR[1, 0], -60, 60)
                    data.ctrl[4] = np.clip(T_l, -4.5, 4.5)
                    data.ctrl[5] = np.clip(-T_r, -4.5, 4.5)

                    print(
                        f"[LQR dbg] t={data.time:.3f} "
                        f"L0=({L0_l:.3f},{L0_r:.3f}) "
                        f"phi0=({phi0_l:.3f},{phi0_r:.3f}) "
                        f"th=({theta_ll:.3f},{theta_lr:.3f}) "
                        f"dth=({dot_theta_ll:.3f},{dot_theta_lr:.3f}) "
                        f"pitch={pitch:.3f} dpitch={dot_pitch:.3f}"
                    )
                    print(
                        f"F=({F_bl:.3f},{F_br:.3f}) "
                        f"T_pl1={data.ctrl[0]:.3f} T_pl2={data.ctrl[1]:.3f} "
                        f"T_pr1={data.ctrl[2]:.3f} T_pr2={data.ctrl[3]:.3f} "
                        f"TJ=({T_JL[0, 0]:.3f},{T_JL[1, 0]:.3f},{T_JR[0, 0]:.3f},{T_JR[1, 0]:.3f})"
                    )

                mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
