import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "sp_lqr"))
from mujoco_mpc import (
    DEFAULT_EXPORT_PATH,
    MujocoMpcController,
    load_or_export_mujoco_mpc,
)

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
        y_C = l1 * math.sin(phi1) + l2 * math.sin(phi2)
        phi3 = wrap(math.atan2(y_C - y_D, x_C - x_D))
        l_0 = (x_C**2 + y_C**2) ** 0.5
        phi_0 = math.atan2(y_C, x_C)
        return phi2, phi3, l_0, phi_0
    except Exception:
        return 0, 0, 0.15, 1.57


def ik(L0, phi0, l1, l2, l3, l4, l5):
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


class PID_control:
    def __init__(self, kp, ki, kd, target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0
        self.last_error = 0

    def position_pid(self, current, dt):
        error = self.target - current
        self.integral += error * dt
        self.integral = np.clip(self.integral, -500, 500)
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return np.clip(
            self.kp * error + self.ki * self.integral + self.kd * derivative,
            -200,
            200,
        )


def quat_to_euler(quat):
    w, x, y, z = quat
    pitch = wrap(math.asin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = wrap(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return pitch, -yaw


def load_wheel_radius(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    return params["R_w"]


def main():
    model = mujoco.MjModel.from_xml_path("MJCF/balance_bot.xml")
    data = mujoco.MjData(model)

    yaml_path = os.path.join(os.path.dirname(__file__), "sp_lqr", "sjtu.yaml")
    wheel_radius = load_wheel_radius(yaml_path)
    mpc_model = load_or_export_mujoco_mpc(
        yaml_path=os.path.join(os.path.dirname(__file__), "sp_lqr", "sjtu.yaml"),
        export_path=DEFAULT_EXPORT_PATH,
        sample_time=0.01,
        horizon=10,
    )
    mpc_controller = MujocoMpcController(mpc_model)
    mpc_interval_steps = max(1, int(round(mpc_model.sample_time / model.opt.timestep)))

    target_L0 = 0.15
    target_phi0 = math.pi / 2
    kp = 800.0
    kd = 100.0

    left_leg_pid = PID_control(3000, 0, 150, target_L0)
    right_leg_pid = PID_control(3000, 0, 150, target_L0)

    l_front_idx = model.joint("Left_front_joint").qposadr[0]
    l_rear_idx = model.joint("Left_rear_joint").qposadr[0]
    r_front_idx = model.joint("Right_front_joint").qposadr[0]
    r_rear_idx = model.joint("Right_rear_joint").qposadr[0]

    l_front_dof = model.joint("Left_front_joint").dofadr[0]
    l_rear_dof = model.joint("Left_rear_joint").dofadr[0]
    r_front_dof = model.joint("Right_front_joint").dofadr[0]
    r_rear_dof = model.joint("Right_rear_joint").dofadr[0]
    l_wheel_dof = model.joint("Left_Wheel_joint").dofadr[0]
    r_wheel_dof = model.joint("Right_Wheel_joint").dofadr[0]

    l_front_ctrl = model.actuator("Left_front_motor").id
    l_rear_ctrl = model.actuator("Left_rear_motor").id
    r_front_ctrl = model.actuator("Right_front_motor").id
    r_rear_ctrl = model.actuator("Right_rear_motor").id
    l_wheel_ctrl = model.actuator("Left_Wheel_motor").id
    r_wheel_ctrl = model.actuator("Right_Wheel_motor").id

    data.qpos[2] = 0.25
    data.qpos[l_front_idx] = 1.0
    data.qpos[l_rear_idx] = -1.0
    data.qpos[r_front_idx] = -1.0
    data.qpos[r_rear_idx] = 1.0

    last_phi0_l = 0.0
    last_phi0_r = 0.0
    last_pitch = 0.0
    last_yaw = 0.0
    phi0_initialized = False
    pitch_initialized = False
    yaw_initialized = False
    s = 0.0
    cached_control = np.zeros(4, dtype=np.float64)
    expect_state = np.zeros(mpc_model.nx, dtype=np.float64)
    expect_state[4] = 0
    expect_state[6] = 0
    mpc_step_counter = 0
    last_debug_print_time = -1.0

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
                dtheta_wl = data.qvel[l_wheel_dof]
                dtheta_wr = -data.qvel[r_wheel_dof]
                ds = 0.5 * wheel_radius * (dtheta_wl + dtheta_wr)
                s += ds * dt

                # 无论是否处于MPC阶段，每一帧都计算状态，防止状态机切换时导数出现毛刺(spike)
                sensor_adr = model.sensor_adr[baselink_quat_id]
                base_quat = data.sensordata[sensor_adr : sensor_adr + 4]
                pitch, yaw = quat_to_euler(base_quat)

                if yaw_initialized:
                    dot_yaw = angle_diff(yaw, last_yaw) / dt
                else:
                    dot_yaw = 0.0
                    yaw_initialized = True
                last_yaw = yaw

                if pitch_initialized:
                    dot_pitch = angle_diff(pitch, last_pitch) / dt
                else:
                    dot_pitch = 0.0
                    pitch_initialized = True
                last_pitch = pitch

                phi_l1 = wrap(math.pi - data.qpos[l_rear_idx])
                phi_l4 = wrap(data.qpos[l_front_idx])
                phi_r1 = wrap(math.pi + data.qpos[r_rear_idx])
                phi_r4 = wrap(-data.qpos[r_front_idx])

                p2l, p3l, L0_l, phi0_l = getPhi(phi_l1, phi_l4, L1, L2, L3, L4, L5)
                p2r, p3r, L0_r, phi0_r = getPhi(phi_r1, phi_r4, L1, L2, L3, L4, L5)

                theta_ll = wrap(-math.pi / 2 + phi0_l + pitch)
                theta_lr = wrap(-math.pi / 2 + phi0_r + pitch)

                if phi0_initialized:
                    dot_phi0_l = angle_diff(phi0_l, last_phi0_l) / dt
                    dot_phi0_r = angle_diff(phi0_r, last_phi0_r) / dt
                else:
                    dot_phi0_l = 0.0
                    dot_phi0_r = 0.0
                    phi0_initialized = True
                last_phi0_l = phi0_l
                last_phi0_r = phi0_r

                dot_theta_ll = dot_phi0_l + dot_pitch
                dot_theta_lr = dot_phi0_r + dot_pitch

                real_state = np.array(
                    [
                        s,
                        ds,
                        yaw,
                        dot_yaw,
                        theta_ll,
                        dot_theta_ll,
                        theta_lr,
                        dot_theta_lr,
                        pitch,
                        dot_pitch,
                    ],
                    dtype=np.float64,
                )

                if data.time < 0.8:
                    data.ctrl[:] = 0
                    cached_control[:] = 0
                    mpc_step_counter = 0

                    expect_state[0] = s
                    expect_state[2] = yaw
                    expect_state[4] = 0
                    expect_state[6] = 0
                elif data.time < 1.1:
                    # expect_state = np.zeros(mpc_model.nx, dtype=np.float64)
                    p1, p4 = ik(target_L0, target_phi0, L1, L2, L3, L4, L5)

                    q_l_rear_target = wrap(math.pi - p1)
                    q_l_front_target = wrap(p4)
                    q_r_rear_target = wrap(p1 - math.pi)
                    q_r_front_target = wrap(-p4)

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
                    data.ctrl[l_wheel_ctrl] = 0
                    data.ctrl[r_wheel_ctrl] = 0
                    cached_control[:] = 0
                    mpc_step_counter = 0

                    # 跟踪结束前夕，只将当前的累积位移s和偏航yaw设为期望，使得轮子不猛烈回退。
                    # 腿摆角 theta 必须设为0以保持直立平衡。
                    expect_state[0] = s
                    expect_state[2] = yaw
                    expect_state[4] = -0.01
                    expect_state[6] = -0.01
                else:
                    left_leg_pid.target = target_L0
                    right_leg_pid.target = target_L0
                    # 添加重力前馈（安全保护的除法），G / cos(theta)
                    left_leg_force = left_leg_pid.position_pid(L0_l, dt) + (
                        mpc_model.default_leg_force / max(0.5, math.cos(theta_ll))
                    )
                    right_leg_force = right_leg_pid.position_pid(L0_r, dt) + (
                        mpc_model.default_leg_force / max(0.5, math.cos(theta_lr))
                    )

                    left_j_t = Mat_JRM(phi0_l, phi_l1, p2l, p3l, phi_l4, L0_l, L1, L4)
                    right_j_t = Mat_JRM(phi0_r, phi_r1, p2r, p3r, phi_r4, L0_r, L1, L4)

                    if mpc_step_counter == 0:
                        mpc_output = mpc_controller.solve(
                            x0=real_state,
                            x_ref=expect_state,
                            left_leg_length=L0_l,
                            right_leg_length=L0_r,
                            left_j_t=left_j_t,
                            right_j_t=right_j_t,
                            left_leg_force=left_leg_force,
                            right_leg_force=right_leg_force,
                        )
                        cached_control = mpc_output.control
                        if not mpc_output.solved:
                            print(
                                f"[MPC] status={mpc_output.status} iter={mpc_output.iterations}"
                            )

                    mpc_step_counter = (mpc_step_counter + 1) % mpc_interval_steps

                    wheel_torque_l, wheel_torque_r, hip_torque_l, hip_torque_r = (
                        cached_control
                    )
                    left_joint_torque = left_j_t @ np.array(
                        [left_leg_force, hip_torque_l],
                        dtype=np.float64,
                    )
                    right_joint_torque = right_j_t @ np.array(
                        [right_leg_force, hip_torque_r],
                        dtype=np.float64,
                    )

                    data.ctrl[l_rear_ctrl] = np.clip(-left_joint_torque[0], -60, 60)
                    data.ctrl[l_front_ctrl] = np.clip(left_joint_torque[1], -60, 60)
                    data.ctrl[r_rear_ctrl] = np.clip(right_joint_torque[0], -60, 60)
                    data.ctrl[r_front_ctrl] = np.clip(-right_joint_torque[1], -60, 60)
                    data.ctrl[l_wheel_ctrl] = np.clip(wheel_torque_l, -4.5, 4.5)
                    data.ctrl[r_wheel_ctrl] = np.clip(-wheel_torque_r, -4.5, 4.5)

                    if data.time - last_debug_print_time >= 0.1:
                        print(
                            f"[state] t={data.time:.3f} "
                            f"s={real_state[0]:.4f} ds={real_state[1]:.4f} "
                            f"phi={real_state[2]:.4f} dphi={real_state[3]:.4f} "
                            f"th_l={real_state[4]:.4f} dth_l={real_state[5]:.4f} "
                            f"th_r={real_state[6]:.4f} dth_r={real_state[7]:.4f} "
                            f"pitch={real_state[8]:.4f} dpitch={real_state[9]:.4f}"
                        )
                        print(
                            f"[ctrl] Tw_l={wheel_torque_l:.4f} Tw_r={wheel_torque_r:.4f} "
                            f"Tp_l={hip_torque_l:.4f} Tp_r={hip_torque_r:.4f} "
                            f"F0_l={left_leg_force:.4f} F0_r={right_leg_force:.4f}"
                            f"L0_l={L0_l:.4f} L0_r={L0_r:.4f}"
                        )
                        last_debug_print_time = data.time

                mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
