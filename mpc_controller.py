import argparse
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from utils.math_tools import angle_diff, quat_to_euler, wrap
from utils.pid import PID
from utils.vmc import L1, L2, L3, L4, L5, VMC

sys.path.append(os.path.join(os.path.dirname(__file__), "sp_lqr"))
from mujoco_mpc import (
    DEFAULT_EXPORT_PATH,
    MujocoMpcController,
    load_or_export_mujoco_mpc,
)


def load_wheel_radius(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)
    return params["R_w"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml")
    return parser.parse_args()


def main(yaml_path: str):
    model = mujoco.MjModel.from_xml_path("MJCF/balance_bot.xml")
    data = mujoco.MjData(model)

    wheel_radius = load_wheel_radius(yaml_path)
    mpc_model = load_or_export_mujoco_mpc(
        yaml_path=yaml_path,
        export_path=DEFAULT_EXPORT_PATH,
        sample_time=0.01,
        horizon=10,
    )
    mpc_controller = MujocoMpcController(mpc_model)
    mpc_interval_steps = max(1, int(round(mpc_model.sample_time / model.opt.timestep)))
    hip_torque_limit = mpc_model.hip_torque_limit

    target_L0 = 0.15
    target_phi0 = math.pi / 2
    kp = 800.0
    kd = 100.0
    vmc = VMC(L1, L2, L3, L4, L5)

    left_leg_pid = PID(30000, 2000, 500, target_L0, output_limit=300)
    right_leg_pid = PID(30000, 2000, 500, target_L0, output_limit=300)
    per_leg_support_force = (
        0.5 * float(np.sum(model.body_mass)) * abs(float(model.opt.gravity[2]))
    )

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
    leg_force_initialized = False

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

                # 无论是否处于MPC阶段，每一帧都计算状态，防止状态机切换时导数出现毛刺(spinverse_kinematicse)
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

                p2l, p3l, L0_l, phi0_l = vmc.forward_kinematics(phi_l1, phi_l4)
                p2r, p3r, L0_r, phi0_r = vmc.forward_kinematics(phi_r1, phi_r4)

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
                    expect_state[4] = -0.005
                    expect_state[6] = -0.005
                elif data.time < 1.1:
                    # expect_state = np.zeros(mpc_model.nx, dtype=np.float64)
                    p1, p4 = vmc.inverse_kinematics(target_L0, target_phi0)

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
                    print(
                        f"[standup] : l_front_ctrl={data.ctrl[l_front_ctrl]:.4f} l_rear_ctrl={data.ctrl[l_rear_ctrl]:.4f} r_front_ctrl={data.ctrl[r_front_ctrl]:.4f} r_rear_ctrl={data.ctrl[r_rear_ctrl]:.4f}"
                    )
                    data.ctrl[l_wheel_ctrl] = 0
                    data.ctrl[r_wheel_ctrl] = 0
                    cached_control[:] = 0
                    mpc_step_counter = 0
                    leg_force_initialized = False

                    # 跟踪结束前夕，只将当前的累积位移s和偏航yaw设为期望，使得轮子不猛烈回退。
                    # 腿摆角 theta 必须设为0以保持直立平衡。
                    expect_state[0] = s
                    expect_state[2] = yaw
                    expect_state[4] = -0.038
                    expect_state[6] = -0.038
                else:
                    left_leg_pid.target = target_L0
                    right_leg_pid.target = target_L0

                    if not leg_force_initialized:
                        left_leg_pid.clear(target_L0 - L0_l)
                        right_leg_pid.clear(target_L0 - L0_r)
                        leg_force_initialized = True

                    left_leg_force = left_leg_pid.calc(L0_l, dt) - (
                        per_leg_support_force / max(abs(math.cos(theta_ll)), 0.2)
                    )

                    right_leg_force = right_leg_pid.calc(L0_r, dt) - (
                        per_leg_support_force / max(abs(math.cos(theta_lr)), 0.2)
                    )

                    left_j_t = vmc.mat_jrm(phi0_l, phi_l1, p2l, p3l, phi_l4, L0_l)
                    right_j_t = vmc.mat_jrm(phi0_r, phi_r1, p2r, p3r, phi_r4, L0_r)

                    left_leg_joint_torque = np.clip(
                        vmc.virtual_force_to_joint_torque(
                            left_j_t, left_leg_force, 0.0
                        ),
                        -hip_torque_limit,
                        hip_torque_limit,
                    )
                    right_leg_joint_torque = np.clip(
                        vmc.virtual_force_to_joint_torque(
                            right_j_t, right_leg_force, 0.0
                        ),
                        -hip_torque_limit,
                        hip_torque_limit,
                    )
                    left_leg_force = float(
                        vmc.joint_torque_to_virtual_force(
                            left_j_t, left_leg_joint_torque
                        )[0]
                    )
                    right_leg_force = float(
                        vmc.joint_torque_to_virtual_force(
                            right_j_t, right_leg_joint_torque
                        )[0]
                    )

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

                    left_joint_torque = (
                        left_leg_joint_torque
                        + vmc.virtual_force_to_joint_torque(left_j_t, 0.0, hip_torque_l)
                    )
                    right_joint_torque = (
                        right_leg_joint_torque
                        + vmc.virtual_force_to_joint_torque(
                            right_j_t, 0.0, hip_torque_r
                        )
                    )

                    data.ctrl[l_rear_ctrl] = np.clip(
                        -left_joint_torque[0], -hip_torque_limit, hip_torque_limit
                    )
                    data.ctrl[l_front_ctrl] = np.clip(
                        left_joint_torque[1], -hip_torque_limit, hip_torque_limit
                    )
                    data.ctrl[r_rear_ctrl] = np.clip(
                        right_joint_torque[0], -hip_torque_limit, hip_torque_limit
                    )
                    data.ctrl[r_front_ctrl] = np.clip(
                        -right_joint_torque[1], -hip_torque_limit, hip_torque_limit
                    )
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
    args = parse_args()
    main(args.yaml)
