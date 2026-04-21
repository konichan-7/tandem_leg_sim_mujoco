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
from utils.vmc import VMC

# Add sp_lqr to path to import sjtu modules
sys.path.append(os.path.join(os.path.dirname(__file__), "sp_lqr"))
from sjtu.lqr import compute_lqr_controller


# Leg Kinematics (derived from leg.py)
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
    K, _, _, _, _ = compute_lqr_controller(yaml_path, verbose=True)

    target_L0 = 0.2
    target_phi0 = math.pi / 2
    kp, kd = 800.0, 100.0

    F0_l = PID(5000, 0, 1000, target_L0, output_limit=2000, integral_limit=500)
    F0_r = PID(5000, 0, 1000, target_L0, output_limit=2000, integral_limit=500)

    vmc = VMC(0.215, 0.254, 0.254, 0.215, 0.0)

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

    # Init pose
    data.qpos[2] = 0.25
    data.qpos[l_front_idx] = 1.0
    data.qpos[l_rear_idx] = -1.0
    data.qpos[r_front_idx] = -1.0
    data.qpos[r_rear_idx] = 1.0

    last_phi0_l, last_phi0_r = 0, 0
    last_pitch = 0.0
    last_yaw = 0.0
    phi0_initialized = False
    pitch_initialized = False
    yaw_initialized = False
    s = 0.0
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
                dtheta_l = data.qvel[l_wheel_dof]
                dtheta_r = -data.qvel[r_wheel_dof]
                ds = 0.5 * wheel_radius * (dtheta_l + dtheta_r)
                s += ds * dt

                if data.time < 0.8:
                    data.ctrl[:] = 0
                elif data.time < 0.95:
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
                else:
                    sensor_adr = model.sensor_adr[baselink_quat_id]
                    base_quat = data.sensordata[sensor_adr : sensor_adr + 4]

                    pitch, yaw = quat_to_euler(base_quat)
                    if not yaw_initialized:
                        last_yaw = yaw
                        yaw_initialized = True
                        dot_yaw = 0
                    else:
                        dot_yaw = angle_diff(yaw, last_yaw) / dt
                        last_yaw = yaw

                    if not pitch_initialized:
                        last_pitch = pitch
                        pitch_initialized = True
                        dot_pitch = 0
                    else:
                        dot_pitch = angle_diff(pitch, last_pitch) / dt
                        last_pitch = pitch

                    phi_l1 = wrap(math.pi - data.qpos[l_rear_idx])
                    phi_l4 = wrap(data.qpos[l_front_idx])
                    phi_r1 = wrap(math.pi + data.qpos[r_rear_idx])
                    phi_r4 = wrap(-data.qpos[r_front_idx])

                    p2l, p3l, L0_l, phi0_l = vmc.forward_kinematics(phi_l1, phi_l4)
                    p2r, p3r, L0_r, phi0_r = vmc.forward_kinematics(phi_r1, phi_r4)

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
                            [s],
                            [ds],
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
                    dF_0_l = F0_l.calc(L0_l, dt)
                    dF_0_r = F0_r.calc(L0_r, dt)
                    gravity_l = 6.5 * 9.8 * math.cos(theta_ll)
                    gravity_r = 6.5 * 9.8 * math.cos(theta_lr)
                    F_bl = gravity_l + dF_0_l
                    F_br = gravity_r + dF_0_r

                    JRM_L = vmc.mat_jrm(phi0_l, phi_l1, p2l, p3l, phi_l4, L0_l)
                    JRM_R = vmc.mat_jrm(phi0_r, phi_r1, p2r, p3r, phi_r4, L0_r)
                    T_JL = vmc.virtual_force_to_joint_torque(JRM_L, F_bl, T_pl)
                    T_JR = vmc.virtual_force_to_joint_torque(JRM_R, F_br, T_pr)
                    print(
                        f"F_bl={F_bl:.3f} F_br={F_br:.3f} T_pl={T_pl:.3f} T_pr={T_pr:.3f}"
                    )
                    print(f"JRM_L=\n{JRM_L}\nJRM_R=\n{JRM_R}")

                    data.ctrl[l_rear_ctrl] = np.clip(-T_JL[0], -60, 60)
                    data.ctrl[l_front_ctrl] = np.clip(T_JL[1], -60, 60)
                    data.ctrl[r_rear_ctrl] = np.clip(T_JR[0], -60, 60)
                    data.ctrl[r_front_ctrl] = np.clip(-T_JR[1], -60, 60)

                    data.ctrl[4] = np.clip(T_l, -4.5, 4.5)
                    data.ctrl[5] = np.clip(-T_r, -4.5, 4.5)

                    print(
                        f"[LQR dbg] t={data.time:.3f} "
                        f"L0=({L0_l:.5f},{L0_r:.5f}) "
                        f"phi0=({phi0_l:.3f},{phi0_r:.3f}) "
                        f"th=({theta_ll:.3f},{theta_lr:.3f}) "
                        f"dth=({dot_theta_ll:.3f},{dot_theta_lr:.3f}) "
                        f"pitch={pitch:.3f} dpitch={dot_pitch:.3f}"
                    )

                mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    args = parse_args()
    main(args.yaml)
