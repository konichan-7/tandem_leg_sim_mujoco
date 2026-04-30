import argparse
import math
import os
from pathlib import Path
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml

from hang_demo_read_motor_angles import (
    LEFT_FRONT_JOINT,
    LEFT_PHI1_ECD,
    LEFT_PHI4_ECD,
    LEFT_REAR_JOINT,
    RIGHT_FRONT_JOINT,
    RIGHT_PHI1_ECD,
    RIGHT_PHI4_ECD,
    RIGHT_REAR_JOINT,
    XML_PATH,
    leg_phi,
    vmc,
)
from utils.math_tools import angle_diff, quat_to_euler, wrap
from utils.pid import PID

sys.path.append(os.path.join(os.path.dirname(__file__), "sp_lqr"))
from sjtu.lqr import compute_lqr_controller

DEFAULT_YAML_PATH = Path(__file__).parent / "configs" / "lqr.yaml"
BASE_INIT_Z = 0.25
TARGET_L0 = 0.05
TARGET_PHI0 = math.pi / 2
TARGET_VELOCITY = 0.0
TARGET_YAW = 0.0
LQR_START = 0.1
STAND_KP = 5.0
STAND_KD = 0.5
LEG_FORCE_KP = 100.0
LEG_FORCE_KD = 5.0
LEG_FORCE_LIMIT = 12.0
LEG_FORCE_INTEGRAL_LIMIT = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="?", default=str(DEFAULT_YAML_PATH))
    return parser.parse_args()


def load_params(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def qpos(model: mujoco.MjModel, data: mujoco.MjData, joint: str) -> float:
    return float(data.qpos[model.joint(joint).qposadr[0]])


def qvel(model: mujoco.MjModel, data: mujoco.MjData, joint: str) -> float:
    return float(data.qvel[model.joint(joint).dofadr[0]])


def ctrl_id(model: mujoco.MjModel, actuator: str) -> int:
    return model.actuator(actuator).id


def leg_targets(l0: float, phi0: float) -> dict[str, float]:
    phi1, phi4 = vmc.inverse_kinematics(l0, phi0)
    return {
        LEFT_FRONT_JOINT: wrap(phi1 + LEFT_PHI1_ECD),
        LEFT_REAR_JOINT: wrap(phi4 + LEFT_PHI4_ECD),
        RIGHT_FRONT_JOINT: wrap(phi1 + RIGHT_PHI1_ECD),
        RIGHT_REAR_JOINT: wrap(phi4 + RIGHT_PHI4_ECD),
    }


def apply_joint_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ctrl: dict[str, int],
    targets: dict[str, float],
    kp: float,
    kd: float,
) -> None:
    for joint, target in targets.items():
        data.ctrl[ctrl[joint]] = kp * wrap(
            target - qpos(model, data, joint)
        ) - kd * qvel(model, data, joint)


def main(yaml_path: str) -> None:
    params = load_params(yaml_path)
    wheel_radius = params["R_w"]
    gravity = params["g"]
    control_limits = params["lqr"]["control_limits"]
    K, _, _, _, _ = compute_lqr_controller(yaml_path, verbose=True)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    ctrl = {
        LEFT_FRONT_JOINT: ctrl_id(model, "left_front_motor"),
        LEFT_REAR_JOINT: ctrl_id(model, "left_rear_motor"),
        RIGHT_FRONT_JOINT: ctrl_id(model, "right_front_motor"),
        RIGHT_REAR_JOINT: ctrl_id(model, "right_rear_motor"),
        "left_wheel_joint": ctrl_id(model, "left_wheel_motor"),
        "right_wheel_joint": ctrl_id(model, "right_wheel_motor"),
    }

    stand_targets = leg_targets(TARGET_L0, TARGET_PHI0)
    data.qpos[2] = BASE_INIT_Z
    mujoco.mj_forward(model, data)

    l0_pid_l = PID(
        LEG_FORCE_KP,
        0.0,
        LEG_FORCE_KD,
        TARGET_L0,
        output_limit=LEG_FORCE_LIMIT,
        integral_limit=LEG_FORCE_INTEGRAL_LIMIT,
    )
    l0_pid_r = PID(
        LEG_FORCE_KP,
        0.0,
        LEG_FORCE_KD,
        TARGET_L0,
        output_limit=LEG_FORCE_LIMIT,
        integral_limit=LEG_FORCE_INTEGRAL_LIMIT,
    )
    half_weight = 0.5 * float(np.sum(model.body_mass[1:])) * gravity

    baselink_quat_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SENSOR, "baselink_quat"
    )

    last_phi0_l = 0.0
    last_phi0_r = 0.0
    last_pitch = 0.0
    last_yaw = 0.0
    phi0_ready = False
    pitch_ready = False
    yaw_ready = False
    s = 0.0
    paused = False

    def key_callback(keycode: int) -> None:
        nonlocal paused
        if chr(keycode) == " ":
            paused = not paused

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            if not paused:
                dtheta_l = qvel(model, data, "left_wheel_joint")
                dtheta_r = qvel(model, data, "right_wheel_joint")
                ds = 0.5 * wheel_radius * (dtheta_l + dtheta_r)
                s += ds * dt

                if data.time < LQR_START:
                    apply_joint_pd(model, data, ctrl, stand_targets, STAND_KP, STAND_KD)
                else:
                    sensor_adr = model.sensor_adr[baselink_quat_id]
                    base_quat = data.sensordata[sensor_adr : sensor_adr + 4]
                    pitch, yaw = quat_to_euler(base_quat)

                    if yaw_ready:
                        dot_yaw = angle_diff(yaw, last_yaw) / dt
                    else:
                        dot_yaw = 0.0
                        yaw_ready = True
                    last_yaw = yaw

                    if pitch_ready:
                        dot_pitch = angle_diff(pitch, last_pitch) / dt
                    else:
                        dot_pitch = 0.0
                        pitch_ready = True
                    last_pitch = pitch

                    phi = leg_phi(model, data)
                    phi_l1 = phi["left_phi1"]
                    phi_l4 = phi["left_phi4"]
                    phi_r1 = phi["right_phi1"]
                    phi_r4 = phi["right_phi4"]

                    p2l, p3l, l0_l, phi0_l = vmc.forward_kinematics(phi_l1, phi_l4)
                    p2r, p3r, l0_r, phi0_r = vmc.forward_kinematics(phi_r1, phi_r4)

                    theta_ll = wrap(-math.pi / 2 + phi0_l + pitch)
                    theta_lr = wrap(-math.pi / 2 + phi0_r + pitch)

                    print(
                        f"phi0_l: {math.degrees(phi0_l):.2f}°, phi0_r: {math.degrees(phi0_r):.2f}° | "
                        f"theta_ll: {math.degrees(theta_ll):.2f}°, theta_lr: {math.degrees(theta_lr):.2f}°"
                        f"pitch: {math.degrees(pitch):.2f}°, yaw: {math.degrees(yaw):.2f}°"
                    )

                    if phi0_ready:
                        dot_phi0_l = angle_diff(phi0_l, last_phi0_l) / dt
                        dot_phi0_r = angle_diff(phi0_r, last_phi0_r) / dt
                    else:
                        dot_phi0_l = 0.0
                        dot_phi0_r = 0.0
                        phi0_ready = True
                    last_phi0_l = phi0_l
                    last_phi0_r = phi0_r

                    real_state = np.matrix(
                        [
                            [s],
                            [ds],
                            [yaw],
                            [dot_yaw],
                            [theta_ll],
                            [dot_phi0_l + dot_pitch],
                            [theta_lr],
                            [dot_phi0_r + dot_pitch],
                            [pitch],
                            [dot_pitch],
                        ]
                    )
                    expect_state = np.matrix(
                        [
                            [0.0],
                            [TARGET_VELOCITY],
                            [TARGET_YAW],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                            [0.0],
                        ]
                    )

                    u = K * (expect_state - real_state)
                    t_l = u[0, 0].item()
                    t_r = u[1, 0].item()
                    t_pl = u[2, 0].item()
                    t_pr = u[3, 0].item()

                    l0_pid_l.target = TARGET_L0
                    l0_pid_r.target = TARGET_L0
                    f_bl = half_weight * math.cos(theta_ll) + l0_pid_l.calc(l0_l, dt)
                    f_br = half_weight * math.cos(theta_lr) + l0_pid_r.calc(l0_r, dt)

                    j_l = vmc.mat_jrm(phi0_l, phi_l1, p2l, p3l, phi_l4, l0_l)
                    j_r = vmc.mat_jrm(phi0_r, phi_r1, p2r, p3r, phi_r4, l0_r)
                    tau_l = vmc.virtual_force_to_joint_torque(j_l, f_bl, t_pl)
                    tau_r = vmc.virtual_force_to_joint_torque(j_r, f_br, t_pr)

                    data.ctrl[ctrl[LEFT_FRONT_JOINT]] = np.clip(
                        tau_l[0],
                        -control_limits["T_bl_max"],
                        control_limits["T_bl_max"],
                    )
                    data.ctrl[ctrl[LEFT_REAR_JOINT]] = np.clip(
                        tau_l[1],
                        -control_limits["T_bl_max"],
                        control_limits["T_bl_max"],
                    )
                    data.ctrl[ctrl[RIGHT_FRONT_JOINT]] = np.clip(
                        tau_r[0],
                        -control_limits["T_br_max"],
                        control_limits["T_br_max"],
                    )
                    data.ctrl[ctrl[RIGHT_REAR_JOINT]] = np.clip(
                        tau_r[1],
                        -control_limits["T_br_max"],
                        control_limits["T_br_max"],
                    )
                    data.ctrl[ctrl["left_wheel_joint"]] = np.clip(
                        t_l, -control_limits["T_wl_max"], control_limits["T_wl_max"]
                    )
                    data.ctrl[ctrl["right_wheel_joint"]] = np.clip(
                        t_r, -control_limits["T_wr_max"], control_limits["T_wr_max"]
                    )

                mujoco.mj_step(model, data)

            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    args = parse_args()
    main(args.yaml)
