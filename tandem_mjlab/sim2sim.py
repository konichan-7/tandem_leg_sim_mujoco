from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import deque
from contextlib import nullcontext
from itertools import count
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

from tandem_mjlab.keyboard import (
    KEYBOARD_SPEED,
    KEYBOARD_YAW_RATE,
    LINEAR_ACCELERATION,
    YAW_ACCELERATION,
    KeyboardCommand,
    ramp_command,
)
from tandem_mjlab.model import (
    ACTUATED_JOINTS,
    DECIMATION,
    HISTORY_LENGTH,
    OBSERVATION_DIM,
    LEG_ACTION_SCALE,
    LEG_DAMPING,
    LEG_LENGTH_BODIES,
    LEG_STIFFNESS,
    LINEAR_COMMAND_RANGE,
    LOOPS,
    PHYSICS_TIMESTEP,
    POLICY_PERIOD,
    WHEEL_ACTION_SCALE,
    WHEEL_DAMPING,
    YAW_COMMAND_RANGE,
    get_spec,
    apply_simulation_options,
    initial_joint_positions,
    motor_limits,
    standing_pose,
    wheel_action_limit,
)

REFERENCE_HEIGHT = standing_pose()[1]


def observe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    position_ids: np.ndarray,
    velocity_ids: np.ndarray,
    default_positions: np.ndarray,
    last_action: np.ndarray,
    velocity_command: float,
    yaw_command: float,
    heading_error: float,
) -> np.ndarray:
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    rotation = data.xmat[base_id].reshape(3, 3)
    gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
    command = np.array(
        [velocity_command * 2.0, yaw_command * 0.25, REFERENCE_HEIGHT * 5.0]
    )
    observation = np.concatenate(
        (
            data.sensor("gyro").data * 0.25,
            gravity_body,
            command,
            data.qpos[position_ids[:4]] - default_positions[:4],
            data.qvel[velocity_ids] * 0.05,
            last_action,
        )
    ).astype(np.float32)
    return np.r_[observation, np.sin(heading_error), np.cos(heading_error)].astype(
        np.float32
    )


def ground_contact_bodies(
    model: mujoco.MjModel, data: mujoco.MjData, floor_id: int
) -> set[int]:
    bodies = set()
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if contact.geom1 == floor_id:
            bodies.add(int(model.geom_bodyid[contact.geom2]))
        elif contact.geom2 == floor_id:
            bodies.add(int(model.geom_bodyid[contact.geom1]))
    return bodies


def run(
    policy_path: Path,
    duration: float | None,
    velocity_command: float,
    yaw_command: float,
    csv_path: Path,
    view: bool = False,
    trace_path: Path | None = None,
    keyboard: bool = False,
    keyboard_speed: float = KEYBOARD_SPEED,
    keyboard_yaw_rate: float = KEYBOARD_YAW_RATE,
    linear_acceleration: float = LINEAR_ACCELERATION,
    yaw_acceleration: float = YAW_ACCELERATION,
    ramp: bool = False,
    command_schedule: tuple[tuple[float, float, float], ...] | None = None,
    action_delay_frames: int = 0,
) -> dict[str, float | str]:
    if keyboard and sys.platform != "darwin":
        raise ValueError("键盘控制使用仓库的 macOS 输入，请在本机运行。")
    if keyboard and not (
        0.0 <= keyboard_speed <= LINEAR_COMMAND_RANGE[1]
        and 0.0 <= keyboard_yaw_rate <= YAW_COMMAND_RANGE[1]
    ):
        raise ValueError(
            f"键盘速度必须在训练范围内：0–{LINEAR_COMMAND_RANGE[1]} m/s，"
            f"0–{YAW_COMMAND_RANGE[1]} rad/s。"
        )
    if (keyboard or ramp) and (linear_acceleration <= 0.0 or yaw_acceleration <= 0.0):
        raise ValueError("指令斜坡的线加速度和角加速度必须大于零。")
    keyboard_input = KeyboardCommand() if keyboard else None
    if keyboard:
        view = True
        velocity_command = yaw_command = 0.0
        print("按住 1/2 前进/后退，3/4 右转/左转；松开后减速归零，可组合按键。")
    target_velocity, target_yaw = velocity_command, yaw_command
    if ramp:
        velocity_command = yaw_command = 0.0

    model = get_spec().compile()
    apply_simulation_options(model)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    defaults = initial_joint_positions()
    for name, position in defaults.items():
        data.qpos[
            model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        ] = position
    data.qpos[:7] = (0.0, 0.0, REFERENCE_HEIGHT, 1.0, 0.0, 0.0, 0.0)
    mujoco.mj_forward(model, data)
    floor_id = model.geom("floor").id
    base_id = model.body("base_link").id
    leg_body_ids = [model.body(name).id for name in LEG_LENGTH_BODIES]
    wheel_body_ids = {
        model.body("left_wheel_link").id,
        model.body("right_wheel_link").id,
    }

    joint_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in ACTUATED_JOINTS
        ]
    )
    position_ids = model.jnt_qposadr[joint_ids]
    velocity_ids = model.jnt_dofadr[joint_ids]
    default_positions = np.array([defaults[name] for name in ACTUATED_JOINTS])
    stand_effort = np.array([standing_pose()[2][name] for name in ACTUATED_JOINTS[:4]])
    effort_limits = np.array([motor_limits()[name] for name in ACTUATED_JOINTS])
    wheel_limit = wheel_action_limit()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(policy_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if [item.shape[-1] for item in session.get_inputs()] != [
        OBSERVATION_DIM,
        OBSERVATION_DIM * HISTORY_LENGTH,
    ]:
        raise ValueError("Policy must use the release observation protocol (27/135)")
    reference_heading = 0.0
    was_straight = True
    pending_actions = deque(
        (np.zeros(6) for _ in range(action_delay_frames + 1)),
        maxlen=action_delay_frames + 1,
    )
    last_action = np.zeros(6, dtype=np.float32)
    first_observation = observe(
        model,
        data,
        position_ids,
        velocity_ids,
        default_positions,
        last_action,
        velocity_command,
        yaw_command,
        0.0,
    )
    history = deque(
        (first_observation.copy() for _ in range(HISTORY_LENGTH)), maxlen=HISTORY_LENGTH
    )
    samples: list[dict[str, float]] = []
    trace: list[dict[str, float]] = []
    previous_position = data.qpos[:3].copy()
    path_distance = 0.0
    wheel_contact_substeps = {body_id: 0 for body_id in wheel_body_ids}
    both_wheel_substeps = 0
    any_wheel_substeps = 0
    total_substeps = 0
    termination_reason = "duration"
    invalid_contact_body = ""
    wheel_air_time = {body_id: 0.0 for body_id in wheel_body_ids}
    max_wheel_air_time = 0.0

    viewer_context = (
        mujoco.viewer.launch_passive(model, data) if view else nullcontext()
    )
    steps = count() if duration is None else range(round(duration / POLICY_PERIOD))
    with viewer_context as viewer:
        for _ in steps:
            step_start = time.monotonic()
            if command_schedule is not None:
                for start, velocity, yaw in command_schedule:
                    if data.time + 1e-9 >= start:
                        target_velocity, target_yaw = velocity, yaw
            if ramp and keyboard_input is None:
                velocity_command = ramp_command(
                    velocity_command,
                    target_velocity,
                    linear_acceleration,
                    POLICY_PERIOD,
                )
                yaw_command = ramp_command(
                    yaw_command, target_yaw, yaw_acceleration, POLICY_PERIOD
                )
            if keyboard_input is not None:
                linear_direction, yaw_direction = keyboard_input.direction()
                velocity_command = ramp_command(
                    velocity_command,
                    linear_direction * keyboard_speed,
                    linear_acceleration,
                    POLICY_PERIOD,
                )
                yaw_command = ramp_command(
                    yaw_command,
                    yaw_direction * keyboard_yaw_rate,
                    yaw_acceleration,
                    POLICY_PERIOD,
                )
            heading = float(np.arctan2(data.xmat[base_id, 3], data.xmat[base_id, 0]))
            straight = abs(yaw_command) < 1e-6
            if not straight or not was_straight:
                reference_heading = heading
            heading_error = float(
                np.arctan2(
                    np.sin(heading - reference_heading),
                    np.cos(heading - reference_heading),
                )
            )
            was_straight = straight
            observation = observe(
                model,
                data,
                position_ids,
                velocity_ids,
                default_positions,
                last_action,
                velocity_command,
                yaw_command,
                heading_error,
            )
            history.append(observation)
            previous_action = last_action
            last_action = session.run(
                ["actions"],
                {
                    "obs": observation[None, :],
                    "obs_history": np.concatenate(history)[None, :],
                },
            )[0][0]
            action = last_action.clip(
                (-0.5,) * 4 + (-wheel_limit,) * 2, (0.5,) * 4 + (wheel_limit,) * 2
            )
            pending_actions.append(action)
            action = pending_actions[0]
            leg_speed_squared = 0.0
            for _ in range(DECIMATION):
                leg_speed_squared += np.square(data.qvel[velocity_ids[:4]]).mean()
                leg_effort = (
                    stand_effort
                    + LEG_STIFFNESS
                    * (
                        default_positions[:4]
                        + LEG_ACTION_SCALE * action[:4]
                        - data.qpos[position_ids[:4]]
                    )
                    - LEG_DAMPING * data.qvel[velocity_ids[:4]]
                )
                wheel_effort = WHEEL_DAMPING * (
                    WHEEL_ACTION_SCALE * action[4:] - data.qvel[velocity_ids[4:]]
                )
                data.ctrl[:] = np.clip(
                    np.r_[leg_effort, wheel_effort], -effort_limits, effort_limits
                )
                if trace_path is not None:
                    trace.append(
                        {
                            "time": float(data.time),
                            **{
                                f"position_{index}": float(value)
                                for index, value in enumerate(data.qpos[position_ids])
                            },
                            **{
                                f"action_{index}": float(value)
                                for index, value in enumerate(action)
                            },
                            **{
                                f"effort_{index}": float(value)
                                for index, value in enumerate(data.ctrl)
                            },
                            **{
                                f"velocity_{index}": float(value)
                                for index, value in enumerate(data.qvel[velocity_ids])
                            },
                        }
                    )
                mujoco.mj_step(model, data)
                total_substeps += 1
                contact_bodies = ground_contact_bodies(model, data, floor_id)
                invalid_bodies = contact_bodies - wheel_body_ids
                if invalid_bodies:
                    invalid_contact_body = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, next(iter(invalid_bodies))
                    )
                    termination_reason = "nonwheel_ground_contact"
                    break
                for body_id in wheel_body_ids:
                    wheel_contact_substeps[body_id] += body_id in contact_bodies
                    wheel_air_time[body_id] = (
                        0.0
                        if body_id in contact_bodies
                        else wheel_air_time[body_id] + PHYSICS_TIMESTEP
                    )
                max_wheel_air_time = max(max_wheel_air_time, *wheel_air_time.values())
                if max(wheel_air_time.values()) > 5 * POLICY_PERIOD:
                    termination_reason = "wheel_airborne"
                    break
                both_wheel_substeps += wheel_body_ids.issubset(contact_bodies)
                any_wheel_substeps += bool(contact_bodies & wheel_body_ids)

            mujoco.mj_forward(model, data)
            rotation = data.xmat[base_id].reshape(3, 3)
            gravity_body = rotation.T @ np.array([0.0, 0.0, -1.0])
            leg_positions = data.xpos[leg_body_ids]
            hips = leg_positions[:4].reshape(2, 2, 3).mean(axis=1)
            leg_offsets = (leg_positions[4:] - hips) @ rotation
            leg_lengths = np.linalg.norm(leg_offsets[:, [0, 2]], axis=1)
            displacement = data.qpos[:3] - previous_position
            forward_velocity = (rotation.T @ (displacement / POLICY_PERIOD))[0]
            path_distance += float(np.linalg.norm(displacement[:2]))
            previous_position = data.qpos[:3].copy()
            loop_gap = max(
                np.linalg.norm(data.site(first).xpos - data.site(second).xpos)
                for first, second in LOOPS
            )
            samples.append(
                {
                    "time": float(data.time),
                    "command_velocity_mps": velocity_command,
                    "command_yaw_rps": yaw_command,
                    "x": float(data.qpos[0]),
                    "y": float(data.qpos[1]),
                    "yaw": float(np.arctan2(rotation[1, 0], rotation[0, 0])),
                    "height": float(data.qpos[2]),
                    "left_leg_length_m": float(leg_lengths[0]),
                    "right_leg_length_m": float(leg_lengths[1]),
                    "tilt": float(np.arccos(np.clip(-gravity_body[2], -1.0, 1.0))),
                    "roll_tilt": float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0))),
                    "forward_velocity": float(forward_velocity),
                    "vertical_velocity": float(data.qvel[2]),
                    "instant_velocity": float((rotation.T @ data.qvel[:3])[0]),
                    "lateral_velocity": float((rotation.T @ data.qvel[:3])[1]),
                    "yaw_rate": float(data.sensor("gyro").data[2]),
                    "heading_error": heading_error,
                    "leg_joint_speed_rms": float(
                        np.sqrt(leg_speed_squared / DECIMATION)
                    ),
                    "leg_action_step_rms": float(
                        np.sqrt(np.square(last_action[:4] - previous_action[:4]).mean())
                    ),
                    "loop_gap": float(loop_gap),
                    "wheel_contacts": float(len(contact_bodies & wheel_body_ids)),
                    "nonwheel_contacts": float(len(invalid_bodies)),
                }
            )
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, POLICY_PERIOD - (time.monotonic() - step_start)))
                if not viewer.is_running():
                    termination_reason = "viewer_closed"
                    break
            if termination_reason != "duration":
                break
            if samples[-1]["height"] < REFERENCE_HEIGHT - 0.015:
                termination_reason = "low_body"
                break
            if samples[-1]["tilt"] > np.pi / 3:
                termination_reason = "tilt"
                break

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=samples[0])
        writer.writeheader()
        writer.writerows(samples)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=trace[0])
            writer.writeheader()
            writer.writerows(trace)

    return {
        "policy": str(policy_path),
        "max_world_lateral_drift_m": max(abs(s["y"]) for s in samples),
        "final_speed_error_mps": abs(
            samples[-1]["instant_velocity"] - velocity_command
        ),
        "command_velocity_mps": velocity_command,
        "command_yaw_rps": yaw_command,
        "command_mode": "keyboard" if keyboard else "ramp" if ramp else "constant",
        "duration_s": float(samples[-1]["time"]),
        "distance_m": float(samples[-1]["x"]),
        "world_y_m": samples[-1]["y"],
        "path_distance_m": path_distance,
        "mean_forward_velocity_mps": float(
            np.mean([s["forward_velocity"] for s in samples])
        ),
        "mean_speed_error_mps": float(
            np.mean(
                [
                    abs(s["forward_velocity"] - s["command_velocity_mps"])
                    for s in samples
                ]
            )
        ),
        "max_tilt_deg": float(np.degrees(max(s["tilt"] for s in samples))),
        "mean_abs_roll_deg": float(
            np.degrees(np.mean([abs(s["roll_tilt"]) for s in samples]))
        ),
        "max_abs_roll_deg": float(
            np.degrees(max(abs(s["roll_tilt"]) for s in samples))
        ),
        "mean_yaw_rate_rps": float(np.mean([s["yaw_rate"] for s in samples])),
        "mean_yaw_error_rps": float(
            np.mean([abs(s["yaw_rate"] - s["command_yaw_rps"]) for s in samples])
        ),
        "mean_planar_speed_mps": float(
            np.mean(
                [
                    np.hypot(s["instant_velocity"], s["lateral_velocity"])
                    for s in samples
                ]
            )
        ),
        "max_planar_displacement_m": max(np.hypot(s["x"], s["y"]) for s in samples),
        "max_loop_gap_mm": float(1000 * max(s["loop_gap"] for s in samples)),
        "left_wheel_contact_fraction": wheel_contact_substeps[
            model.body("left_wheel_link").id
        ]
        / total_substeps,
        "right_wheel_contact_fraction": wheel_contact_substeps[
            model.body("right_wheel_link").id
        ]
        / total_substeps,
        "both_wheels_contact_fraction": both_wheel_substeps / total_substeps,
        "any_wheel_contact_fraction": any_wheel_substeps / total_substeps,
        "max_wheel_air_time_s": max_wheel_air_time,
        "termination_reason": termination_reason,
        "invalid_contact_body": invalid_contact_body,
        "trajectory": str(csv_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "policy",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("policies") / "locomotion.onnx",
    )
    parser.add_argument("--duration", type=float, help="可选评估时长（秒），默认不限时")
    parser.add_argument("--velocity", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/sim2sim_trajectory.csv"),
    )
    parser.add_argument("--view", action="store_true")
    parser.add_argument(
        "--ramp", action="store_true", help="固定目标使用键盘相同的指令斜坡"
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="启用 macOS 按住控制，自动打开 MuJoCo 窗口",
    )
    parser.add_argument("--keyboard-speed", type=float, default=KEYBOARD_SPEED)
    parser.add_argument("--keyboard-yaw-rate", type=float, default=KEYBOARD_YAW_RATE)
    parser.add_argument(
        "--linear-acceleration",
        type=float,
        default=LINEAR_ACCELERATION,
        help="键盘指令线加速度上限（m/s²）",
    )
    parser.add_argument(
        "--yaw-acceleration",
        type=float,
        default=YAW_ACCELERATION,
        help="键盘指令角加速度上限（rad/s²）",
    )
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.policy,
                args.duration,
                args.velocity,
                args.yaw,
                args.csv,
                args.view,
                args.trace,
                args.keyboard,
                args.keyboard_speed,
                args.keyboard_yaw_rate,
                args.linear_acceleration,
                args.yaw_acceleration,
                args.ramp,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
