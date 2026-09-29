import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from demo import PATHS
from utils import DemoLqrController


@dataclass(frozen=True)
class Scenario:
    name: str
    pitch: float = 0.0
    roll: float = 0.0
    push: float = 0.0
    speed: float = 0.0
    yaw_rate: float = 0.0
    gimbal: float = 0.0
    leg_direction: float = 0.0
    reverse: bool = False


SCENARIOS = (
    Scenario("stand"),
    Scenario("pitch_positive", pitch=5.0),
    Scenario("pitch_negative", pitch=-5.0),
    Scenario("roll", roll=2.0),
    Scenario("push", push=50.0),
    Scenario("forward", speed=0.25),
    Scenario("backward", speed=-0.25),
    Scenario("turn", yaw_rate=0.35),
    Scenario("arc", speed=0.2, yaw_rate=0.35),
    Scenario("gimbal", gimbal=0.1),
)


def verify(controller: DemoLqrController, scenario: Scenario) -> dict:
    controller.reset(np.deg2rad(scenario.pitch))
    model, data = controller.model, controller.data
    gimbal_dofs = np.array(
        [model.joint(f"{name}_joint").dofadr[0] for name in ("yaw", "pitch")]
    )
    gimbal_qpos = np.array(
        [model.joint(f"{name}_joint").qposadr[0] for name in ("yaw", "pitch")]
    )
    displacement = np.zeros(model.nv)
    displacement[3] = np.deg2rad(scenario.roll)
    displacement[gimbal_dofs] = scenario.gimbal
    mujoco.mj_integratePos(model, data.qpos, displacement, 1)
    base = model.body("base_link").id
    peak_tilt = np.zeros(2)
    peak_torque = np.zeros(model.nu)
    peak_closure = 0.0
    tracking = []
    motion = []
    last_command_time = 4.0 if scenario.reverse else 1.0
    command_end = last_command_time + 7.0
    duration = command_end + 7.0
    steps = round(duration / model.opt.timestep)
    for index in range(steps):
        active_command = 1 < data.time < command_end
        direction = -1 if scenario.reverse and data.time >= last_command_time else 1
        desired_speed = scenario.speed * direction
        desired_yaw_rate = scenario.yaw_rate * direction
        controller.command(
            (
                desired_speed / controller.command_config["linear_velocity"]
                if active_command
                else 0.0
            ),
            (
                desired_yaw_rate / controller.command_config["yaw_rate"]
                if active_command
                else 0.0
            ),
            scenario.leg_direction if active_command else 0.0,
        )
        data.xfrc_applied[base, 0] = scenario.push if 2 < data.time < 2.2 else 0.0
        controller.step()
        peak_torque = np.maximum(peak_torque, np.abs(data.ctrl))
        if index % 10:
            continue
        mujoco.mj_forward(model, data)
        rpy = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler("xyz")
        peak_tilt = np.maximum(peak_tilt, np.abs(rpy[:2]))
        closure = np.linalg.norm(
            data.site_xpos[model.eq_obj1id] - data.site_xpos[model.eq_obj2id], axis=1
        ).max()
        peak_closure = max(peak_closure, closure)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f"{scenario.name}: nonfinite state")
        if any(warning.number for warning in data.warning):
            raise RuntimeError(f"{scenario.name}: MuJoCo warning")
        if np.max(np.abs(rpy[:2])) > 0.25 or closure > 0.001:
            raise RuntimeError(
                f"{scenario.name}: lost balance or excessive closure error"
            )
        heading = np.array(
            [np.cos(controller.target_yaw), np.sin(controller.target_yaw)]
        )
        forward_speed = heading @ data.qvel[:2]
        motion.append([data.time, forward_speed, data.qvel[5]])
        if last_command_time + 3 < data.time < command_end - 1:
            tracking.append(
                [
                    forward_speed - desired_speed,
                    data.qvel[5] - desired_yaw_rate,
                ]
            )

    mujoco.mj_forward(model, data)
    rpy = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler("xyz")
    heading_error = np.arctan2(
        np.sin(rpy[2] - controller.target_yaw),
        np.cos(rpy[2] - controller.target_yaw),
    )
    forward_error = np.array(
        [np.cos(controller.target_yaw), np.sin(controller.target_yaw)]
    ) @ (data.qpos[:2] - controller.target_xy)
    tracking_error = np.mean(np.abs(tracking), axis=0)
    wheel_geoms = {
        model.geom(f"{side}_wheel_link_collision").id for side in ("left", "right")
    }
    contacting = {geom for contact in data.contact for geom in contact.geom}
    if abs(data.time - duration) > 1e-8 or not data.eq_active.all():
        raise RuntimeError(f"{scenario.name}: simulation reset or inactive loop")
    if not wheel_geoms.issubset(contacting):
        raise RuntimeError(f"{scenario.name}: wheel ground contact lost")
    if (
        np.max(np.abs(rpy[:2])) > 0.005
        or abs(heading_error) > 0.01
        or np.linalg.norm(data.qvel[:6]) > 0.01
        or abs(forward_error) > 0.01
        or np.max(np.abs(data.qpos[gimbal_qpos])) > 0.001
    ):
        raise RuntimeError(
            f"{scenario.name}: stopped reference error: position={forward_error:.4f} m, "
            f"heading={heading_error:.4f} rad, attitude={rpy[:2]}, "
            f"velocity={data.qvel[:6]}, gimbal={data.qpos[gimbal_qpos]}"
        )
    if np.any(tracking_error > 0.03):
        raise RuntimeError(f"{scenario.name}: velocity tracking error={tracking_error}")
    if scenario.speed and not scenario.yaw_rate and peak_tilt[0] > np.deg2rad(0.5):
        raise RuntimeError(f"{scenario.name}: straight-line roll exceeds 0.5 degrees")
    if np.any(peak_torque > controller.design.limits + 1e-12):
        raise RuntimeError(f"{scenario.name}: motor limit exceeded")
    leg_heights = np.array(
        [
            data.xpos[model.body(f"{side}_front_link").id, 2]
            - data.xpos[model.body(f"{side}_wheel_link").id, 2]
            for side in ("left", "right")
        ]
    )
    if np.max(np.abs(leg_heights - controller.target_l0)) > 0.002:
        raise RuntimeError(f"{scenario.name}: leg height tracking error")
    result = {
        "scenario": scenario.name,
        "duration_s": duration,
        "peak_roll_deg": np.rad2deg(peak_tilt[0]).item(),
        "peak_pitch_deg": np.rad2deg(peak_tilt[1]).item(),
        "peak_closure_mm": (peak_closure * 1000).item(),
        "final_forward_error_m": forward_error.item(),
        "final_heading_error_rad": heading_error.item(),
        "speed_mae_m_s": tracking_error[0].item(),
        "yaw_rate_mae_rad_s": tracking_error[1].item(),
        "peak_motor_fraction": np.max(peak_torque / controller.design.limits).item(),
        "final_leg_height_error_mm": (
            np.max(np.abs(leg_heights - controller.target_l0)) * 1000
        ).item(),
    }
    if (scenario.speed or scenario.yaw_rate) and not scenario.reverse:
        samples = np.array(motion)
        channel, target = (
            (1, scenario.speed) if scenario.speed else (2, scenario.yaw_rate)
        )
        running = samples[(samples[:, 0] >= 1) & (samples[:, 0] < command_end)]
        stopped = samples[samples[:, 0] >= command_end]
        reached = np.flatnonzero(running[:, channel] / target >= 0.9)
        unsettled = np.flatnonzero(np.abs(stopped[:, channel] / target) > 0.1)
        if not reached.size or not unsettled.size or unsettled[-1] + 1 == len(stopped):
            raise RuntimeError(f"{scenario.name}: response did not settle")
        result["rise_to_90_s"] = (running[reached[0], 0] - 1).item()
        result["stop_to_10_s"] = (stopped[unsettled[-1] + 1, 0] - command_end).item()
        result["overshoot_pct"] = (
            100 * (np.max(running[:, channel] / target) - 1)
        ).item()
    print(
        f"{scenario.name:16s} PASS  tilt={np.rad2deg(peak_tilt)} deg  "
        f"closure={peak_closure * 1000:.3f} mm",
        flush=True,
    )
    return result


def verify_spin(controller: DemoLqrController, initial_yaw: float) -> dict:
    controller.reset()
    model, data, gimbal = controller.model, controller.data, controller.gimbal
    quaternion = np.array([np.cos(initial_yaw / 2), 0, 0, np.sin(initial_yaw / 2)])
    mujoco.mju_mulQuat(data.qpos[3:7], quaternion, controller.design.qpos[3:7])
    controller.target_yaw = initial_yaw
    mujoco.mj_forward(model, data)
    wheel_geoms = {
        model.geom(f"{side}_wheel_link_collision").id for side in ("left", "right")
    }
    peak_aim = peak_closure = peak_wheel_gap = 0.0
    peak_release_aim = 0.0
    floor = model.geom("floor").id
    peak_tilt = np.zeros(2)
    tracking = []
    duration = 28.0
    for index in range(round(duration / model.opt.timestep)):
        spin = 1 <= data.time < 9 or 12 <= data.time < 19
        controller.command(1.0 if spin else 0.0, -1.0 if spin else 0.0, 0.0, spin)
        if spin and (
            controller.desired_velocity != 0.0
            or controller.desired_yaw_rate != controller.command_config["spin_yaw_rate"]
        ):
            raise RuntimeError("spin: Shift command priority failed")
        controller.step()
        if np.any(np.abs(data.ctrl) > controller.design.limits + 1e-12):
            raise RuntimeError("spin: motor limit exceeded")
        if index % 10:
            continue
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("spin: nonfinite state")
        if any(warning.number for warning in data.warning) or not data.eq_active.all():
            raise RuntimeError("spin: MuJoCo warning or inactive loop")
        wheel_gap = max(
            mujoco.mj_geomDistance(model, data, wheel, floor, 0.1, None)
            for wheel in wheel_geoms
        )
        peak_wheel_gap = max(peak_wheel_gap, wheel_gap)
        tilt = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler("xyz")[:2]
        peak_tilt = np.maximum(peak_tilt, np.abs(tilt))
        closure = np.linalg.norm(
            data.site_xpos[model.eq_obj1id] - data.site_xpos[model.eq_obj2id], axis=1
        ).max()
        peak_closure = max(peak_closure, closure)
        if gimbal.holding:
            direction = data.xmat[gimbal.pitch_body].reshape(3, 3)[:, 0]
            aim = np.arccos(np.clip(direction @ gimbal.direction, -1.0, 1.0))
            peak_aim = max(peak_aim, aim)
            if not spin:
                peak_release_aim = max(peak_release_aim, aim)
        if 4 <= data.time < 9 or 15 <= data.time < 19:
            tracking.append(data.qvel[5] - controller.command_config["spin_yaw_rate"])

    delta = data.qpos[gimbal.qpos] - gimbal.home
    final_gimbal = np.arctan2(np.sin(delta), np.cos(delta))
    tracking_error = np.mean(np.abs(tracking))
    heading = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler("xyz")[2]
    heading_error = heading - np.arctan2(gimbal.direction[1], gimbal.direction[0])
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
    if peak_aim > np.deg2rad(1.0) or tracking_error > 0.05:
        raise RuntimeError(
            f"spin: pointing={np.rad2deg(peak_aim)}, rate={tracking_error}"
        )
    if np.max(peak_tilt) > 0.25 or peak_closure > 0.001 or peak_wheel_gap > 0.001:
        raise RuntimeError("spin: lost balance or excessive closure error")
    contacting = {geom for contact in data.contact for geom in contact.geom}
    if not wheel_geoms.issubset(contacting):
        raise RuntimeError("spin: stopped wheel ground contact lost")
    if (
        abs(data.time - duration) > 1e-8
        or not gimbal.holding
        or controller.returning
        or abs(heading_error) > 0.001
        or np.max(np.abs(final_gimbal)) > 0.001
        or np.linalg.norm(data.qvel[:6]) > 0.01
    ):
        raise RuntimeError("spin: release failed to settle")
    result = {
        "scenario": f"spin_yaw_{initial_yaw}",
        "duration_s": duration,
        "peak_aim_error_deg": np.rad2deg(peak_aim).item(),
        "peak_release_aim_error_deg": np.rad2deg(peak_release_aim).item(),
        "final_heading_error_deg": np.rad2deg(heading_error).item(),
        "yaw_rate_mae_rad_s": tracking_error.item(),
        "peak_roll_deg": np.rad2deg(peak_tilt[0]).item(),
        "peak_pitch_deg": np.rad2deg(peak_tilt[1]).item(),
        "peak_closure_mm": (peak_closure * 1000).item(),
        "peak_wheel_gap_mm": peak_wheel_gap * 1000,
    }
    print(f"{result['scenario']} PASS {result}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="?", type=Path, default=PATHS.lqr_yaml)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    controller = DemoLqrController(args.yaml)
    speed = controller.command_config["linear_velocity"]
    yaw_rate = controller.command_config["yaw_rate"]
    scenarios = SCENARIOS + (
        Scenario("command_forward", speed=speed),
        Scenario("command_backward", speed=-speed),
        Scenario("command_turn", yaw_rate=yaw_rate),
        Scenario("command_arc", speed=speed, yaw_rate=yaw_rate),
        Scenario("reverse_forward", speed=speed, reverse=True),
        Scenario("reverse_turn", yaw_rate=yaw_rate, reverse=True),
        Scenario("leg_raise", leg_direction=1),
        Scenario("leg_lower", leg_direction=-1),
        Scenario("raise_forward", speed=speed, leg_direction=1),
        Scenario("lower_forward", speed=speed, leg_direction=-1),
    )
    results = [verify(controller, scenario) for scenario in scenarios]
    results.extend(verify_spin(controller, yaw) for yaw in (0.0, 2.8))
    report = {
        "mujoco_version": mujoco.__version__,
        "model_sha256": sha256(PATHS.xml.read_bytes()).hexdigest(),
        "config_sha256": sha256(args.yaml.read_bytes()).hexdigest(),
        "spectral_radius": controller.design.spectral_radius,
        "equilibrium_residual": controller.design.equilibrium_residual,
        "gain": controller.design.gain.tolist(),
        "equilibrium_qpos": controller.design.qpos.tolist(),
        "feedforward_torque": controller.design.torque.tolist(),
        "position_dofs": controller.design.position_dofs.tolist(),
        "velocity_dofs": controller.design.velocity_dofs.tolist(),
        "actuators": [
            controller.model.actuator(i).name for i in range(controller.model.nu)
        ],
        "results": results,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Passed {len(results)} scenarios; spectral radius={controller.design.spectral_radius:.8f}"
    )


if __name__ == "__main__":
    main()
