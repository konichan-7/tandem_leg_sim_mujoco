import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter, perf_counter_ns

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from controllers.lqr_controller import LqrController
from controllers.whole_body_lqr_controller import WholeBodyLqrController
from utils.control import move_towards
from utils.paths import (
    CONFIG_BASE_PATH,
    LQR_CONFIG_PATH,
    MODEL_PATH,
    ROOT,
    WHOLE_BODY_LQR_CONFIG_PATH,
)

Controller = LqrController | WholeBodyLqrController


@dataclass(frozen=True)
class Command:
    start: float
    end: float
    forward: float = 0.0
    turn: float = 0.0
    height: float = 0.0
    spin: bool = False


@dataclass(frozen=True)
class Scenario:
    name: str
    duration: float = 8.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    gimbal_rad: float = 0.0
    force: tuple[float, float, float] = (0.0, 0.0, 0.0)
    commands: tuple[Command, ...] = ()


SCENARIOS = (
    Scenario("stand"),
    Scenario("pitch_positive", pitch_deg=5.0),
    Scenario("pitch_negative", pitch_deg=-5.0),
    Scenario("roll", roll_deg=2.0),
    Scenario("push_forward", force=(50.0, 0.0, 0.0)),
    Scenario("push_down", force=(0.0, 0.0, -50.0)),
    Scenario("forward", 10.0, commands=(Command(1, 5, forward=1),)),
    Scenario(
        "reverse", 12.0, commands=(Command(1, 4, forward=1), Command(4, 7, forward=-1))
    ),
    Scenario("turn", 10.0, commands=(Command(1, 5, turn=1),)),
    Scenario("arc", 10.0, commands=(Command(1, 5, forward=1, turn=1),)),
    Scenario("height_raise", commands=(Command(1, 1.8, height=1),)),
    Scenario("height_lower", commands=(Command(1, 1.8, height=-1),)),
    Scenario(
        "height_cycle",
        10.0,
        commands=(
            Command(1, 1.8, height=1),
            Command(2.8, 4.4, height=-1),
            Command(5.4, 6.2, height=1),
        ),
    ),
    Scenario(
        "raise_forward",
        10.0,
        commands=(Command(1, 5, forward=1), Command(1, 1.8, height=1)),
    ),
    Scenario("gimbal", gimbal_rad=0.1),
    Scenario("spin", 15.0, commands=(Command(1, 5, spin=True),)),
)

COLUMNS = (
    "time_s",
    "world_x_m",
    "world_y_m",
    "ref_x_m",
    "ref_y_m",
    "world_position_error_m",
    "odometry_error_m",
    "forward_velocity_m_s",
    "ref_velocity_m_s",
    "yaw_rate_rad_s",
    "ref_yaw_rate_rad_s",
    "roll_deg",
    "pitch_deg",
    "left_leg_height_m",
    "right_leg_height_m",
    "ref_leg_height_m",
    "left_support_N",
    "right_support_N",
    "closure_mm",
    "gimbal_world_error_deg",
    "max_motor_fraction",
    "control_us",
    "step_us",
)


class Observer:
    def __init__(self, controller: Controller) -> None:
        self.controller = controller
        model, data = controller.model, controller.data
        self.base = model.body("base_link").id
        self.hips = [model.body(f"{side}_front_link").id for side in ("left", "right")]
        self.wheels = [
            model.body(f"{side}_wheel_link").id for side in ("left", "right")
        ]
        self.wheel_geoms = [
            model.geom(f"{side}_wheel_link_collision").id for side in ("left", "right")
        ]
        self.radii = model.geom_size[self.wheel_geoms, 0]
        self.pitch_body = model.body("pitch_link").id
        self.rotation_jacobian = np.empty((3, model.nv))
        self.contact_force = np.empty(6)
        self.reference_xy = data.qpos[:2].copy()
        self.reference_yaw = 0.0
        self.reference_velocity = self.reference_rate = 0.0
        self.reference_distance = self.distance = 0.0
        self.reference_height = controller.params["control"]["leg_height"]
        self.weight = model.body_mass.sum() * np.linalg.norm(model.opt.gravity)

    def reference(self, command: tuple[float, float, float, bool], dt: float) -> None:
        forward, turn, height, spin = command
        config = self.controller.command_config
        self.reference_velocity = move_towards(
            self.reference_velocity,
            0.0 if spin else forward * config["linear_velocity"],
            config["linear_acceleration"] * dt,
        )
        self.reference_rate = move_towards(
            self.reference_rate,
            config["spin_yaw_rate"] if spin else turn * config["yaw_rate"],
            config["yaw_acceleration"] * dt,
        )
        self.reference_yaw += self.reference_rate * dt
        heading = np.array([np.cos(self.reference_yaw), np.sin(self.reference_yaw)])
        self.reference_xy += self.reference_velocity * heading * dt
        self.reference_distance += self.reference_velocity * dt
        self.reference_height = np.clip(
            self.reference_height + height * config["leg_length_velocity"] * dt,
            config["leg_length_min"],
            config["leg_length_max"],
        )

    def sample(self, dt: float, control_us: float, step_us: float) -> np.ndarray:
        model, data = self.controller.model, self.controller.data
        mujoco.mj_forward(model, data)
        lateral = data.xmat[self.base].reshape(3, 3)[:, 1]
        speed = 0.0
        for radius, body in zip(self.radii, self.wheels):
            mujoco.mj_jacBody(model, data, None, self.rotation_jacobian, body)
            speed += radius * (lateral @ self.rotation_jacobian @ data.qvel) / 2
        self.distance += speed * dt
        support = np.zeros(2)
        for contact_id, contact in enumerate(data.contact):
            for side, wheel in enumerate(self.wheel_geoms):
                if wheel in contact.geom:
                    mujoco.mj_contactForce(model, data, contact_id, self.contact_force)
                    world_force = contact.frame.reshape(3, 3).T @ self.contact_force[:3]
                    support[side] += world_force[2] * (
                        1 if contact.geom[1] == wheel else -1
                    )
        heading = np.array([np.cos(self.reference_yaw), np.sin(self.reference_yaw)])
        roll, pitch, _ = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler(
            "xyz"
        )
        heights = data.xpos[self.hips, 2] - data.xpos[self.wheels, 2]
        closure = np.linalg.norm(
            data.site_xpos[model.eq_obj1id] - data.site_xpos[model.eq_obj2id], axis=1
        ).max()
        direction = data.xmat[self.pitch_body].reshape(3, 3)[:, 0]
        aim = (
            self.controller.gimbal.direction
            if self.controller.gimbal.holding
            else data.xmat[self.base].reshape(3, 3)[:, 0]
        )
        aim_error = np.rad2deg(np.arccos(np.clip(direction @ aim, -1.0, 1.0)))
        return np.concatenate(
            (
                np.array(
                    [
                        data.time,
                        *data.qpos[:2],
                        *self.reference_xy,
                        np.linalg.norm(data.qpos[:2] - self.reference_xy),
                        self.distance - self.reference_distance,
                        heading @ data.qvel[:2],
                        self.reference_velocity,
                        data.qvel[5],
                        self.reference_rate,
                        np.rad2deg(roll),
                        np.rad2deg(pitch),
                        *heights,
                        self.reference_height,
                        *support,
                        closure * 1000,
                        aim_error,
                        np.max(np.abs(data.ctrl) / self.controller.design.limits),
                        control_us,
                        step_us,
                    ]
                ),
                data.ctrl.copy(),
            )
        )


def command_at(scenario: Scenario, time: float) -> tuple[float, float, float, bool]:
    active = [
        command for command in scenario.commands if command.start <= time < command.end
    ]
    return (
        sum(command.forward for command in active),
        sum(command.turn for command in active),
        sum(command.height for command in active),
        any(command.spin for command in active),
    )


def simulate(
    controller: Controller,
    scenario: Scenario,
    output: Path,
    name: str,
    sample_every: int,
    initial_qpos: np.ndarray,
    initial_torque: np.ndarray,
) -> dict:
    controller.reset()
    model, data = controller.model, controller.data
    data.qpos[:] = initial_qpos
    data.ctrl[:] = initial_torque
    displacement = np.zeros(model.nv)
    displacement[3] = np.deg2rad(scenario.roll_deg)
    displacement[4] = np.deg2rad(scenario.pitch_deg)
    displacement[controller.gimbal.dofs] = scenario.gimbal_rad
    mujoco.mj_integratePos(model, data.qpos, displacement, 1)
    mujoco.mj_forward(model, data)
    observer = Observer(controller)
    initial = observer.sample(0.0, 0.0, 0.0)
    if (
        scenario.name == "stand"
        and abs(initial[16:18].sum() - observer.weight) > 0.01 * observer.weight
    ):
        raise ValueError("Measured wheel support does not balance gravity")
    rows = [initial]
    control_times, step_times = [], []
    maxima = np.abs(initial).copy()
    force_square_sum = 0.0
    peak_support = initial[16:18].sum() / observer.weight
    saturated_steps = 0
    failures = []
    last_command = None
    dt = model.opt.timestep
    steps = round(scenario.duration / dt)
    completed_steps = 0
    for index in range(steps):
        time = index * dt
        command = command_at(scenario, time)
        if command != last_command:
            controller.command(*command[:3], spin=command[3])
            last_command = command
        observer.reference(command, dt)
        data.xfrc_applied[observer.base, :3] = (
            scenario.force if 1.0 <= time < 1.2 else 0.0
        )
        start = perf_counter_ns()
        controller.control()
        controlled = perf_counter_ns()
        mujoco.mj_step(model, data)
        stepped = perf_counter_ns()
        control_us, step_us = (controlled - start) / 1000, (stepped - controlled) / 1000
        if not all(
            np.isfinite(values).all() for values in (data.qpos, data.qvel, data.ctrl)
        ):
            failures.append("nonfinite physical state or control")
            break
        row = observer.sample(dt, control_us, step_us)
        if not np.isfinite(row).all():
            failures.append("nonfinite observation")
            break
        completed_steps += 1
        control_times.append(control_us)
        step_times.append(step_us)
        maxima = np.maximum(maxima, np.abs(row))
        force_square_sum += (row[16:18].sum() / observer.weight - 1) ** 2
        peak_support = max(peak_support, row[16:18].sum() / observer.weight)
        saturated_steps += row[20] >= 0.999
        if index % sample_every == 0 or index == steps - 1:
            rows.append(row)
        if max(abs(row[11]), abs(row[12])) > np.rad2deg(0.25):
            failures.append("roll/pitch exceeded 0.25 rad")
        if row[18] > 1.0:
            failures.append("closure exceeded 1 mm")
        if row[20] > 1 + 1e-10:
            failures.append("motor limit exceeded")
        if any(warning.number for warning in data.warning) or not data.eq_active.all():
            failures.append("MuJoCo warning or inactive equality")
        if abs(data.time - (index + 1) * dt) > dt / 2:
            failures.append("simulation time reset")
        if failures:
            if index % sample_every:
                rows.append(row)
            break
    trace = np.stack(rows)
    trace_name = f"{scenario.name}_{name}.csv"
    columns = COLUMNS + tuple(model.actuator(i).name + "_Nm" for i in range(model.nu))
    np.savetxt(
        output / trace_name, trace, delimiter=",", header=",".join(columns), comments=""
    )
    steady = trace[(trace[:, 0] >= 3) & (trace[:, 0] < 5)]
    if completed_steps == 0:
        raise RuntimeError(f"{name}/{scenario.name}: no valid step: {failures}")
    return {
        "controller": name,
        "scenario": scenario.name,
        "status": "failed" if failures else "completed",
        "failures": failures,
        "completed_s": completed_steps * dt,
        "requested_s": scenario.duration,
        "peak_roll_deg": maxima[11].item(),
        "peak_pitch_deg": maxima[12].item(),
        "peak_closure_mm": maxima[18].item(),
        "peak_support_weight_ratio": peak_support.item(),
        "support_deviation_rms_weight_ratio": np.sqrt(
            force_square_sum / completed_steps
        ).item(),
        "final_world_position_error_m": trace[-1, 5].item(),
        "final_odometry_error_m": trace[-1, 6].item(),
        "final_leg_height_error_m": np.max(
            np.abs(trace[-1, 13:15] - trace[-1, 15])
        ).item(),
        "steady_velocity_mae_m_s": (
            np.mean(np.abs(steady[:, 7] - steady[:, 8])).item() if len(steady) else None
        ),
        "steady_yaw_rate_mae_rad_s": (
            np.mean(np.abs(steady[:, 9] - steady[:, 10])).item()
            if len(steady)
            else None
        ),
        "peak_gimbal_error_deg": maxima[19].item(),
        "peak_motor_fraction": maxima[20].item(),
        "saturated_step_fraction": saturated_steps / completed_steps,
        "control_median_us": np.median(control_times).item(),
        "control_p95_us": np.percentile(control_times, 95).item(),
        "step_median_us": np.median(step_times).item(),
        "trace": trace_name,
    }


def source_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/lqr_comparison")
    parser.add_argument(
        "--scenarios", nargs="+", choices=[case.name for case in SCENARIOS]
    )
    parser.add_argument("--sample-every", type=int, default=10)
    args = parser.parse_args()
    if args.sample_every <= 0:
        raise ValueError("sample-every must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [
        MODEL_PATH,
        CONFIG_BASE_PATH,
        LQR_CONFIG_PATH,
        WHOLE_BODY_LQR_CONFIG_PATH,
        ROOT / "compare_lqr.py",
    ]
    paths += sorted((ROOT / "controllers").glob("*.py")) + sorted(
        (ROOT / "utils").glob("*.py")
    )
    paths += sorted((ROOT / "modelling").glob("*.py"))
    paths += sorted((ROOT / "MJCF").glob("*.STL")) + sorted(
        (ROOT / "MJCF").glob("*.obj")
    )
    hashes = source_hashes(paths)
    controllers = {}
    initialization = {}
    for name, cls in (("lqr10", LqrController), ("lqr24", WholeBodyLqrController)):
        start = perf_counter()
        controllers[name] = cls()
        initialization[name] = perf_counter() - start
    ten, whole = controllers.values()
    if (
        ten.command_config != whole.command_config
        or ten.params["control"]["leg_height"] != whole.params["control"]["leg_height"]
    ):
        raise ValueError(
            "Comparison requires identical command settings and nominal height"
        )
    if not np.array_equal(ten.design.limits, whole.design.limits):
        raise ValueError("Comparison requires identical actuator limits")
    selected = [
        case
        for case in SCENARIOS
        if args.scenarios is None or case.name in args.scenarios
    ]
    report = {
        "mujoco_version": mujoco.__version__,
        "architectures": {
            name: {
                "state_count": controller.design.gain.shape[1],
                "lqr_input_count": controller.design.gain.shape[0],
                "gimbal_control": "independent_pd",
            }
            for name, controller in controllers.items()
        },
        "source_sha256": hashes,
        "initialization_s": initialization,
        "sample_every_steps": args.sample_every,
        "initial_qpos": ten.design.qpos.tolist(),
        "initial_torque": ten.design.torque.tolist(),
        "nominal_joint_difference_max_rad": np.max(
            np.abs(ten.design.qpos[7:] - whole.design.qpos[7:])
        ).item(),
        "scenarios": [asdict(case) for case in selected],
        "results": [],
    }
    for index, case in enumerate(selected):
        order = list(controllers.items())
        if index % 2:
            order.reverse()
        for name, controller in order:
            result = simulate(
                controller,
                case,
                args.output,
                name,
                args.sample_every,
                ten.design.qpos,
                ten.design.torque,
            )
            report["results"].append(result)
            print(
                f'{name}/{case.name}: {result["status"]}, pitch={result["peak_pitch_deg"]:.3f} deg, closure={result["peak_closure_mm"]:.3f} mm',
                flush=True,
            )
            (args.output / "comparison.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
    if source_hashes(paths) != hashes:
        raise RuntimeError(
            "Sources changed during comparison; rerun against a fixed version"
        )
    fields = [key for key in report["results"][0] if key != "failures"]
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in report["results"])
    print(f'Wrote {len(report["results"])} trials to {args.output}', flush=True)
    if any(row["status"] == "failed" for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
