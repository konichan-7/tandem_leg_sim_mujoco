from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from tandem_mjlab.keyboard import LINEAR_ACCELERATION, YAW_ACCELERATION
from tandem_mjlab.model import MODEL_PATH
from tandem_mjlab.sim2sim import run


def cases(final: bool) -> list[dict]:
    commands = [("stand", 0.0, 0.0)]
    commands += [
        (f"drive_{v:+g}", v, 0.0)
        for v in (0.3, -0.3, 0.6, -0.6, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0)
    ]
    commands += [
        (f"spin_{w:+g}", 0.0, w) for w in (0.8, -0.8, 2.0, -2.0, 3.5, -3.5, 5.0, -5.0)
    ]
    commands += [
        (f"turn_{v:+g}_{w:+g}", v, w)
        for v in (1.0, -1.0)
        for w in (1.0, -1.0, 2.0, -2.0)
    ]
    result = [
        dict(case=name, velocity=v, yaw=w, duration=20.0, delay=0, schedule=None)
        for name, v, w in commands
    ]
    for name, schedule in (
        (
            "forward_stop_reverse",
            ((0.0, 1.0, 0.0), (5.0, 0.0, 0.0), (8.0, -1.0, 0.0), (14.0, 0.0, 0.0)),
        ),
        (
            "drive_turn_drive",
            ((0.0, 1.0, 0.0), (5.0, 1.0, 1.0), (10.0, 1.0, 0.0), (15.0, 0.0, 0.0)),
        ),
        ("spin_reverse_stop", ((0.0, 0.0, 2.0), (5.0, 0.0, -2.0), (10.0, 0.0, 0.0))),
    ):
        result.append(
            dict(
                case=name,
                velocity=0.0,
                yaw=0.0,
                duration=20.0,
                delay=0,
                schedule=schedule,
            )
        )
    if final:
        result.append(
            dict(
                case="long_stand",
                velocity=0.0,
                yaw=0.0,
                duration=60.0,
                delay=0,
                schedule=None,
            )
        )
        for v in (1.0, -1.0, 2.0, -2.0):
            result.append(
                dict(
                    case=f"long_drive_{v:+g}",
                    velocity=v,
                    yaw=0.0,
                    duration=60.0,
                    delay=0,
                    schedule=None,
                )
            )
        for v, w in ((1.0, 0.0), (-1.0, 0.0), (0.0, 2.0), (1.0, 1.0), (-1.0, -1.0)):
            result.append(
                dict(
                    case=f"delay20ms_{v:+g}_{w:+g}",
                    velocity=v,
                    yaw=w,
                    duration=20.0,
                    delay=1,
                    schedule=None,
                )
            )
    return result


def evaluate_case(arguments: tuple[Path, Path, dict]) -> dict:
    policy, output, case = arguments
    csv = output / f"{case['case']}.csv"
    result = run(
        policy,
        case["duration"],
        case["velocity"],
        case["yaw"],
        csv,
        ramp=True,
        linear_acceleration=LINEAR_ACCELERATION,
        yaw_acceleration=YAW_ACCELERATION,
        command_schedule=case["schedule"],
        action_delay_frames=case["delay"],
    )
    data = np.atleast_1d(np.genfromtxt(csv, delimiter=",", names=True))
    result.update(
        case=case["case"],
        action_delay_s=case["delay"] * 0.02,
        finite=all(np.isfinite(data[field]).all() for field in data.dtype.names),
    )
    post = data[data["time"] >= 5.0]
    result["post5_yaw_mae_rps"] = (
        float(np.abs(post["yaw_rate"] - post["command_yaw_rps"]).mean())
        if len(post)
        else None
    )
    result["post5_speed_mae_mps"] = (
        float(np.abs(post["forward_velocity"] - post["command_velocity_mps"]).mean())
        if len(post)
        else None
    )
    result["max_heading_error_deg"] = float(
        np.degrees(np.abs(data["heading_error"]).max())
    )
    result["final_heading_error_deg"] = float(np.degrees(data["heading_error"][-1]))
    result["lateral_path_ratio"] = result["max_world_lateral_drift_m"] / max(
        result["path_distance_m"], 1e-9
    )
    target = case["velocity"] or case["yaw"]
    measured = data["forward_velocity"] if case["velocity"] else data["yaw_rate"]
    crossings = np.flatnonzero(measured * np.sign(target) >= abs(target) * 0.9)
    result["time_to_90pct_s"] = (
        float(data["time"][crossings[0]]) if target and len(crossings) else None
    )
    result["final_speed_mps"] = float(data["instant_velocity"][-1])
    result["final_yaw_rate_rps"] = float(data["yaw_rate"][-1])
    if case["case"] in ("stand", "long_stand"):
        result["post5_leg_joint_speed_rms_rps"] = float(
            np.sqrt(np.mean(post["leg_joint_speed_rms"] ** 2))
        )
        result["post5_leg_action_step_rms"] = float(
            np.sqrt(np.mean(post["leg_action_step_rms"] ** 2))
        )
        result["post5_height_std_m"] = float(post["height"].std())
    return result


def evaluate_process(arguments: tuple[Path, Path, dict]) -> dict:
    policy, output, case = arguments
    command = [
        sys.executable,
        "-m",
        "tandem_mjlab.function.locomotion.evaluate",
        str(policy),
        "--output",
        str(output),
        "--case",
        case["case"],
        "--final",
    ]
    with (output / f"{case['case']}.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((output / f"{case['case']}.json").read_text())


def summarize(results: list[dict], baseline: dict | None) -> dict:
    drive = [
        r
        for r in results
        if r["case"].startswith("drive_") and r["case"] != "drive_turn_drive"
    ]
    failed = [
        r["case"]
        for r in results
        if r["termination_reason"] != "duration"
        or not r["finite"]
        or r["both_wheels_contact_fraction"] < 0.98
    ]
    heading_failed = [
        r["case"]
        for r in drive
        if r["max_heading_error_deg"] > 2.0 or r["lateral_path_ratio"] > 0.02
    ]
    turning = [r for r in results if r["case"].startswith(("spin_", "turn_"))]
    yaw_failed = [
        r["case"]
        for r in turning
        if r["post5_yaw_mae_rps"] is None
        or r["post5_yaw_mae_rps"] > max(0.1, abs(r["command_yaw_rps"]) * 0.1)
    ]
    speed_mae = float(np.mean([r["mean_speed_error_mps"] for r in drive]))
    speed_failed = []
    if baseline is not None:
        previous = {r["case"]: r for r in baseline["results"]}
        baseline_mae = np.mean(
            [previous[r["case"]]["mean_speed_error_mps"] for r in drive]
        )
        if speed_mae > baseline_mae * 1.1:
            speed_failed.append("mean_drive_speed_mae")
        for result in drive:
            old = previous[result["case"]]["time_to_90pct_s"]
            new = result["time_to_90pct_s"]
            if new is None or new > old * 1.1:
                speed_failed.append(result["case"])
    long_failed = [
        r["case"]
        for r in results
        if r["case"].startswith("long_drive_")
        and (
            r["max_heading_error_deg"] > 2.0
            or r["lateral_path_ratio"] > 0.02
            or (
                baseline is not None
                and r["post5_speed_mae_mps"]
                > previous[r["case"].removeprefix("long_")]["post5_speed_mae_mps"] * 1.1
            )
        )
    ]
    stop_failed = [
        r["case"]
        for r in results
        if r["case"]
        in ("forward_stop_reverse", "drive_turn_drive", "spin_reverse_stop")
        and (abs(r["final_speed_mps"]) > 0.05 or abs(r["final_yaw_rate_rps"]) > 0.1)
    ]
    stand_failed = [
        r["case"]
        for r in results
        if r["case"] in ("stand", "long_stand")
        and (
            r["max_planar_displacement_m"] > 0.02
            or r["post5_speed_mae_mps"] > 0.01
            or r["post5_yaw_mae_rps"] > 0.05
            or r["post5_leg_joint_speed_rms_rps"] > 0.1
            or r["post5_leg_action_step_rms"] > 0.01
            or r["post5_height_std_m"] > 0.001
        )
    ]
    sequence_failed = [
        r["case"]
        for r in results
        if r["case"] in ("forward_stop_reverse", "drive_turn_drive")
        and r["max_heading_error_deg"] > 2.0
    ]
    delay_failed = []
    for result in results:
        if not result["case"].startswith("delay20ms_"):
            continue
        if result["command_yaw_rps"]:
            failure = result["post5_yaw_mae_rps"] > max(
                0.1, abs(result["command_yaw_rps"]) * 0.1
            )
        else:
            failure = (
                result["max_heading_error_deg"] > 2.0
                or result["lateral_path_ratio"] > 0.02
            )
        if baseline is not None and result["command_velocity_mps"]:
            v, w = result["command_velocity_mps"], result["command_yaw_rps"]
            reference = previous[f"turn_{v:+g}_{w:+g}" if w else f"drive_{v:+g}"]
            failure |= (
                result["mean_speed_error_mps"] > reference["mean_speed_error_mps"] * 1.1
                or result["time_to_90pct_s"] is None
                or result["time_to_90pct_s"] > reference["time_to_90pct_s"] * 1.1
            )
        if failure:
            delay_failed.append(result["case"])
    return dict(
        failed_stability_cases=failed,
        failed_heading_cases=heading_failed,
        failed_yaw_cases=yaw_failed,
        failed_speed_regressions=speed_failed,
        failed_long_cases=long_failed,
        failed_stop_cases=stop_failed,
        failed_stand_cases=stand_failed,
        failed_sequence_cases=sequence_failed,
        failed_delay_cases=delay_failed,
        mean_drive_speed_mae_mps=speed_mae,
        worst_drive_lateral_m=max(r["max_world_lateral_drift_m"] for r in drive),
        max_tilt_deg=max(r["max_tilt_deg"] for r in results),
        min_both_wheels_contact_fraction=min(
            r["both_wheels_contact_fraction"] for r in results
        ),
        accepted=not any(
            (
                failed,
                heading_failed,
                yaw_failed,
                speed_failed,
                long_failed,
                stop_failed,
                stand_failed,
                sequence_failed,
                delay_failed,
            )
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "policy",
        type=Path,
        nargs="?",
        default=Path(__file__).parents[2] / "policies/locomotion.onnx",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.case is not None:
        case = next(case for case in cases(args.final) if case["case"] == args.case)
        result = evaluate_case((args.policy, args.output, case))
        (args.output / f"{args.case}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        return
    args.output.mkdir(parents=True)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    report = dict(
        policy=str(args.policy),
        policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
        linear_acceleration_mps2=LINEAR_ACCELERATION,
        yaw_acceleration_rps2=YAW_ACCELERATION,
        results=[],
    )
    selected = cases(args.final)
    if args.quick:
        selected = [
            case
            for case in selected
            if case["case"]
            in (
                "stand",
                "drive_+1",
                "drive_-1",
                "drive_+2",
                "drive_-2",
                "spin_+2",
                "spin_-2",
                "turn_+1_+1",
            )
        ]
    arguments = [(args.policy, args.output, case) for case in selected]
    with ThreadPoolExecutor(args.workers) as executor:
        for result in executor.map(evaluate_process, arguments):
            report["results"].append(result)
            (args.output / "results.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                result["case"],
                result["termination_reason"],
                result["max_heading_error_deg"],
                flush=True,
            )
    report["summary"] = summarize(report["results"], baseline)
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
