import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mujoco
import mujoco.viewer
import numpy as np

from demo import PATHS, STATE_NAMES
from utils import DemoLqrController, Plotter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BODY_NAME = "base_link"
DISTURBANCES = {
    "horizontal": np.array([20.0, 0.0, 0.0], dtype=float),
    "vertical": np.array([0.0, 0.0, 20.0], dtype=float),
}


@dataclass
class DisturbanceResponse:
    name: str
    time: np.ndarray
    pitch: np.ndarray
    velocity: np.ndarray
    wheel_torque: np.ndarray
    start_time: float
    end_time: float


class DisturbanceRecorder(Plotter):
    def __init__(self) -> None:
        super().__init__(STATE_NAMES, print_interval=math.inf)
        self.time: list[float] = []
        self.feedback: list[np.ndarray] = []
        self.wheel_torque: list[np.ndarray] = []

    def record(
        self,
        time: float,
        expected: np.ndarray,
        feedback: np.ndarray,
    ) -> None:
        self.time.append(time)
        self.feedback.append(feedback.copy())

    def record_wheel_torque(self, wheel_torque: np.ndarray) -> None:
        self.wheel_torque.append(wheel_torque.copy())

    def response(
        self,
        name: str,
        start_time: float,
        end_time: float,
    ) -> DisturbanceResponse:
        feedback = np.vstack(self.feedback)
        time_array = np.array(self.time)
        time_array -= time_array[0]

        return DisturbanceResponse(
            name,
            time_array,
            feedback[:, STATE_NAMES.index("pitch")],
            feedback[:, STATE_NAMES.index("ds")],
            np.vstack(self.wheel_torque),
            start_time,
            end_time,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="?", default=str(PATHS.lqr_yaml))
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--disturbance-time", type=float, default=0.5)
    parser.add_argument("--disturbance-duration", type=float, default=0.1)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--snapshot-width", type=int, default=2560)
    parser.add_argument("--snapshot-height", type=int, default=1440)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def set_disturbance(
    controller: DemoLqrController,
    body_id: int,
    force: np.ndarray,
    start_time: float,
    end_time: float,
) -> None:
    controller.data.xfrc_applied[body_id] = 0.0

    if start_time <= controller.data.time < end_time:
        controller.data.xfrc_applied[body_id, :3] = force


def record_wheel_torque(
    controller: DemoLqrController,
    recorder: DisturbanceRecorder,
) -> None:
    if len(recorder.wheel_torque) < len(recorder.time):
        recorder.record_wheel_torque(controller.wheel_torque)


def sync_viewer(
    controller: DemoLqrController,
    recorder: DisturbanceRecorder,
    viewer: mujoco.viewer.Handle,
    body_id: int,
    force: np.ndarray,
    start_time: float,
    end_time: float,
) -> None:
    step_start = time.time()
    set_disturbance(controller, body_id, force, start_time, end_time)
    controller.step()
    record_wheel_torque(controller, recorder)
    viewer.sync()

    sleep_time = controller.model.opt.timestep - (time.time() - step_start)
    if sleep_time > 0.0:
        time.sleep(sleep_time)


def save_snapshot(
    controller: DemoLqrController,
    path: Path,
    width: int,
    height: int,
) -> None:
    controller.model.vis.global_.offwidth = max(
        controller.model.vis.global_.offwidth,
        width,
    )
    controller.model.vis.global_.offheight = max(
        controller.model.vis.global_.offheight,
        height,
    )

    with mujoco.Renderer(controller.model, height=height, width=width) as renderer:
        renderer.update_scene(controller.data, camera="paper_right")
        plt.imsave(path, renderer.render())


def simulate(
    yaml_path: Path,
    name: str,
    force: np.ndarray,
    duration: float,
    disturbance_time: float,
    disturbance_duration: float,
    output_dir: Path,
    snapshot_width: int,
    snapshot_height: int,
    visualize: bool,
) -> DisturbanceResponse:
    recorder = DisturbanceRecorder()
    controller = DemoLqrController(yaml_path, recorder)
    controller.last_l0_print = math.inf

    body_id = controller.model.body(BODY_NAME).id
    end_time = disturbance_time + disturbance_duration
    snapshot_path = output_dir / f"disturbance_recovery_{name}_end.png"
    snapshot_saved = False

    if visualize:
        controller.paused = True

        def key_callback(keycode: int) -> None:
            if chr(keycode) == " ":
                controller.toggle_pause()

        print(
            f"{name} disturbance: force = {force} N, "
            f"t = [{disturbance_time:.2f}, {end_time:.2f}] s"
        )
        with mujoco.viewer.launch_passive(
            controller.model,
            controller.data,
            key_callback=key_callback,
        ) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(
                controller.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "paper_right",
            )

            while viewer.is_running() and controller.data.time < duration:
                sync_viewer(
                    controller,
                    recorder,
                    viewer,
                    body_id,
                    force,
                    disturbance_time,
                    end_time,
                )
                if not snapshot_saved and controller.data.time >= end_time:
                    save_snapshot(
                        controller,
                        snapshot_path,
                        snapshot_width,
                        snapshot_height,
                    )
                    snapshot_saved = True
    else:
        while controller.data.time < duration:
            set_disturbance(controller, body_id, force, disturbance_time, end_time)
            controller.step()
            record_wheel_torque(controller, recorder)
            if not snapshot_saved and controller.data.time >= end_time:
                save_snapshot(
                    controller,
                    snapshot_path,
                    snapshot_width,
                    snapshot_height,
                )
                snapshot_saved = True

    return recorder.response(name, disturbance_time, end_time)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def mark_disturbance(ax: plt.Axes, response: DisturbanceResponse) -> None:
    ax.axvspan(
        response.start_time,
        response.end_time,
        color="0.82",
        alpha=0.5,
        linewidth=0.0,
        label="disturbance",
    )


def plot_response(
    response: DisturbanceResponse,
    output_dir: Path,
) -> None:
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(6.2, 6.4),
        sharex=True,
        constrained_layout=True,
    )

    for ax in axes:
        mark_disturbance(ax, response)

    axes[0].plot(response.time, response.pitch, color="#2ca02c")
    axes[0].set_title("Body pitch response", pad=5, fontweight="bold")
    axes[0].set_ylabel(r"$\theta_b$ [rad]")
    axes[0].legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.15",
        framealpha=1.0,
    )

    axes[1].plot(response.time, response.velocity, color="#1f77b4")
    axes[1].set_title("Forward velocity response", pad=5, fontweight="bold")
    axes[1].set_ylabel(r"$\dot{s}$ [m s$^{-1}$]")

    axes[2].plot(response.time, response.wheel_torque[:, 0], color="#9467bd")
    axes[2].set_title("Left wheel hub torque", pad=5, fontweight="bold")
    axes[2].set_ylabel(r"$T_{wl}$ [N m]")

    axes[3].plot(response.time, response.wheel_torque[:, 1], color="#ff7f0e")
    axes[3].set_title("Right wheel hub torque", pad=5, fontweight="bold")
    axes[3].set_ylabel(r"$T_{wr}$ [N m]")
    axes[3].set_xlabel("time [s]")

    for ax in axes:
        ax.grid(True, color="0.88", linewidth=0.55)
        ax.tick_params(direction="in", top=True, right=True, width=0.8)
        ax.set_xlim(0.0, response.time[-1])

    file_stem = f"disturbance_recovery_{response.name}"
    fig.savefig(output_dir / f"{file_stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{file_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main(
    yaml_path: str,
    duration: float,
    disturbance_time: float,
    disturbance_duration: float,
    output_dir: Path,
    snapshot_width: int,
    snapshot_height: int,
    visualize: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, force in DISTURBANCES.items():
        response = simulate(
            Path(yaml_path),
            name,
            force,
            duration,
            disturbance_time,
            disturbance_duration,
            output_dir,
            snapshot_width,
            snapshot_height,
            visualize,
        )
        plot_response(response, output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.yaml,
        args.duration,
        args.disturbance_time,
        args.disturbance_duration,
        args.output_dir,
        args.snapshot_width,
        args.snapshot_height,
        args.visualize,
    )
