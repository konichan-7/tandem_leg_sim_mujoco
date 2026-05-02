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


@dataclass
class VelocityTracking:
    time: np.ndarray
    target_velocity: np.ndarray
    actual_velocity: np.ndarray
    pitch: np.ndarray
    wheel_torque: np.ndarray


class VelocityRecorder(Plotter):
    def __init__(self) -> None:
        super().__init__(STATE_NAMES, print_interval=math.inf)
        self.time: list[float] = []
        self.expected: list[np.ndarray] = []
        self.feedback: list[np.ndarray] = []
        self.wheel_torque: list[np.ndarray] = []

    def record(
        self,
        time: float,
        expected: np.ndarray,
        feedback: np.ndarray,
    ) -> None:
        self.time.append(time)
        self.expected.append(expected.copy())
        self.feedback.append(feedback.copy())

    def record_wheel_torque(self, wheel_torque: np.ndarray) -> None:
        self.wheel_torque.append(wheel_torque.copy())

    def tracking(self) -> VelocityTracking:
        expected = np.vstack(self.expected)
        feedback = np.vstack(self.feedback)
        time_array = np.array(self.time)
        time_array -= time_array[0]

        return VelocityTracking(
            time_array,
            expected[:, STATE_NAMES.index("ds")],
            feedback[:, STATE_NAMES.index("ds")],
            feedback[:, STATE_NAMES.index("pitch")],
            np.vstack(self.wheel_torque),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="?", default=str(PATHS.lqr_yaml))
    parser.add_argument("--velocity", type=float, default=0.5)
    parser.add_argument("--step-time", type=float, default=0.5)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def update_target_state(
    controller: DemoLqrController,
    target_velocity: float,
    step_time: float,
) -> None:
    target_time = max(controller.data.time - step_time, 0.0)
    controller.target_s = target_velocity * target_time
    controller.target_velocity = (
        target_velocity if controller.data.time >= step_time else 0.0
    )


def sync_viewer(
    controller: DemoLqrController,
    recorder: VelocityRecorder,
    viewer: mujoco.viewer.Handle,
    target_velocity: float,
    step_time: float,
) -> None:
    step_start = time.time()
    update_target_state(controller, target_velocity, step_time)
    controller.step()
    record_wheel_torque(controller, recorder)
    viewer.sync()

    sleep_time = controller.model.opt.timestep - (time.time() - step_start)
    if sleep_time > 0.0:
        time.sleep(sleep_time)


def record_wheel_torque(
    controller: DemoLqrController,
    recorder: VelocityRecorder,
) -> None:
    if len(recorder.wheel_torque) < len(recorder.time):
        recorder.record_wheel_torque(controller.wheel_torque)


def simulate(
    yaml_path: Path,
    target_velocity: float,
    step_time: float,
    duration: float,
    visualize: bool,
) -> VelocityTracking:
    recorder = VelocityRecorder()
    controller = DemoLqrController(yaml_path, recorder)
    controller.ignore_s_error = True
    controller.last_l0_print = math.inf

    if visualize:
        controller.paused = True

        def key_callback(keycode: int) -> None:
            if chr(keycode) == " ":
                controller.toggle_pause()

        print(
            f"target velocity steps to {target_velocity:.2f} m/s at "
            f"{step_time:.2f} s, press SPACE to start/pause"
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
                    target_velocity,
                    step_time,
                )
    else:
        while controller.data.time < duration:
            update_target_state(controller, target_velocity, step_time)
            controller.step()
            record_wheel_torque(controller, recorder)

    return recorder.tracking()


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


def plot_tracking(
    tracking: VelocityTracking,
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

    axes[0].plot(
        tracking.time,
        tracking.target_velocity,
        color="#d62728",
        linestyle="--",
        label=r"target $\dot{s}$",
    )
    axes[0].plot(
        tracking.time,
        tracking.actual_velocity,
        color="#1f77b4",
        label=r"actual $\dot{s}$",
    )
    axes[0].set_title("Velocity response", pad=5, fontweight="bold")
    axes[0].set_ylabel(r"$\dot{s}$ [m s$^{-1}$]")
    axes[0].legend(
        loc="lower right",
        frameon=True,
        fancybox=False,
        edgecolor="0.15",
        framealpha=1.0,
    )

    axes[1].plot(tracking.time, tracking.pitch, color="#2ca02c")
    axes[1].set_title("Body pitch response", pad=5, fontweight="bold")
    axes[1].set_ylabel(r"$\theta_b$ [rad]")

    axes[2].plot(
        tracking.time,
        tracking.wheel_torque[:, 0],
        color="#9467bd",
    )
    axes[2].set_title("Left wheel hub torque", pad=5, fontweight="bold")
    axes[2].set_ylabel(r"$T_{wl}$ [N m]")

    axes[3].plot(
        tracking.time,
        tracking.wheel_torque[:, 1],
        color="#ff7f0e",
    )
    axes[3].set_title("Right wheel hub torque", pad=5, fontweight="bold")
    axes[3].set_ylabel(r"$T_{wr}$ [N m]")
    axes[3].set_xlabel("time [s]")

    for ax in axes:
        ax.grid(True, color="0.88", linewidth=0.55)
        ax.tick_params(direction="in", top=True, right=True, width=0.8)
        ax.set_xlim(0.0, tracking.time[-1])

    fig.savefig(output_dir / "velocity_tracking.png", bbox_inches="tight")
    fig.savefig(output_dir / "velocity_tracking.pdf", bbox_inches="tight")
    plt.close(fig)


def main(
    yaml_path: str,
    target_velocity: float,
    step_time: float,
    duration: float,
    output_dir: Path,
    visualize: bool,
) -> None:
    tracking = simulate(
        Path(yaml_path),
        target_velocity,
        step_time,
        duration,
        visualize,
    )
    plot_tracking(tracking, output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.yaml,
        args.velocity,
        args.step_time,
        args.duration,
        args.output_dir,
        args.visualize,
    )
