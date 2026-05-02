import argparse
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mujoco
import mujoco.viewer
import numpy as np

from demo import CONTROL, PATHS
from utils import DemoLqrController, Plotter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class LegLengthResponse:
    name: str
    time: np.ndarray
    target_leg_length: np.ndarray
    leg_length: np.ndarray
    leg_force: np.ndarray
    pitch: np.ndarray


class LegLengthRecorder(Plotter):
    def __init__(self) -> None:
        super().__init__(("pitch",), print_interval=math.inf)
        self.time: list[float] = []
        self.pitch: list[float] = []
        self.target_leg_length: list[float] = []
        self.leg_length: list[np.ndarray] = []
        self.leg_force: list[np.ndarray] = []

    def record(
        self,
        time: float,
        expected: np.ndarray,
        feedback: np.ndarray,
    ) -> None:
        self.time.append(time)
        self.pitch.append(float(feedback[-2]))

    def record_controller(self, controller: DemoLqrController) -> None:
        if len(self.leg_length) < len(self.time):
            self.target_leg_length.append(controller.target_l0)
            self.leg_length.append(controller.leg_length.copy())
            self.leg_force.append(controller.leg_force.copy())

    def response(self, name: str) -> LegLengthResponse:
        time_array = np.array(self.time)
        time_array -= time_array[0]

        return LegLengthResponse(
            name,
            time_array,
            np.array(self.target_leg_length),
            np.vstack(self.leg_length),
            np.vstack(self.leg_force),
            np.array(self.pitch),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="?", default=str(PATHS.lqr_yaml))
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--base-length", type=float, default=0.05)
    parser.add_argument("--step-time", type=float, default=0.5)
    parser.add_argument("--step-height", type=float, default=0.05)
    parser.add_argument("--sine-amplitude", type=float, default=0.01)
    parser.add_argument("--sine-frequency", type=float, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def step_target(
    time_value: float,
    base_length: float,
    step_time: float,
    step_height: float,
    sine_amplitude: float,
    sine_frequency: float,
) -> float:
    return base_length + (step_height if time_value >= step_time else 0.0)


def sine_target(
    time_value: float,
    base_length: float,
    step_time: float,
    step_height: float,
    sine_amplitude: float,
    sine_frequency: float,
) -> float:
    sine_time = max(time_value - step_time, 0.0)
    return base_length + sine_amplitude * math.sin(
        2.0 * math.pi * sine_frequency * sine_time
    )


def sync_viewer(
    controller: DemoLqrController,
    recorder: LegLengthRecorder,
    viewer: mujoco.viewer.Handle,
    target_l0: float,
) -> None:
    step_start = time.time()
    controller.target_l0 = target_l0
    controller.step()
    recorder.record_controller(controller)
    viewer.sync()

    sleep_time = controller.model.opt.timestep - (time.time() - step_start)
    if sleep_time > 0.0:
        time.sleep(sleep_time)


def simulate(
    yaml_path: Path,
    name: str,
    target_fn: Callable[[float, float, float, float, float, float], float],
    duration: float,
    base_length: float,
    step_time: float,
    step_height: float,
    sine_amplitude: float,
    sine_frequency: float,
    visualize: bool,
) -> LegLengthResponse:
    recorder = LegLengthRecorder()
    controller = DemoLqrController(yaml_path, recorder)
    controller.last_l0_print = math.inf
    controller.target_l0 = base_length

    if visualize:
        controller.paused = True

        def key_callback(keycode: int) -> None:
            if chr(keycode) == " ":
                controller.toggle_pause()

        print(f"{name} leg length tracking, press SPACE to start/pause")
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
                target_l0 = target_fn(
                    controller.data.time,
                    base_length,
                    step_time,
                    step_height,
                    sine_amplitude,
                    sine_frequency,
                )
                sync_viewer(controller, recorder, viewer, target_l0)
    else:
        while controller.data.time < duration:
            controller.target_l0 = target_fn(
                controller.data.time,
                base_length,
                step_time,
                step_height,
                sine_amplitude,
                sine_frequency,
            )
            controller.step()
            recorder.record_controller(controller)

    return recorder.response(name)


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


def plot_response(response: LegLengthResponse, output_dir: Path) -> None:
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
        response.time,
        response.target_leg_length,
        color="#d62728",
        linestyle="--",
        label=r"target $l_0$",
    )
    axes[0].plot(
        response.time,
        response.leg_length[:, 0],
        color="#9467bd",
        label=r"left $l_0$",
    )
    axes[0].plot(
        response.time,
        response.leg_length[:, 1],
        color="#ff7f0e",
        label=r"right $l_0$",
    )
    axes[0].set_title("Leg length tracking", pad=5, fontweight="bold")
    axes[0].set_ylabel(r"$l_0$ [m]")
    axes[0].legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.15",
        framealpha=1.0,
    )

    axes[1].plot(
        response.time,
        response.leg_force[:, 0],
        color="#9467bd",
    )
    axes[1].set_title("Left leg thrust response", pad=5, fontweight="bold")
    axes[1].set_ylabel(r"$F_{0l}$ [N]")

    axes[2].plot(
        response.time,
        response.leg_force[:, 1],
        color="#ff7f0e",
    )
    axes[2].set_title("Right leg thrust response", pad=5, fontweight="bold")
    axes[2].set_ylabel(r"$F_{0r}$ [N]")

    axes[3].plot(response.time, response.pitch, color="#2ca02c")
    axes[3].set_title("Body pitch response", pad=5, fontweight="bold")
    axes[3].set_ylabel(r"$\theta_b$ [rad]")
    axes[3].set_xlabel("time [s]")

    for ax in axes:
        ax.grid(True, color="0.88", linewidth=0.55)
        ax.tick_params(direction="in", top=True, right=True, width=0.8)
        ax.set_xlim(0.0, response.time[-1])

    fig.savefig(
        output_dir / f"leg_length_tracking_{response.name}.png",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / f"leg_length_tracking_{response.name}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main(
    yaml_path: str,
    duration: float,
    base_length: float,
    step_time: float,
    step_height: float,
    sine_amplitude: float,
    sine_frequency: float,
    output_dir: Path,
    visualize: bool,
) -> None:
    responses = (
        simulate(
            Path(yaml_path),
            "step",
            step_target,
            duration,
            base_length,
            step_time,
            step_height,
            sine_amplitude,
            sine_frequency,
            visualize,
        ),
        simulate(
            Path(yaml_path),
            "sine",
            sine_target,
            duration,
            base_length,
            step_time,
            step_height,
            sine_amplitude,
            sine_frequency,
            visualize,
        ),
    )

    for response in responses:
        plot_response(response, output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(
        args.yaml,
        args.duration,
        args.base_length,
        args.step_time,
        args.step_height,
        args.sine_amplitude,
        args.sine_frequency,
        args.output_dir,
        args.visualize,
    )
