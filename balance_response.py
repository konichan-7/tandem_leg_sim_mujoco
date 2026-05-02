import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import mujoco
import mujoco.viewer
import numpy as np

from demo import JOINTS, PATHS, STATE_NAMES
from utils import DemoLqrController, Plotter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PITCH_ERRORS = (0.05, 0.1, 0.15)
PLOT_STATES = ("pitch", "theta_ll", "theta_lr", "ds")


@dataclass
class BalanceResponse:
    time: np.ndarray
    state: np.ndarray


class BalanceRecorder(Plotter):
    def __init__(self) -> None:
        super().__init__(STATE_NAMES, print_interval=math.inf)
        self.time: list[float] = []
        self.state: list[np.ndarray] = []

    def record(
        self,
        time: float,
        expected: np.ndarray,
        feedback: np.ndarray,
    ) -> None:
        self.time.append(time)
        self.state.append(feedback.copy())

    def response(self) -> BalanceResponse:
        return BalanceResponse(np.array(self.time), np.vstack(self.state))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="?", default=str(PATHS.lqr_yaml))
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def initialize_pose(controller: DemoLqrController, pitch_error: float) -> None:
    for joint in JOINTS.leg:
        controller.data.qpos[controller.model.joint(joint).qposadr[0]] = pitch_error

    mujoco.mj_forward(controller.model, controller.data)


def sync_viewer(
    controller: DemoLqrController,
    viewer: mujoco.viewer.Handle,
) -> None:
    step_start = time.time()
    controller.step()
    viewer.sync()

    sleep_time = controller.model.opt.timestep - (time.time() - step_start)
    if sleep_time > 0.0:
        time.sleep(sleep_time)


def simulate(
    yaml_path: Path,
    duration: float,
    pitch_error: float,
    visualize: bool,
) -> BalanceResponse:
    recorder = BalanceRecorder()
    controller = DemoLqrController(yaml_path, recorder)
    controller.last_l0_print = math.inf
    initialize_pose(controller, pitch_error)

    if visualize:
        controller.paused = True

        def key_callback(keycode: int) -> None:
            if chr(keycode) == " ":
                controller.toggle_pause()

        print(f"初始关节角={pitch_error:.2f} rad，按 SPACE 开始/暂停。")
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
                sync_viewer(controller, viewer)
    else:
        while controller.data.time < duration:
            controller.step()

    response = recorder.response()
    response.time -= response.time[0]
    return response


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


def plot_responses(
    responses: dict[float, BalanceResponse],
    output_dir: Path,
) -> None:
    setup_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    state_ids = {name: STATE_NAMES.index(name) for name in PLOT_STATES}
    colors = ("#d62728", "#2ca02c", "#1f77b4")
    labels = {
        "pitch": r"$\theta_b$ [rad]",
        "theta_ll": r"$\theta_{ll}$ [rad]",
        "theta_lr": r"$\theta_{lr}$ [rad]",
        "ds": r"$\dot{s}$ [m s$^{-1}$]",
    }
    titles = {
        "pitch": "Body pitch response",
        "theta_ll": "Left leg angle response",
        "theta_lr": "Right leg angle response",
        "ds": "Longitudinal velocity response",
    }

    fig, axes = plt.subplots(
        len(PLOT_STATES),
        1,
        figsize=(6.2, 6.0),
        sharex=True,
        constrained_layout=True,
    )

    for ax, state_name in zip(axes, PLOT_STATES):
        for color, (pitch_error, response) in zip(colors, responses.items()):
            ax.plot(
                response.time,
                response.state[:, state_ids[state_name]],
                color=color,
                label=rf"initial pitch error = {pitch_error:.2f} rad",
            )

        ax.set_title(titles[state_name], pad=5, fontweight="bold")
        ax.set_ylabel(labels[state_name])
        ax.grid(True, color="0.88", linewidth=0.55)
        ax.tick_params(direction="in", top=True, right=True, width=0.8)
        ax.set_xlim(0.0, max(response.time[-1] for response in responses.values()))

    axes[-1].set_xlabel("time [s]")
    axes[0].legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.15",
        framealpha=1.0,
    )

    fig.savefig(output_dir / "balance_response.png", bbox_inches="tight")
    fig.savefig(output_dir / "balance_response.pdf", bbox_inches="tight")
    plt.close(fig)


def main(
    yaml_path: str,
    duration: float,
    output_dir: Path,
    visualize: bool,
) -> None:
    responses = {
        pitch_error: simulate(Path(yaml_path), duration, pitch_error, visualize)
        for pitch_error in PITCH_ERRORS
    }
    plot_responses(responses, output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args.yaml, args.duration, args.output_dir, args.visualize)
