import argparse
from pathlib import Path
import time

import mujoco.viewer

from demo import PATHS, STATE_NAMES
from utils import DemoLqrController, Plotter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="?", default=str(PATHS.lqr_yaml))
    return parser.parse_args()


def main(yaml_path: str) -> None:
    plotter = Plotter(STATE_NAMES)
    controller = DemoLqrController(Path(yaml_path), plotter)

    def key_callback(keycode: int) -> None:
        if chr(keycode) == " ":
            controller.toggle_pause()

    with mujoco.viewer.launch_passive(
        controller.model,
        controller.data,
        key_callback=key_callback,
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()
            controller.step()
            viewer.sync()

            sleep_time = controller.model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    plotter.render()


if __name__ == "__main__":
    args = parse_args()
    main(args.yaml)
