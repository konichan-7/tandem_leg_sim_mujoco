import argparse
from pathlib import Path

from demo import CONTROL, KEY_COMMANDS, PATHS
from utils import DemoLqrController
from utils.viewer import run_interactive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml", nargs="?", type=Path, default=PATHS.lqr_yaml)
    return parser.parse_args()


def main(yaml_path: Path) -> None:
    controller = DemoLqrController(yaml_path)
    run_interactive(controller, KEY_COMMANDS, CONTROL.viewer_fps, "LQR Controller")


if __name__ == "__main__":
    args = parse_args()
    main(args.yaml)
