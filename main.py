import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

from controllers.lqr_controller import LqrController
from controllers.whole_body_lqr_controller import WholeBodyLqrController
from utils.paths import LQR_CONFIG_PATH, MODEL_PATH, ROOT, WHOLE_BODY_LQR_CONFIG_PATH

CONTROLLERS = {
    "lqr": (LqrController, LQR_CONFIG_PATH),
    "whole_body_lqr": (WholeBodyLqrController, WHOLE_BODY_LQR_CONFIG_PATH),
}
KEY_COMMANDS = {
    "0": (0.0, 0.0, 0.0),
    "1": (1.0, 0.0, 0.0),
    "2": (-1.0, 0.0, 0.0),
    "3": (0.0, -1.0, 0.0),
    "4": (0.0, 1.0, 0.0),
    "5": (0.0, 0.0, 1.0),
    "6": (0.0, 0.0, -1.0),
}


def create_controller(
    model_path: Path = MODEL_PATH,
    controller: str = "lqr",
    terrain: str = "flat",
    config: Path | None = None,
) -> LqrController | WholeBodyLqrController:
    terrain_path = (
        ROOT / "MJCF" / "terrains" / f"{terrain}.xml"
        if terrain in ("flat", "step")
        else Path(terrain)
    )
    robot = mujoco.MjSpec.from_file(str(model_path.resolve()))
    ground = mujoco.MjSpec.from_file(str(terrain_path.resolve()))
    floor = robot.geom("floor")
    if floor is not None:
        robot.delete(floor)
    robot.attach(ground, prefix="terrain/", frame=robot.worldbody.add_frame())
    cls, default_config = CONTROLLERS[controller]
    return cls(default_config if config is None else config, model=robot.compile())


def main() -> None:
    parser = argparse.ArgumentParser(description="Wheel-legged robot simulation")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Robot MJCF")
    parser.add_argument("--controller", choices=CONTROLLERS, default="lqr")
    parser.add_argument("--terrain", default="flat", help="flat, step, or terrain MJCF")
    parser.add_argument("--config", type=Path, help="Controller YAML override")
    parser.add_argument("--headless", action="store_true", help="Run without a viewer")
    parser.add_argument("--duration", type=float, default=5.0, help="Headless seconds")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    controller = create_controller(
        args.model, args.controller, args.terrain, args.config
    )
    if args.headless:
        for _ in range(math.ceil(args.duration / controller.model.opt.timestep)):
            controller.step()
        if not np.isfinite(controller.data.qpos).all() or any(
            warning.number for warning in controller.data.warning
        ):
            raise RuntimeError(
                "Simulation produced a nonfinite state or MuJoCo warning"
            )
        print(f"{args.controller}: simulated {controller.data.time:.3f} s")
    else:
        from utils.viewer import run_interactive

        run_interactive(controller, KEY_COMMANDS, args.controller)


if __name__ == "__main__":
    main()
