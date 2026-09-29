from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class DemoPaths:
    xml: Path = ROOT / "MJCF" / "leg_hero.xml"
    lqr_yaml: Path = ROOT / "configs" / "lqr.yaml"


@dataclass(frozen=True)
class DemoControl:
    viewer_fps: float = 60.0


PATHS = DemoPaths()
CONTROL = DemoControl()
KEY_COMMANDS = {
    "0": (0.0, 0.0, 0.0),
    "1": (1.0, 0.0, 0.0),
    "2": (-1.0, 0.0, 0.0),
    "3": (0.0, -1.0, 0.0),
    "4": (0.0, 1.0, 0.0),
    "5": (0.0, 0.0, 1.0),
    "6": (0.0, 0.0, -1.0),
}
