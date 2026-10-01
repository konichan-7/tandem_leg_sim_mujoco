from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "MJCF" / "leg_hero.xml"
CONFIG_BASE_PATH = ROOT / "configs" / "base.yaml"
LQR_CONFIG_PATH = ROOT / "configs" / "lqr.yaml"
WHOLE_BODY_LQR_CONFIG_PATH = ROOT / "configs" / "whole_body_lqr.yaml"
