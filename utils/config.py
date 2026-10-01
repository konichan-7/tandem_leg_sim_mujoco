from pathlib import Path

import yaml

from utils.paths import CONFIG_BASE_PATH

NUMBER = (int, float)

SCHEMA = {
    "control": {"leg_height": NUMBER, "motor_limits": dict},
    "command": {
        "linear_velocity": NUMBER,
        "yaw_rate": NUMBER,
        "spin_yaw_rate": NUMBER,
        "linear_acceleration": NUMBER,
        "yaw_acceleration": NUMBER,
        "leg_length_velocity": NUMBER,
        "leg_length_min": NUMBER,
        "leg_length_max": NUMBER,
    },
    "jump": {
        "linear_velocity": NUMBER,
        "leg_length_min": NUMBER,
        "leg_length_max": NUMBER,
        "crouch_velocity": NUMBER,
        "extension_velocity": NUMBER,
        "retraction_velocity": NUMBER,
        "landing_time": NUMBER,
        "landing_leg_kp": NUMBER,
        "landing_leg_kd": NUMBER,
        "air_hold_time": NUMBER,
        "thrust_gravity_scale": NUMBER,
        "momentum_gain": NUMBER,
        "air_joint_kp": NUMBER,
        "air_joint_kd": NUMBER,
        "air_attitude_kp": NUMBER,
        "air_attitude_kd": NUMBER,
        "air_posture_weight": NUMBER,
        "air_leg_kp": NUMBER,
        "air_leg_kd": NUMBER,
        "air_leg_weight": NUMBER,
        "air_gimbal_kp": NUMBER,
        "air_gimbal_kd": NUMBER,
        "liftoff_time": NUMBER,
        "liftoff_force_ratio": NUMBER,
        "landing_force_ratio": NUMBER,
        "ready_height_error": NUMBER,
        "tuck_height_error": NUMBER,
        "ready_speed": NUMBER,
        "ready_angular_speed": NUMBER,
        "ready_time": NUMBER,
    },
    "leg": {
        "kp": NUMBER,
        "ki": NUMBER,
        "kd": NUMBER,
        "integral_limit": NUMBER,
        "force_limit": NUMBER,
    },
    "gimbal": {"kp": NUMBER, "kd": NUMBER},
    "lqr": {
        "state_limits": dict,
        "control_limits": dict,
        "Q_weights": dict,
        "landing_Q_weights": dict,
        "R_weights": dict,
    },
    "modelling": {"finite_difference_step": NUMBER},
}


def read_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return config


def merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def validate(config: dict) -> None:
    for section, fields in config.items():
        known = SCHEMA.get(section)
        if known is None:
            raise ValueError(f"Unknown config section: {section}")
        if not isinstance(fields, dict):
            raise ValueError(f"Config section must be a mapping: {section}")
        for name, value in fields.items():
            expected = known.get(name)
            if expected is None:
                raise ValueError(f"Unknown config key: {section}.{name}")
            if isinstance(value, bool) or not isinstance(value, expected):
                raise ValueError(f"Wrong config type: {section}.{name} = {value!r}")


def load_config(path: Path) -> dict:
    config = merge(read_config(CONFIG_BASE_PATH), read_config(path))
    validate(config)
    return config
