import unittest

from utils.config import load_config, merge, read_config, validate
from utils.paths import CONFIG_BASE_PATH, LQR_CONFIG_PATH, WHOLE_BODY_LQR_CONFIG_PATH

OVERRIDES = (LQR_CONFIG_PATH, WHOLE_BODY_LQR_CONFIG_PATH)


class ConfigTests(unittest.TestCase):
    def test_base_file_is_valid(self) -> None:
        validate(read_config(CONFIG_BASE_PATH))

    def test_deep_merge_replaces_leaves_and_keeps_siblings(self) -> None:
        merged = merge(
            {"jump": {"a": 1.0, "b": 2.0}, "lqr": {"q": 5.0}},
            {"jump": {"b": 3.0}, "extra": {"c": 4.0}},
        )
        expected = {
            "jump": {"a": 1.0, "b": 3.0},
            "lqr": {"q": 5.0},
            "extra": {"c": 4.0},
        }
        if merged != expected:
            raise ValueError(f"Deep merge produced {merged}")

    def test_override_files_only_carry_differences(self) -> None:
        base = read_config(CONFIG_BASE_PATH)
        for path in OVERRIDES:
            override = read_config(path)
            for section, fields in override.items():
                shared = base.get(section, {})
                for name, value in fields.items():
                    if name in shared and shared[name] == value:
                        raise ValueError(
                            f"{path.name} restates the base value {section}.{name}; "
                            "delete it or change it"
                        )

    def test_merged_configs_keep_the_shared_sections(self) -> None:
        base = read_config(CONFIG_BASE_PATH)
        for path in OVERRIDES:
            config = load_config(path)
            for section in ("control", "command", "gimbal"):
                if config[section] != base[section]:
                    raise ValueError(
                        f"{path.name} changed the shared section {section}"
                    )
            for name, value in base["jump"].items():
                if config["jump"][name] != value:
                    raise ValueError(f"{path.name} dropped the shared jump.{name}")

    def test_branches_carry_their_own_sections(self) -> None:
        ten, whole = (load_config(path) for path in OVERRIDES)
        if "leg" not in ten or "modelling" in ten:
            raise ValueError("The 10-state config must hold only the VMC leg gains")
        if "modelling" not in whole or "leg" in whole:
            raise ValueError("The 24-state config must hold only the modelling flags")
        if "ready_time" not in ten["jump"] or "ready_time" in whole["jump"]:
            raise ValueError("ready_time belongs to the 10-state jump branch")
        if "air_gimbal_kp" not in whole["jump"] or "air_gimbal_kp" in ten["jump"]:
            raise ValueError("air_gimbal_kp belongs to the 24-state jump branch")

    def test_unknown_keys_and_types_are_rejected(self) -> None:
        rejected = (
            {"bogus": {"kp": 1.0}},
            {"jump": {"ready_speeed": 0.05}},
            {"jump": {"ready_speed": "0.05"}},
            {"control": 0.24},
        )
        for config in rejected:
            try:
                validate(config)
            except ValueError:
                continue
            raise ValueError(f"Config was accepted: {config}")


if __name__ == "__main__":
    unittest.main()
