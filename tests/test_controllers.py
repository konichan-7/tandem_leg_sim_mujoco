import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from experiment.compare_lqr import Observer
from controllers.whole_body_lqr_controller import WholeBodyLqrController
from main import create_controller


class WholeBodyLqrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.controller = WholeBodyLqrController()

    def setUp(self) -> None:
        self.controller.reset()

    def test_equilibrium_contact_force_and_discrete_stability(self) -> None:
        controller = self.controller
        design = controller.design
        if design.gain.shape != (6, 24):
            raise ValueError("Wrong whole-body feedback dimension")
        radius = np.max(np.abs(np.linalg.eigvals(design.a - design.b @ design.gain)))
        if radius >= 1 or abs(radius - design.spectral_radius) > 1e-10:
            raise ValueError("Discrete stability check failed")
        observer = Observer(controller)
        row = observer.sample(0.0, 0.0, 0.0)
        if abs(row[16:18].sum() - observer.weight) > 0.001 * observer.weight:
            raise ValueError("Wheel contact forces do not balance gravity")
        if np.linalg.norm(controller.data.qacc, ord=np.inf) > 1e-5:
            raise ValueError("Standing pose is not a dynamic equilibrium")

    def test_pitch_recovery_preserves_closed_chains(self) -> None:
        controller = self.controller
        controller.reset(np.deg2rad(5.0))
        for _ in range(6000):
            controller.step()
            if not np.isfinite(controller.data.qpos).all():
                raise ValueError("Nonfinite state")
            if np.any(np.abs(controller.data.ctrl) > controller.design.limits + 1e-10):
                raise ValueError("Motor limits exceeded")
        model, data = controller.model, controller.data
        mujoco.mj_forward(model, data)
        attitude = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler("xyz")
        closure = np.linalg.norm(
            data.site_xpos[model.eq_obj1id] - data.site_xpos[model.eq_obj2id], axis=1
        ).max()
        if np.max(np.abs(attitude[:2])) > 0.005 or closure > 0.001:
            raise ValueError("Pitch disturbance did not recover within closure limits")
        if not data.eq_active.all() or any(warning.number for warning in data.warning):
            raise ValueError("Constraint disabled or MuJoCo warning")

    def test_gimbal_command_is_independent_of_chassis_gain(self) -> None:
        controller = self.controller
        gain = controller.design.gain.copy()
        outputs = []
        try:
            for enabled in (True, False):
                controller.reset(np.deg2rad(1))
                controller.data.qpos[controller.gimbal.qpos] += [0.01, -0.02]
                mujoco.mj_forward(controller.model, controller.data)
                controller.design.gain[:] = gain if enabled else 0
                controller.control()
                outputs.append(controller.data.ctrl.copy())
        finally:
            controller.design.gain[:] = gain
        cloud, chassis = controller.gimbal.actuators, controller.chassis_actuators
        if not np.array_equal(outputs[0][cloud], outputs[1][cloud]):
            raise ValueError("Chassis feedback changed the independent gimbal command")
        if np.allclose(outputs[0][chassis], outputs[1][chassis]):
            raise ValueError("Chassis feedback did not affect chassis actuators")


class EntryPointTests(unittest.TestCase):
    def test_controller_and_terrain_combinations(self) -> None:
        for name in ("lqr", "whole_body_lqr"):
            for terrain in ("flat", "step"):
                with self.subTest(controller=name, terrain=terrain):
                    controller = create_controller(controller=name, terrain=terrain)
                    model, data = controller.model, controller.data
                    floors = np.count_nonzero(
                        model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
                    )
                    if floors != 1 or model.geom("terrain/floor").group[0] != 1:
                        raise ValueError(
                            "Terrain composition duplicated or lost the floor"
                        )
                    if terrain == "step" and model.geom("terrain/step").size[2] != 0.1:
                        raise ValueError("Wrong step height")
                    for _ in range(1000):
                        controller.step()
                    if np.linalg.norm(data.qvel[:6]) > 0.01 or any(
                        warning.number for warning in data.warning
                    ):
                        raise ValueError("Composed scene did not remain standing")


if __name__ == "__main__":
    unittest.main()
