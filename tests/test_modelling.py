from pathlib import Path
from scipy.optimize import least_squares
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from main import create_controller
from modelling.base import planar_equality_jacobian
from modelling.kinematics import Leg
from modelling.lqr_design import design_lqr, design_whole_body_lqr
from modelling.vmc_modelling import VmcModelling
from modelling.whole_body_modelling import WholeBodyModelling
from utils.paths import MODEL_PATH, LQR_CONFIG_PATH, WHOLE_BODY_LQR_CONFIG_PATH


class ModellingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        cls.whole = WholeBodyModelling(cls.model)
        cls.vmc = VmcModelling(cls.model)
        cls.whole_config = yaml.safe_load(WHOLE_BODY_LQR_CONFIG_PATH.read_text())
        cls.vmc_config = yaml.safe_load(LQR_CONFIG_PATH.read_text())

    def test_open_model_partition_and_closed_loop_prediction(self) -> None:
        design = design_whole_body_lqr(self.whole, self.whole_config)
        plant, model = design.plant, self.model
        if plant.b.shape != (24, 6) or plant.b_gimbal.shape != (24, 2):
            raise ValueError("Wrong chassis/gimbal input partition")
        if np.intersect1d(plant.chassis_actuators, plant.gimbal_actuators).size:
            raise ValueError("Chassis and gimbal share a motor")
        data = mujoco.MjData(model)
        data.qpos[:] = plant.point.qpos
        data.ctrl[:] = plant.point.torque
        mujoco.mj_forward(model, data)
        jacobian = planar_equality_jacobian(model, data)
        if np.linalg.norm(jacobian @ plant.lift[: model.nv, :11], ord=np.inf) > 1e-8:
            raise ValueError("Position lift violates planar loop closure")
        mujoco.mj_step(model, data)
        nominal_qpos, nominal_qvel = data.qpos.copy(), data.qvel.copy()
        rng = np.random.default_rng(7)
        state = rng.normal(size=24) * 1e-7
        effort = rng.normal(size=6) * 1e-7
        full = plant.lift @ state
        mujoco.mj_resetData(model, data)
        data.qpos[:] = plant.point.qpos
        mujoco.mj_integratePos(model, data.qpos, full[: model.nv], 1)
        data.qvel[:] = full[model.nv :]
        data.ctrl[:] = plant.point.torque
        data.ctrl[plant.chassis_actuators] += effort
        data.ctrl[plant.gimbal_actuators] -= (
            self.whole_config["gimbal"]["kp"] * state[plant.gimbal_position_columns]
            + self.whole_config["gimbal"]["kd"] * state[plant.gimbal_velocity_columns]
        )
        mujoco.mj_step(model, data)
        difference = np.empty(model.nv)
        mujoco.mj_differentiatePos(model, difference, 1, nominal_qpos, data.qpos)
        actual = np.r_[difference, data.qvel - nominal_qvel][plant.rows]
        predicted = design.a @ state + design.b @ effort
        if np.max(np.abs(actual - predicted)) > 2e-8:
            raise ValueError(
                f"PD-closed model disagrees with physical step: {np.max(np.abs(actual-predicted))}"
            )

    def test_changed_mjcf_and_reordered_actuators(self) -> None:
        xml = ET.parse(MODEL_PATH)
        xml.getroot().find("compiler").set("meshdir", str(MODEL_PATH.parent))
        inertia = xml.getroot().find(".//body[@name='pitch_link']/inertial")
        inertia.set("mass", str(float(inertia.get("mass")) * 1.2))
        actuators = xml.getroot().find("actuator")
        actuators[:] = list(reversed(list(actuators)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "同构机器人.xml"
            xml.write(path, encoding="unicode")
            model = mujoco.MjModel.from_xml_path(str(path))
            terrain_path = Path(directory) / "地形.xml"
            terrain_path.write_text(
                '<mujoco><worldbody><geom name="floor" type="plane" '
                'size="5 5 0.1" group="1" contype="2" conaffinity="1" '
                'friction="0.8 0.005 0.0001" condim="3" solref="0.005 1"/>'
                "</worldbody></mujoco>",
                encoding="utf-8",
            )
            for name in ("lqr", "whole_body_lqr"):
                controller = create_controller(
                    model_path=path,
                    controller=name,
                    terrain=str(terrain_path),
                    config=(
                        LQR_CONFIG_PATH if name == "lqr" else WHOLE_BODY_LQR_CONFIG_PATH
                    ),
                )
                for _ in range(1000):
                    controller.step()
                if (
                    not np.isfinite(controller.data.qvel).all()
                    or np.linalg.norm(controller.data.qvel[:6]) > 0.01
                ):
                    raise ValueError("Changed MJCF did not remain standing")
                if np.any(
                    np.abs(controller.data.ctrl) > controller.design.limits + 1e-10
                ):
                    raise ValueError("Reordered actuator limits are wrong")
        new_whole = WholeBodyModelling(model)
        new_vmc = VmcModelling(model)
        chassis = design_whole_body_lqr(new_whole, self.whole_config)
        virtual = design_lqr(new_vmc, self.vmc_config)
        if set(chassis.plant.chassis_actuators) & set(chassis.plant.gimbal_actuators):
            raise ValueError("Actuator reordering mixed ownership")
        if np.allclose(chassis.plant.a, self.whole.linearize(0.24).a):
            raise ValueError("Whole-body model ignored the changed gimbal mass")
        if (
            virtual.plant.physical_params["m_b"]
            <= self.vmc.linearize(0.24).physical_params["m_b"]
        ):
            raise ValueError("Equivalent model ignored the changed gimbal mass")
        if new_whole.linearize(0.24) is not chassis.plant:
            raise ValueError("Model cache was not reused within this robot")
        for builder in (new_whole, new_vmc):
            if builder.operating_point(0.24) is not builder.operating_point(0.24):
                raise ValueError("Operating-point cache was not reused")


class VmcTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))

    def test_closed_chain_geometry_and_virtual_work(self) -> None:
        model = self.model
        data = mujoco.MjData(model)
        for side, offset in (("left", 0), ("right", 2)):
            leg = Leg(model, side)
            names = (
                "front",
                "front_child1",
                "rear",
                "rear_child1",
                "rear_child2",
                "rear_child3",
            )
            qpos = [model.joint(f"{side}_{name}_joint").qposadr[0] for name in names]
            hip, wheel = [
                model.body(f"{side}_{name}_link").id for name in ("front", "wheel")
            ]
            equalities = slice(offset, offset + 2)
            for height in (0.10, 0.20, 0.24, 0.28, 0.40, 0.44):
                for forward in (-0.03, 0.0, 0.03):

                    def residual(angles: np.ndarray) -> np.ndarray:
                        data.qpos[qpos] = angles
                        mujoco.mj_kinematics(model, data)
                        return np.concatenate(
                            (
                                (
                                    data.site_xpos[model.eq_obj1id[equalities]]
                                    - data.site_xpos[model.eq_obj2id[equalities]]
                                )[:, [0, 2]].ravel(),
                                (data.xpos[wheel] - data.xpos[hip])[[0, 2]]
                                - [forward, -height],
                            )
                        )

                    result = least_squares(
                        residual, data.qpos[qpos], xtol=1e-12, ftol=1e-12, gtol=1e-12
                    )
                    self.assertTrue(result.success)
                    self.assertLess(np.max(np.abs(residual(result.x))), 1e-8)
                    leg.update(data)
                    actual = leg.length * np.array(
                        [np.cos(leg.angle), np.sin(leg.angle)]
                    )
                    np.testing.assert_allclose(
                        actual, [forward, height], atol=3e-6, rtol=0
                    )
                    phi = leg.offsets + leg.signs * data.qpos[leg.qpos]
                    derivative = np.empty((2, 2))
                    for i in range(2):
                        delta = np.zeros(2)
                        delta[i] = 1e-6 * leg.signs[i]
                        plus = np.array(leg.vmc.forward_kinematics(*(phi + delta))[2:])
                        minus = np.array(leg.vmc.forward_kinematics(*(phi - delta))[2:])
                        derivative[:, i] = (plus - minus) / 2e-6
                    np.testing.assert_allclose(
                        leg.j_t, derivative.T, atol=1e-8, rtol=1e-6
                    )
                    force = np.array([80.0, -2.0])
                    torque = leg.torque(*force)
                    np.testing.assert_allclose(
                        leg.vmc.joint_torque_to_virtual_force(leg.j_t, torque),
                        force,
                        atol=1e-10,
                    )
                    joint_speed = np.array([0.3, -0.4])
                    self.assertAlmostEqual(
                        torque @ joint_speed,
                        force @ (derivative @ joint_speed),
                        places=6,
                    )


if __name__ == "__main__":
    unittest.main()
