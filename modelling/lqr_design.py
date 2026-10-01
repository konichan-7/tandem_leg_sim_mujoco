from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

from modelling.base import StateSpaceModel
from modelling.vmc_modelling import VmcModel, VmcModelling
from modelling.whole_body_modelling import WholeBodyModel, WholeBodyModelling

Model = TypeVar("Model", bound=StateSpaceModel)


@dataclass(frozen=True)
class LqrDesign(Generic[Model]):
    plant: Model
    gain: np.ndarray
    a: np.ndarray
    b: np.ndarray
    limits: np.ndarray
    spectral_radius: float

    @property
    def qpos(self) -> np.ndarray:
        return self.plant.point.qpos

    @property
    def torque(self) -> np.ndarray:
        return self.plant.point.torque


def motor_limits(
    modelling: VmcModelling | WholeBodyModelling, config: dict
) -> np.ndarray:
    limits = np.array(
        [
            config["control"]["motor_limits"][modelling.model.actuator(i).name]
            for i in range(modelling.model.nu)
        ]
    )
    if np.any(limits <= 0):
        raise ValueError("Motor limits must be positive")
    return limits


def solve_lqr(
    plant: StateSpaceModel, q: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, float]:
    if np.any(np.diag(q) < 0) or np.any(np.diag(r) <= 0):
        raise ValueError(
            "LQR requires nonnegative state weights and positive input weights"
        )
    a, b = plant.a, plant.b
    if plant.timestep == 0:
        p = solve_continuous_are(a, b, q, r)
        gain = np.linalg.solve(r, b.T @ p)
        poles = np.linalg.eigvals(a - b @ gain)
        if poles.real.max() >= 0:
            raise ValueError("Continuous LQR is not stable")
        return gain, poles.real.max().item()
    p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    radius = np.abs(np.linalg.eigvals(a - b @ gain)).max().item()
    if radius >= 1:
        raise ValueError(f"Discrete LQR is not stable: {radius}")
    return gain, radius


def design_lqr(modelling: VmcModelling, config: dict) -> LqrDesign[VmcModel]:
    plant = modelling.linearize(config["control"]["leg_height"])
    tuning = config["lqr"]
    q = np.diag(
        [
            tuning["Q_weights"][name] / tuning["state_limits"][name + "_max"] ** 2
            for name in plant.state_names
        ]
    )
    r = np.diag(
        [
            tuning["R_weights"][name] / tuning["control_limits"][name + "_max"] ** 2
            for name in plant.input_names
        ]
    )
    gain, abscissa = solve_lqr(plant, q, r)
    return LqrDesign(
        plant,
        gain,
        plant.a,
        plant.b,
        motor_limits(modelling, config),
        np.exp(abscissa * modelling.model.opt.timestep).item(),
    )


def gimbal_feedback(plant: WholeBodyModel, kp: float, kd: float) -> np.ndarray:
    feedback = np.zeros((len(plant.gimbal_actuators), len(plant.state_names)))
    feedback[np.arange(len(plant.gimbal_actuators)), plant.gimbal_position_columns] = kp
    feedback[np.arange(len(plant.gimbal_actuators)), plant.gimbal_velocity_columns] = kd
    return feedback


def design_whole_body_lqr(
    modelling: WholeBodyModelling, config: dict
) -> LqrDesign[WholeBodyModel]:
    plant = modelling.linearize(config["control"]["leg_height"])
    gimbal = config["gimbal"]
    h = gimbal_feedback(plant, gimbal["kp"], gimbal["kd"])
    a = plant.a - plant.b_gimbal @ h
    limits = motor_limits(modelling, config)
    if np.any(np.abs(plant.point.torque) >= limits):
        raise ValueError("Standing feedforward exceeds motor limits")
    weights = config["lqr"]["Q_weights"]
    q = np.diag(
        [weights["position"][name] for name in plant.position_names]
        + [weights["velocity"][name] for name in plant.velocity_names]
    )
    gimbal_columns = np.r_[plant.gimbal_position_columns, plant.gimbal_velocity_columns]
    if np.any(np.diag(q)[gimbal_columns] != 0):
        raise ValueError(
            "Gimbal tracking belongs to its independent controller; LQR gimbal state weights must be zero"
        )
    r = np.diag(1 / limits[plant.chassis_actuators] ** 2)
    controlled_plant = StateSpaceModel(
        a, plant.b, plant.timestep, plant.state_names, plant.input_names, plant.point
    )
    gain, radius = solve_lqr(controlled_plant, q, r)
    return LqrDesign(plant, gain, a, plant.b, limits, radius)
