from collections.abc import Mapping
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ImuData:
    quat: np.ndarray
    acc: np.ndarray
    gyro: np.ndarray


class MujocoSensorReader:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

    def sensor(self, name: str) -> np.ndarray:
        return self.data.sensor(name).data.copy()

    def scalar(self, name: str) -> float:
        return float(self.data.sensor(name).data[0])

    def imu(self, quat: str, acc: str, gyro: str) -> ImuData:
        return ImuData(self.sensor(quat), self.sensor(acc), self.sensor(gyro))

    def joint_positions(self, sensors: Mapping[str, str]) -> dict[str, float]:
        return {name: self.scalar(sensor) for name, sensor in sensors.items()}

    def joint_velocity(self, sensor: str) -> float:
        return self.scalar(sensor)

    def joint_velocities(self, sensors: Mapping[str, str]) -> dict[str, float]:
        return {name: self.joint_velocity(sensor) for name, sensor in sensors.items()}


class MujocoActuatorWriter:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        actuators: Mapping[str, str],
    ) -> None:
        self.data = data
        self.ids = {
            name: model.actuator(actuator).id
            for name, actuator in actuators.items()
        }

    def set(self, name: str, value: float) -> None:
        self.data.ctrl[self.ids[name]] = value

    def set_many(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            self.set(name, value)

    def zero(self) -> None:
        self.data.ctrl[:] = 0.0
