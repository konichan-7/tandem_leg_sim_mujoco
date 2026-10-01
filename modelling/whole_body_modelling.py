from dataclasses import dataclass

import mujoco
import numpy as np

from modelling.base import (
    RobotModelling,
    StateSpaceModel,
    planar_equality_jacobian,
)


@dataclass(frozen=True)
class WholeBodyModel(StateSpaceModel):
    b_gimbal: np.ndarray
    position_dofs: np.ndarray
    velocity_dofs: np.ndarray
    position_names: tuple[str, ...]
    velocity_names: tuple[str, ...]
    chassis_actuators: np.ndarray
    gimbal_actuators: np.ndarray
    gimbal_position_columns: np.ndarray
    gimbal_velocity_columns: np.ndarray
    lift: np.ndarray
    rows: np.ndarray


class WholeBodyModelling(RobotModelling):
    symmetric_equilibrium = False

    def __init__(
        self, model: mujoco.MjModel, finite_difference_step: float = 1e-6
    ) -> None:
        super().__init__(model)
        if finite_difference_step <= 0:
            raise ValueError("Finite difference step must be positive")
        self.finite_difference_step = finite_difference_step
        self._models: dict[float, WholeBodyModel] = {}

    def linearize(self, height: float) -> WholeBodyModel:
        if height in self._models:
            return self._models[height]
        model, layout = self.model, self.layout
        point = self.operating_point(height)
        data = mujoco.MjData(model)
        data.qpos[:] = point.qpos
        data.ctrl[:] = point.torque
        mujoco.mj_forward(model, data)
        if np.linalg.norm(data.qacc, ord=np.inf) > 1e-5 or data.ncon < 2:
            raise ValueError("Standing point is not a forward-dynamics equilibrium")
        velocity_dofs = np.r_[[0, 2, 3, 4, 5], layout.dofs]
        jacobian = planar_equality_jacobian(model, data)
        tangent = np.zeros((model.nv, len(velocity_dofs)))
        tangent[velocity_dofs] = np.eye(len(velocity_dofs))
        tangent[layout.passive] = -np.linalg.solve(
            jacobian[:, layout.passive], jacobian[:, velocity_dofs]
        )
        wheel_dofs = np.array(
            [model.joint(f"{side}_wheel_joint").dofadr[0] for side in ("left", "right")]
        )
        position_columns = np.flatnonzero(~np.isin(velocity_dofs, wheel_dofs))
        position_dofs = velocity_dofs[position_columns]
        if np.linalg.norm(jacobian @ tangent, ord=np.inf) > 1e-8:
            raise ValueError("Reduced tangent violates closed-chain equalities")
        lift = np.zeros((2 * model.nv, len(position_dofs) + len(velocity_dofs)))
        lift[: model.nv, : len(position_dofs)] = tangent[:, position_columns]
        lift[model.nv :, len(position_dofs) :] = tangent
        rows = np.r_[position_dofs, model.nv + velocity_dofs]
        full_a = np.empty((2 * model.nv, 2 * model.nv))
        full_b = np.empty((2 * model.nv, model.nu))
        mujoco.mjd_transitionFD(
            model, data, self.finite_difference_step, True, full_a, full_b, None, None
        )
        a = (full_a @ lift)[rows]
        b = full_b[rows][:, layout.chassis]
        b_gimbal = full_b[rows][:, layout.gimbal]
        velocity_names = ("forward", "height", "roll", "pitch", "heading") + tuple(
            model.joint(model.actuator_trnid[i, 0]).name for i in layout.active
        )
        position_names = tuple(velocity_names[i] for i in position_columns)
        gimbal_dofs = model.jnt_dofadr[model.actuator_trnid[layout.gimbal, 0]]
        gimbal_position = np.array(
            [np.flatnonzero(position_dofs == dof)[0] for dof in gimbal_dofs]
        )
        gimbal_velocity = len(position_dofs) + np.array(
            [np.flatnonzero(velocity_dofs == dof)[0] for dof in gimbal_dofs]
        )
        result = WholeBodyModel(
            a,
            b,
            model.opt.timestep,
            tuple("position/" + name for name in position_names)
            + tuple("velocity/" + name for name in velocity_names),
            tuple(model.actuator(i).name for i in layout.chassis),
            point,
            b_gimbal,
            position_dofs,
            velocity_dofs,
            position_names,
            velocity_names,
            layout.chassis,
            layout.gimbal,
            gimbal_position,
            gimbal_velocity,
            lift,
            rows,
        )
        self._models[height] = result
        return result
