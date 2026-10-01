from abc import ABC, abstractmethod
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class OperatingPoint:
    qpos: np.ndarray
    torque: np.ndarray
    residual: float


@dataclass(frozen=True)
class StateSpaceModel:
    a: np.ndarray
    b: np.ndarray
    timestep: float
    state_names: tuple[str, ...]
    input_names: tuple[str, ...]
    point: OperatingPoint


class RobotLayout:
    def __init__(self, model: mujoco.MjModel) -> None:
        chassis = tuple(
            f"{side}_{part}_motor"
            for side in ("left", "right")
            for part in ("front", "rear")
        ) + ("left_wheel_motor", "right_wheel_motor")
        gimbal = ("yaw_motor", "pitch_motor")
        self.chassis = np.array([model.actuator(name).id for name in chassis])
        self.gimbal = np.array([model.actuator(name).id for name in gimbal])
        self.active = np.concatenate((self.chassis, self.gimbal))
        if model.nu != len(self.active) or model.na != 0:
            raise ValueError("Expected six chassis motors and two direct gimbal motors")
        for name in chassis + gimbal:
            motor = model.actuator(name)
            if (
                motor.trntype[0] != mujoco.mjtTrn.mjTRN_JOINT
                or model.joint(motor.trnid[0]).name != name.replace("_motor", "_joint")
                or not np.array_equal(motor.gear, [1, 0, 0, 0, 0, 0])
            ):
                raise ValueError(f"Expected a unit-gear joint torque motor: {name}")
        if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.neq != 4:
            raise ValueError(
                "Expected a floating base and four closed-chain connections"
            )
        if np.any(model.eq_type != mujoco.mjtEq.mjEQ_CONNECT):
            raise ValueError("Closed chains must use connect equalities")
        if model.opt.timestep <= 0 or not np.allclose(model.opt.gravity[:2], 0):
            raise ValueError("Expected a positive timestep and vertical gravity")
        self.dofs = model.jnt_dofadr[model.actuator_trnid[self.active, 0]]
        self.passive = np.setdiff1d(np.arange(model.nv), np.r_[np.arange(6), self.dofs])
        if len(self.passive) != 2 * model.neq:
            raise ValueError(
                "Expected two independent planar closure rows per connection"
            )


class RobotModelling(ABC):
    symmetric_equilibrium = True

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.layout = RobotLayout(model)
        self._points: dict[float, OperatingPoint] = {}

    @abstractmethod
    def linearize(self, height: float) -> StateSpaceModel:
        raise NotImplementedError

    def operating_point(self, height: float) -> OperatingPoint:
        if height not in self._points:
            qpos, torque, residual = equilibrium(
                self.model, height, symmetric=self.symmetric_equilibrium
            )
            self._points[height] = OperatingPoint(qpos, torque, residual)
        return self._points[height]


def equilibrium(
    model: mujoco.MjModel, leg_height: float, *, symmetric: bool = True
) -> tuple[np.ndarray, np.ndarray, float]:
    data = mujoco.MjData(model)
    sides = ("left", "right")
    leg_joints = [
        model.joint(f"{side}_{part}_joint")
        for side in sides
        for part in (
            "front",
            "front_child1",
            "rear",
            "rear_child1",
            "rear_child2",
            "rear_child3",
        )
    ]
    leg_qpos = np.array([joint.qposadr[0] for joint in leg_joints])
    leg_dofs = np.array([joint.dofadr[0] for joint in leg_joints])
    wheels = np.array([model.body(f"{side}_wheel_link").id for side in sides])
    radii = np.array(
        [model.geom(f"{side}_wheel_link_collision").size[0] for side in sides]
    )
    base = model.body("base_link").id
    hip_height = model.body("left_front_link").pos[2]
    data.qpos[2] = leg_height + radii.mean() - hip_height

    def geometry(variables: np.ndarray) -> np.ndarray:
        data.qpos[leg_qpos] = variables[:-1]
        mujoco.mj_forward(model, data)
        closure = data.site_xpos[model.eq_obj1id] - data.site_xpos[model.eq_obj2id]
        wheel_positions = data.xpos[wheels]
        return np.concatenate(
            (
                closure[:, [0, 2]].ravel(),
                wheel_positions[:, 0] - variables[-1],
                wheel_positions[:, 2] - radii,
                [variables[-1] - data.subtree_com[base, 0]],
            )
        )

    pose = least_squares(
        geometry,
        np.zeros(len(leg_qpos) + 1),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=300,
    )
    if not pose.success or np.linalg.norm(geometry(pose.x), ord=np.inf) > 1e-8:
        raise ValueError(
            "Requested leg height has no closed standing pose near the URDF assembly"
        )
    geometric_pose = data.qpos.copy()
    active_dofs = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    gimbal = [model.joint(f"{name}_joint") for name in ("yaw", "pitch")]
    gimbal_dofs = np.array([joint.dofadr[0] for joint in gimbal])
    gimbal_qpos = np.array([joint.qposadr[0] for joint in gimbal])
    variable_dofs = np.concatenate(([2, 3, 4], leg_dofs, gimbal_dofs))

    def static_residual(variables: np.ndarray) -> np.ndarray:
        data.qpos[:] = geometric_pose
        displacement = np.zeros(model.nv)
        displacement[variable_dofs] = variables[: len(variable_dofs)]
        mujoco.mj_integratePos(model, data.qpos, displacement, 1)
        data.ctrl[:] = variables[len(variable_dofs) :]
        data.qacc[:] = 0
        mujoco.mj_inverse(model, data)
        residual = data.qfrc_inverse.copy()
        residual[active_dofs] -= data.ctrl
        roll_pitch = Rotation.from_quat(data.qpos[3:7], scalar_first=True).as_euler(
            "xyz"
        )[:2]
        return np.concatenate(
            (
                residual,
                [100 * (data.qpos[2] - geometric_pose[2])],
                100 * roll_pitch,
                100 * data.qpos[gimbal_qpos],
                (
                    [
                        displacement[model.joint("left_front_joint").dofadr[0]]
                        * model.joint("left_front_joint").axis[1]
                        - displacement[model.joint("right_front_joint").dofadr[0]]
                        * model.joint("right_front_joint").axis[1]
                    ]
                    if symmetric
                    else []
                ),
            )
        )

    seed = np.zeros(len(variable_dofs) + model.nu)
    seed[0] = -0.0005
    trim = least_squares(
        static_residual,
        seed,
        x_scale="jac",
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-10,
        max_nfev=300,
    )
    residual = np.linalg.norm(static_residual(trim.x), ord=np.inf).item()
    if not trim.success or residual > 1e-6:
        raise ValueError(
            f"Standing inverse dynamics did not converge: residual={residual:.3e}"
        )
    return data.qpos.copy(), data.ctrl.copy(), residual


def closed_chain_tangent(
    model: mujoco.MjModel, data: mujoco.MjData
) -> tuple[np.ndarray, np.ndarray]:
    active = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    retained = np.concatenate((np.arange(6), active))
    passive = np.setdiff1d(np.arange(model.nv), retained)
    jacobian = planar_equality_jacobian(model, data)
    tangent = np.zeros((model.nv, len(retained)))
    tangent[retained] = np.eye(len(retained))
    tangent[passive, 6:] = -np.linalg.solve(jacobian[:, passive], jacobian[:, active])
    return tangent, retained


def planar_equality_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    jacobian = data.efc_J.reshape(data.nefc, model.nv)
    jacobian = jacobian[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    plane = data.xmat[model.body("base_link").id].reshape(3, 3)[:, [0, 2]]
    jacobian = np.einsum(
        "ci,ncj->nij", plane, jacobian.reshape(model.neq, 3, model.nv)
    ).reshape(-1, model.nv)
    return jacobian
