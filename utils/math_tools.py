from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are
from utils.mujoco_io import equilibrium


@dataclass(frozen=True)
class LqrDesign:
    qpos: np.ndarray
    torque: np.ndarray
    gain: np.ndarray
    position_dofs: np.ndarray
    velocity_dofs: np.ndarray
    limits: np.ndarray
    spectral_radius: float
    equilibrium_residual: float


def design_lqr(model: mujoco.MjModel, config: dict) -> LqrDesign:
    qpos, torque, residual = equilibrium(model, config["control"]["leg_height"])
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = torque
    mujoco.mj_forward(model, data)
    if np.linalg.norm(data.qacc, ord=np.inf) > 1e-5 or data.ncon < 2:
        raise ValueError(
            "Standing pose is not a forward-dynamics equilibrium with wheel contact"
        )

    active = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    joint_names = [model.joint(joint).name for joint in model.actuator_trnid[:, 0]]
    wheel_dofs = np.array(
        [model.joint(f"{side}_wheel_joint").dofadr[0] for side in ("left", "right")]
    )
    velocity_dofs = np.concatenate(([0, 2, 3, 4, 5], active))
    position_columns = np.flatnonzero(~np.isin(velocity_dofs, wheel_dofs))
    position_dofs = velocity_dofs[position_columns]
    passive = np.setdiff1d(np.arange(model.nv), np.concatenate((np.arange(6), active)))
    jacobian = data.efc_J.reshape(data.nefc, model.nv)
    jacobian = jacobian[data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]
    jacobian = jacobian.reshape(model.neq, 3, model.nv)[:, [0, 2]].reshape(-1, model.nv)
    tangent = np.zeros((model.nv, len(velocity_dofs)))
    tangent[velocity_dofs] = np.eye(len(velocity_dofs))
    tangent[passive] = -np.linalg.solve(
        jacobian[:, passive], jacobian[:, velocity_dofs]
    )
    if np.linalg.norm(jacobian @ tangent, ord=np.inf) > 1e-10:
        raise ValueError("Closed-chain tangent does not satisfy the equality Jacobian")

    lift = np.zeros((2 * model.nv, len(position_dofs) + len(velocity_dofs)))
    lift[: model.nv, : len(position_dofs)] = tangent[:, position_columns]
    lift[model.nv :, len(position_dofs) :] = tangent
    rows = np.concatenate((position_dofs, model.nv + velocity_dofs))
    full_a = np.empty((2 * model.nv, 2 * model.nv))
    full_b = np.empty((2 * model.nv, model.nu))
    mujoco.mjd_transitionFD(
        model,
        data,
        config["lqr"]["finite_difference_step"],
        True,
        full_a,
        full_b,
        None,
        None,
    )
    a = (full_a @ lift)[rows]
    b = full_b[rows]
    base_names = ["forward", "height", "roll", "pitch", "heading"]
    velocity_names = base_names + joint_names
    position_names = [velocity_names[index] for index in position_columns]
    q = np.diag(
        [config["lqr"]["Q_weights"]["position"][name] for name in position_names]
        + [config["lqr"]["Q_weights"]["velocity"][name] for name in velocity_names]
    )
    limits = np.array(
        [
            config["lqr"]["control_limits"][model.actuator(index).name]
            for index in range(model.nu)
        ]
    )
    if np.any(np.abs(torque) >= limits):
        raise ValueError("Standing feedforward exceeds configured motor limits")
    r = np.diag(1 / limits**2)
    p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    spectral_radius = np.abs(np.linalg.eigvals(a - b @ gain)).max().item()
    if spectral_radius >= 1:
        raise ValueError(f"Unstable reduced LQR: spectral radius={spectral_radius}")
    return LqrDesign(
        qpos,
        torque,
        gain,
        position_dofs,
        velocity_dofs,
        limits,
        spectral_radius,
        residual,
    )


def move_towards(current: float, target: float, max_delta: float) -> float:
    if current < target:
        return min(current + max_delta, target)
    return max(current - max_delta, target)
