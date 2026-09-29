import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def equilibrium(
    model: mujoco.MjModel, leg_height: float
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
