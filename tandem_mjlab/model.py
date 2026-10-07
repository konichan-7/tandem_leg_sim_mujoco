from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

MODEL_PATH = Path(__file__).resolve().parent.parent / "MJCF/demo/demo.xml"
REFERENCE_PATH = Path(__file__).with_name("nominal_state.json")
NOMINAL_STATE = json.loads(REFERENCE_PATH.read_text())
LEG_JOINTS = (
    "left_front_joint",
    "left_rear_joint",
    "right_front_joint",
    "right_rear_joint",
)
WHEEL_JOINTS = ("left_wheel_joint", "right_wheel_joint")
ACTUATED_JOINTS = LEG_JOINTS + WHEEL_JOINTS
OBSERVATION_DIM = 3 + 3 + 3 + len(LEG_JOINTS) + 2 * len(ACTUATED_JOINTS) + 2
HISTORY_LENGTH = 5
LEG_LENGTH_BODIES = tuple(
    name.removesuffix("_joint") + "_link" for name in ACTUATED_JOINTS
)
LEG_STIFFNESS = 20.0
LEG_DAMPING = 1.0
WHEEL_DAMPING = 0.2
LEG_ACTION_SCALE = 0.5
WHEEL_ACTION_SCALE = 10.0
PHYSICS_TIMESTEP = 0.0005
POLICY_PERIOD = 0.02
HEADING_ERROR_SCALE = 0.05
SIMULATION_OPTIONS = {
    "timestep": PHYSICS_TIMESTEP,
    "integrator": "implicitfast",
    "solver": "newton",
    "cone": "pyramidal",
    "jacobian": "auto",
    "iterations": 50,
    "tolerance": 1e-8,
    "ls_iterations": 50,
    "ls_tolerance": 0.01,
    "ccd_iterations": 50,
    "impratio": 1.0,
    "gravity": (0.0, 0.0, -9.81),
}


def apply_simulation_options(model: mujoco.MjModel) -> None:
    enums = {
        "integrator": (mujoco.mjtIntegrator, "mjINT_"),
        "solver": (mujoco.mjtSolver, "mjSOL_"),
        "cone": (mujoco.mjtCone, "mjCONE_"),
        "jacobian": (mujoco.mjtJacobian, "mjJAC_"),
    }
    for name, value in SIMULATION_OPTIONS.items():
        if name in enums:
            enum, prefix = enums[name]
            value = getattr(enum, prefix + value.upper())
        setattr(model.opt, name, value)


LINEAR_COMMAND_RANGE = (-2.0, 2.0)
YAW_COMMAND_RANGE = (-5.0, 5.0)
DECIMATION = round(POLICY_PERIOD / PHYSICS_TIMESTEP)
LOOPS = (
    ("left_front_site1", "left_rear_site1"),
    ("left_front_site2", "left_rear_site2"),
    ("right_front_site1", "right_rear_site1"),
    ("right_front_site2", "right_rear_site2"),
)


def standing_pose() -> tuple[dict[str, float], float, dict[str, float]]:
    return (
        NOMINAL_STATE["joint_positions"],
        NOMINAL_STATE["base_height_m"],
        NOMINAL_STATE["motor_efforts"],
    )


def motor_limits() -> dict[str, float]:
    return NOMINAL_STATE["motor_limits"]


@lru_cache(maxsize=1)
def nominal_leg_lengths() -> tuple[float, float]:
    model = get_spec().compile()
    data = mujoco.MjData(model)
    for name, position in initial_joint_positions().items():
        data.qpos[model.joint(name).qposadr] = position
    mujoco.mj_forward(model, data)
    positions = data.xpos[[model.body(name).id for name in LEG_LENGTH_BODIES]]
    hips = positions[:4].reshape(2, 2, 3).mean(axis=1)
    offsets = (positions[4:] - hips) @ data.xmat[model.body("base_link").id].reshape(
        3, 3
    )
    lengths = np.linalg.norm(offsets[:, [0, 2]], axis=1)
    return float(lengths[0]), float(lengths[1])


@lru_cache(maxsize=1)
def wheel_radii() -> tuple[float, ...]:
    model = get_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    radii = []
    for name in WHEEL_JOINTS:
        joint = model.joint(name)
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] != model.jnt_bodyid[joint.id]:
                continue
            mesh_id = model.geom_dataid[geom_id]
            start = model.mesh_vertadr[mesh_id]
            vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
            offsets = (
                vertices @ data.geom_xmat[geom_id].reshape(3, 3).T
                + data.geom_xpos[geom_id]
                - data.xanchor[joint.id]
            )
            axis = data.xaxis[joint.id]
            radial = offsets - (offsets @ axis)[:, None] * axis
            radii.append(np.linalg.norm(radial, axis=1).max())
    return tuple(float(radius) for radius in radii)


@lru_cache(maxsize=1)
def wheel_action_limit() -> float:
    model = get_spec().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    half_track = (
        abs(
            data.xanchor[model.joint(WHEEL_JOINTS[0]).id, 1]
            - data.xanchor[model.joint(WHEEL_JOINTS[1]).id, 1]
        )
        / 2
    )
    return float(
        (LINEAR_COMMAND_RANGE[1] + YAW_COMMAND_RANGE[1] * half_track)
        / min(wheel_radii())
        / WHEEL_ACTION_SCALE
    )


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    model = spec.compile()

    wheel_bodies = {model.jnt_bodyid[model.joint(name).id] for name in WHEEL_JOINTS}
    for geom_id, geom in enumerate(spec.geoms):
        if geom.name != "floor":
            geom.contype = 0
            geom.conaffinity = 1
        if model.geom_bodyid[geom_id] in wheel_bodies:
            geom.name = f"{model.body(model.geom_bodyid[geom_id]).name}_collision"
            geom.priority = 1
    return spec


def initial_joint_positions() -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    active_positions, _, _ = standing_pose()
    joint_addresses = {
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
        ): model.jnt_qposadr[joint_id]
        for joint_id in range(model.njnt)
    }
    for name, position in active_positions.items():
        data.qpos[joint_addresses[name]] = position

    passive_names = tuple(
        name
        for name in joint_addresses
        if name is not None and name not in ACTUATED_JOINTS
    )

    def loop_error(passive_positions: np.ndarray) -> np.ndarray:
        for name, position in zip(passive_names, passive_positions, strict=True):
            data.qpos[joint_addresses[name]] = position
        mujoco.mj_forward(model, data)
        return np.concatenate(
            [data.site(first).xpos - data.site(second).xpos for first, second in LOOPS]
        )

    solution = least_squares(loop_error, np.zeros(len(passive_names)))
    return {
        **active_positions,
        **dict(zip(passive_names, map(float, solution.x), strict=True)),
        **dict.fromkeys(WHEEL_JOINTS, 0.0),
    }
