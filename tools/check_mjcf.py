from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

MODEL = Path(__file__).resolve().parents[1] / "MJCF/leg_hero.xml"


def closure_error(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    delta = data.site_xpos[model.eq_obj1id] - data.site_xpos[model.eq_obj2id]
    return np.linalg.norm(delta, axis=1).max().item()


def check_source(model: mujoco.MjModel) -> None:
    urdf = ET.parse(MODEL.with_suffix(".urdf")).getroot()
    for link in urdf.findall("link"):
        name = link.attrib["name"]
        body = model.body(name)
        if name == "yaw_link":
            continue
        inertial = link.find("inertial")
        mass = float(inertial.find("mass").attrib["value"])
        if not np.isclose(body.mass[0], mass, rtol=1e-12):
            raise ValueError(f"{name}: mass differs from URDF")
        values = inertial.find("inertia").attrib
        expected = np.array(
            [
                [float(values[f"i{a}{b}" if a <= b else f"i{b}{a}"]) for b in "xyz"]
                for a in "xyz"
            ]
        )
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, body.iquat)
        rotation = rotation.reshape(3, 3)
        actual = (rotation * body.inertia) @ rotation.T
        if not np.allclose(actual, expected, rtol=1e-5, atol=1e-10):
            raise ValueError(f"{name}: inertia differs from URDF")
    for joint in urdf.findall("joint"):
        name = joint.attrib["name"].replace(
            "right_rear_child2_link", "right_rear_child2_joint"
        )
        expected = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
        if not np.allclose(model.joint(name).axis, expected):
            raise ValueError(f"{name}: axis differs from URDF")


def simulate(fixed_base: bool, driven: bool, duration: float = 5.0) -> None:
    xml = ET.parse(MODEL).getroot()
    xml.find("compiler").set("meshdir", str(MODEL.parent))
    if fixed_base:
        body = xml.find("worldbody/body[@name='base_link']")
        body.remove(body.find("freejoint"))
        body.set("pos", "0 0 0.7")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial_error = closure_error(model, data)
    if model.neq != 4 or model.nu != 8 or initial_error > 1e-10:
        raise ValueError("Unexpected topology or initially open loops")
    eq_rows = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    jacobian = data.efc_J.reshape(data.nefc, model.nv)[eq_rows]
    rank = np.linalg.matrix_rank(jacobian, tol=1e-8)
    if rank != 8:
        raise ValueError(
            f"Expected 8 independent planar closure constraints, got {rank}"
        )
    initial_qpos = data.qpos.copy()
    peak_error = initial_error
    contact_steps = 0
    steps = round(duration / model.opt.timestep)
    for _ in range(steps):
        if driven:
            envelope = np.exp(-data.time)
            data.ctrl[:] = (
                envelope
                * np.sin(2 * np.pi * data.time)
                * np.array([1, -1, 1, -1, 0.1, -0.1, 0.05, 0.05])
            )
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        peak_error = max(peak_error, closure_error(model, data))
        contact_steps += data.ncon > 0
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Nonfinite simulation state")
    if any(warning.number for warning in data.warning):
        raise ValueError(f"MuJoCo warnings: {data.warning}")
    if abs(data.time - duration) > 1e-8 or peak_error > 1e-3:
        raise ValueError(
            f"Simulation reset or closure drift: t={data.time}, error={peak_error}"
        )
    if not fixed_base and not contact_steps:
        raise ValueError("No ground contact")
    if fixed_base and driven and np.linalg.norm(data.qpos - initial_qpos) < 0.01:
        raise ValueError("Actuated mechanism did not move")
    print(
        f"fixed_base={fixed_base}, driven={driven}, duration={duration:g}s, "
        f"rank={rank}, initial_error={initial_error:.3e}m, "
        f"peak_error={peak_error:.3e}m, contact_steps={contact_steps}"
    )


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    check_source(model)
    print(
        f"MuJoCo {mujoco.__version__}: {model.nbody - 1} bodies, "
        f"{model.nq} qpos, {model.nv} velocities, {model.nu} motors, "
        f"{model.neq} loops, mass={model.body_mass.sum():.6f}kg"
    )
    simulate(fixed_base=False, driven=False)
    simulate(fixed_base=True, driven=True)
    simulate(fixed_base=False, driven=True)
