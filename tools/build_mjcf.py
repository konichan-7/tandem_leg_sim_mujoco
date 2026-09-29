from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "MJCF/leg_hero.urdf"
OUTPUT = URDF.with_suffix(".xml")
YAW_DENSITY = 2700.0
LOOP_PINS = {
    "left": ((-0.09627728, 0.09463449), (-0.09336046, -0.02480701)),
    "right": ((-0.11299873, 0.07386668), (-0.08881104, -0.03800210)),
}


def numbers(values: np.ndarray) -> str:
    return " ".join(f"{value:.15g}" for value in values)


def vector(element: ET.Element, key: str) -> np.ndarray:
    return np.fromstring(element.attrib[key], sep=" ")


def stl(name: str) -> trimesh.Trimesh:
    return trimesh.load_mesh(URDF.parent / f"{name}.STL")


def build() -> None:
    urdf = ET.parse(URDF).getroot()
    links = {link.attrib["name"]: link for link in urdf.findall("link")}
    joints = {
        joint.find("child").attrib["link"]: joint for joint in urdf.findall("joint")
    }
    root = ET.Element("mujoco", model="leg_hero")
    ET.SubElement(
        root, "compiler", angle="radian", meshdir=".", inertiafromgeom="false"
    )
    ET.SubElement(
        root,
        "option",
        timestep="0.001",
        integrator="implicitfast",
        solver="Newton",
        iterations="100",
        tolerance="1e-10",
        cone="elliptic",
        jacobian="dense",
    )
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "joint", type="hinge", limited="false", damping="0.02")
    ET.SubElement(
        defaults,
        "geom",
        contype="1",
        conaffinity="2",
        friction="0.8 0.005 0.0001",
        condim="3",
        solref="0.005 1",
    )
    ET.SubElement(defaults, "site", size="0.003", group="3")
    ET.SubElement(
        defaults, "equality", solref="0.003 1", solimp="0.99 0.999 0.001 0.5 2"
    )
    ET.SubElement(root, "statistic", center="0 0 0.4", extent="1.2")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.8 0.8 0.8")
    ET.SubElement(visual, "global", azimuth="135", elevation="-20")
    asset = ET.SubElement(root, "asset")
    stl("base_link").export(URDF.parent / "base_link.obj")
    for name in links:
        extension = "obj" if name == "base_link" else "STL"
        ET.SubElement(
            asset, "mesh", name=name, file=f"{name}.{extension}", inertia="legacy"
        )
    ET.SubElement(
        asset,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0.16 0.20 0.24",
        rgb2="0.23 0.27 0.31",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "material",
        name="floor_material",
        texture="grid",
        texrepeat="4 4",
        texuniform="true",
        reflectance="0.1",
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="5 5 0.1",
        material="floor_material",
        contype="2",
        conaffinity="1",
    )
    bodies = {}
    poses = {}
    wheel_sizes = {}

    def add_link(name: str, parent: ET.Element, transform: np.ndarray) -> None:
        body = ET.SubElement(parent, "body", name=name)
        bodies[name] = body
        if name in joints:
            joint = joints[name]
            origin = joint.find("origin")
            rotation = Rotation.from_euler("xyz", vector(origin, "rpy")).as_matrix()
            local = np.eye(4)
            local[:3, :3] = rotation
            local[:3, 3] = vector(origin, "xyz")
            transform = transform @ local
            quat = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
            body.set("pos", origin.attrib["xyz"])
            body.set("quat", numbers(quat))
            joint_name = joint.attrib["name"].replace(
                "right_rear_child2_link", "right_rear_child2_joint"
            )
            ET.SubElement(
                body, "joint", name=joint_name, axis=joint.find("axis").attrib["xyz"]
            )
        else:
            ET.SubElement(body, "freejoint", name="root")
        poses[name] = transform
        inertial = links[name].find("inertial")
        if name == "yaw_link":
            mesh = stl(name)
            mesh.merge_vertices(digits_vertex=6)
            mesh.update_faces(mesh.unique_faces())
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.fix_normals(multibody=True)
            mesh.density = YAW_DENSITY
            inertia = mesh.moment_inertia
            ET.SubElement(
                body,
                "inertial",
                pos=numbers(mesh.center_mass),
                mass=f"{mesh.mass:.15g}",
                fullinertia=numbers(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]),
            )
            print(
                f"yaw estimate: mass={mesh.mass:.6f} kg, watertight={mesh.is_watertight}"
            )
        else:
            origin = inertial.find("origin")
            values = inertial.find("inertia").attrib
            inertia = np.array(
                [
                    [float(values[f"i{a}{b}" if a <= b else f"i{b}{a}"]) for b in "xyz"]
                    for a in "xyz"
                ]
            )
            rotation = Rotation.from_euler("xyz", vector(origin, "rpy")).as_matrix()
            inertia = rotation @ inertia @ rotation.T
            ET.SubElement(
                body,
                "inertial",
                pos=origin.attrib["xyz"],
                mass=inertial.find("mass").attrib["value"],
                fullinertia=numbers(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]),
            )
        geom = ET.SubElement(
            body,
            "geom",
            name=f"{name}_visual",
            type="mesh",
            mesh=name,
            rgba=links[name].find("visual/material/color").attrib["rgba"],
        )
        if "wheel" in name:
            mesh = stl(name)
            radius = np.linalg.norm(mesh.vertices[:, [0, 2]], axis=1).max()
            low, high = mesh.bounds[:, 1]
            wheel_sizes[name] = radius
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("rgba", "0.08 0.08 0.09 1")
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_collision",
                type="cylinder",
                size=numbers(np.array([radius, (high - low) / 2])),
                pos=numbers(np.array([0, (high + low) / 2, 0])),
                quat="0.707106781186548 0.707106781186548 0 0",
                group="3",
                rgba="0.08 0.08 0.09 1",
            )
        for child_name, joint in joints.items():
            if joint.find("parent").attrib["link"] == name:
                add_link(child_name, body, transform)

    add_link("base_link", world, np.eye(4))
    equality = ET.SubElement(root, "equality")
    for side, pins in LOOP_PINS.items():
        for index, (front, rear, pin) in enumerate(
            zip(
                ("front_link", "front_child1_link"),
                ("rear_child2_link", "rear_child3_link"),
                pins,
            ),
            start=1,
        ):
            first, second = f"{side}_{front}", f"{side}_{rear}"
            center = vector(links[second].find("inertial/origin"), "xyz")
            rear_pin = np.array([pin[0], center[1], pin[1], 1])
            front_pin = np.linalg.solve(poses[first], poses[second] @ rear_pin)
            for name, position, color in (
                (first, front_pin, "1 0.3 0.2 1"),
                (second, rear_pin, "0.2 1 0.3 1"),
            ):
                ET.SubElement(
                    bodies[name],
                    "site",
                    name=f"{side}_loop{index}_{name}",
                    pos=numbers(position[:3]),
                    rgba=color,
                )
            ET.SubElement(
                equality,
                "connect",
                name=f"{side}_loop{index}",
                site1=f"{side}_loop{index}_{first}",
                site2=f"{side}_loop{index}_{second}",
            )
    ET.SubElement(bodies["base_link"], "site", name="imu", pos="0 0 0", rgba="0 1 0 1")
    motors = (
        "left_front",
        "left_rear",
        "right_front",
        "right_rear",
        "left_wheel",
        "right_wheel",
        "yaw",
        "pitch",
    )
    actuator = ET.SubElement(root, "actuator")
    sensor = ET.SubElement(root, "sensor")
    for name in motors:
        ET.SubElement(
            actuator, "motor", name=f"{name}_motor", joint=f"{name}_joint", gear="1"
        )
        for tag in ("jointpos", "jointvel"):
            ET.SubElement(sensor, tag, name=f"{name}_{tag}", joint=f"{name}_joint")
    ET.SubElement(sensor, "framequat", name="base_quat", objtype="site", objname="imu")
    ET.SubElement(sensor, "gyro", name="base_gyro", site="imu")
    ET.SubElement(sensor, "accelerometer", name="base_acc", site="imu")
    height = (
        max(radius - poses[name][2, 3] for name, radius in wheel_sizes.items()) + 0.002
    )
    bodies["base_link"].set("pos", numbers(np.array([0, 0, height])))
    custom = ET.SubElement(root, "custom")
    ET.SubElement(
        custom,
        "text",
        name="yaw_inertia_status",
        data="ESTIMATED from non-watertight STL with corrected winding at 2700 kg/m^3; mass and inertia require calibration",
    )
    ET.SubElement(custom, "numeric", name="yaw_density_kg_m3", data=str(YAW_DENSITY))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(OUTPUT, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    print(f"{OUTPUT}: nq={model.nq}, nv={model.nv}, nu={model.nu}, neq={model.neq}")


if __name__ == "__main__":
    build()
