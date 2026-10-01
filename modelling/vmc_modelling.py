from dataclasses import dataclass
import math

import mujoco
import numpy as np

from modelling.whole_body_modelling import RobotModelling, StateSpaceModel


class VMC:
    def __init__(self, l1: float, l2: float, l3: float, l4: float, l5: float) -> None:
        self.l1, self.l2, self.l3, self.l4, self.l5 = l1, l2, l3, l4, l5

    def forward_kinematics(
        self, phi1: float, phi4: float
    ) -> tuple[float, float, float, float]:
        x_b = -self.l5 / 2 + math.cos(phi1) * self.l1
        y_b = math.sin(phi1) * self.l1
        x_d = self.l5 / 2 + math.cos(phi4) * self.l4
        y_d = math.sin(phi4) * self.l4
        a = 2 * self.l2 * (x_d - x_b)
        b = 2 * self.l2 * (y_d - y_b)
        c = self.l2**2 + (x_d - x_b) ** 2 + (y_d - y_b) ** 2 - self.l3**2
        phi2 = 2 * math.atan2(b + math.sqrt(a * a + b * b - c * c), a + c)
        x_c = x_b + self.l2 * math.cos(phi2)
        y_c = y_b + self.l2 * math.sin(phi2)
        phi3 = math.atan2(y_c - y_d, x_c - x_d)
        return phi2, phi3, math.hypot(x_c, y_c), math.atan2(y_c, x_c)

    def inverse_kinematics(self, l0: float, phi0: float) -> tuple[float, float]:
        x_c, y_c = l0 * math.cos(phi0), l0 * math.sin(phi0)
        dist_ac = math.hypot(x_c + self.l5 / 2, y_c)
        alpha = math.acos(
            (self.l1**2 + dist_ac**2 - self.l2**2) / (2 * self.l1 * dist_ac)
        )
        phi1 = math.atan2(y_c, x_c + self.l5 / 2) + alpha
        dist_ec = math.hypot(x_c - self.l5 / 2, y_c)
        beta = math.acos(
            (self.l4**2 + dist_ec**2 - self.l3**2) / (2 * self.l4 * dist_ec)
        )
        phi4 = math.atan2(y_c, x_c - self.l5 / 2) - beta
        return phi1, phi4

    def mat_jrm(
        self, phi0: float, phi1: float, phi2: float, phi3: float, phi4: float, l0: float
    ) -> np.ndarray:
        denom = math.sin(phi3 - phi2)
        return np.array(
            [
                [
                    self.l1 * math.sin(phi0 - phi3) * math.sin(phi1 - phi2) / denom,
                    self.l1
                    * math.sin(phi1 - phi2)
                    * math.cos(phi0 - phi3)
                    / (denom * l0),
                ],
                [
                    self.l4 * math.sin(phi0 - phi2) * math.sin(phi3 - phi4) / denom,
                    self.l4
                    * math.sin(phi3 - phi4)
                    * math.cos(phi0 - phi2)
                    / (denom * l0),
                ],
            ]
        )

    def virtual_force_to_joint_torque(
        self, j_t: np.ndarray, leg_force: float, hip_torque: float
    ) -> np.ndarray:
        return j_t @ np.array([leg_force, hip_torque])

    def joint_torque_to_virtual_force(
        self, j_t: np.ndarray, joint_torque: np.ndarray
    ) -> np.ndarray:
        return np.linalg.solve(j_t, joint_torque)


class Leg:
    def __init__(self, model: mujoco.MjModel, side: str) -> None:
        joints = [model.joint(f"{side}_{part}_joint") for part in ("front", "rear")]
        self.qpos = np.array([joint.qposadr[0] for joint in joints])
        self.dofs = np.array([joint.dofadr[0] for joint in joints])
        self.actuators = np.array(
            [model.actuator(f"{side}_{part}_motor").id for part in ("front", "rear")]
        )
        self.signs = np.array([joint.axis[1] for joint in joints])
        cranks = np.array(
            [
                model.body(f"{side}_{part}_child1_link").pos[[0, 2]]
                for part in ("front", "rear")
            ]
        )
        self.offsets = np.arctan2(-cranks[:, 1], cranks[:, 0])
        upper = np.linalg.norm(cranks[0])
        lower = np.linalg.norm(model.body(f"{side}_wheel_link").pos[[0, 2]])
        self.vmc = VMC(upper, lower, lower, upper, 0.0)
        self.length = self.angle = 0.0
        self.velocity = np.zeros(2)
        self.j_t = np.zeros((2, 2))
        self.force = np.zeros(2)

    def update(self, data: mujoco.MjData) -> None:
        phi1, phi4 = self.offsets + self.signs * data.qpos[self.qpos]
        phi2, phi3, self.length, self.angle = self.vmc.forward_kinematics(phi1, phi4)
        self.j_t[:] = self.signs[:, None] * self.vmc.mat_jrm(
            self.angle, phi1, phi2, phi3, phi4, self.length
        )
        self.velocity[:] = self.j_t.T @ data.qvel[self.dofs]
        self.force[:] = self.vmc.joint_torque_to_virtual_force(
            self.j_t, data.qfrc_actuator[self.dofs]
        )

    def torque(self, force: float, moment: float) -> np.ndarray:
        return self.vmc.virtual_force_to_joint_torque(self.j_t, force, moment)


STATE_NAMES = (
    "s",
    "ds",
    "phi",
    "dphi",
    "theta_ll",
    "dtheta_ll",
    "theta_lr",
    "dtheta_lr",
    "theta_b",
    "dtheta_b",
)
INPUT_NAMES = ("T_wl", "T_wr", "T_bl", "T_br")


@dataclass(frozen=True)
class VmcModel(StateSpaceModel):
    physical_params: dict[str, float]


def aggregate_inertia(
    model: mujoco.MjModel, data: mujoco.MjData, bodies: list[int]
) -> tuple[float, np.ndarray, np.ndarray]:
    masses = model.body_mass[bodies]
    mass = masses.sum().item()
    center = np.sum(masses[:, None] * data.xipos[bodies], axis=0) / mass
    inertia = np.zeros((3, 3))
    for body in bodies:
        frame = data.ximat[body].reshape(3, 3)
        offset = data.xipos[body] - center
        inertia += (frame * model.body_inertia[body]) @ frame.T
        inertia += model.body_mass[body] * (
            offset @ offset * np.eye(3) - np.outer(offset, offset)
        )
    return mass, center, inertia


def equivalent_parameters(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    body_ids = [model.body(name).id for name in ("base_link", "yaw_link", "pitch_link")]
    mass, center, inertia = aggregate_inertia(model, data, body_ids)
    hip_center = np.mean(
        [data.xpos[model.body(f"{side}_front_link").id] for side in ("left", "right")],
        axis=0,
    )
    wheels = [model.body(f"{side}_wheel_link").id for side in ("left", "right")]
    _, _, whole_inertia = aggregate_inertia(model, data, list(range(1, model.nbody)))
    params = {
        "g": np.linalg.norm(model.opt.gravity).item(),
        "R_w": np.mean(
            [
                model.geom(f"{side}_wheel_link_collision").size[0]
                for side in ("left", "right")
            ]
        ).item(),
        "R_l": (abs(data.xpos[wheels[0], 1] - data.xpos[wheels[1], 1]) / 2).item(),
        "l_c": (hip_center[2] - center[2]).item(),
        "m_b": mass,
        "I_b": inertia[1, 1].item(),
        "I_z": whole_inertia[2, 2].item(),
        "m_w": model.body_mass[wheels].mean().item(),
    }
    wheel_inertias, leg_masses = [], []
    for side, suffix, wheel in zip(("left", "right"), ("l", "r"), wheels):
        frame = data.ximat[wheel].reshape(3, 3)
        wheel_inertias.append(((frame * model.body_inertia[wheel]) @ frame.T)[1, 1])
        bodies = [
            i
            for i in range(model.nbody)
            if model.body(i).name.startswith(f"{side}_") and i != wheel
        ]
        leg_mass, leg_center, leg_inertia = aggregate_inertia(model, data, bodies)
        leg = Leg(model, side)
        leg.update(data)
        hip = data.xpos[model.body(f"{side}_front_link").id]
        direction = hip - data.xpos[wheel]
        direction[1] = 0.0
        direction /= np.linalg.norm(direction)
        wheel_to_com = (leg_center - data.xpos[wheel]) @ direction
        params.update(
            {
                f"l_{suffix}": leg.length,
                f"l_w{suffix}": wheel_to_com.item(),
                f"l_b{suffix}": (leg.length - wheel_to_com).item(),
                f"I_l{suffix}": leg_inertia[1, 1].item(),
            }
        )
        leg_masses.append(leg_mass)
    params["m_l"] = np.mean(leg_masses).item()
    params["I_w"] = np.mean(wheel_inertias).item()
    return params


def equivalent_dynamics(p: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    rw, rl = p["R_w"], p["R_l"]
    ll, lr, lc = p["l_l"], p["l_r"], p["l_c"]
    lwl, lwr, lbl, lbr = p["l_wl"], p["l_wr"], p["l_bl"], p["l_br"]
    mw, ml, mb = p["m_w"], p["m_l"], p["m_b"]
    iw, iz, g = p["I_w"], p["I_z"], p["g"]
    c7 = -(mw * rw**2 + iw + ml * rw**2 + mb * rw**2 / 2)
    c11 = mw * rw * lc + iw * lc / rw + ml * rw * lc
    c16 = iz * rw / (2 * rl) + iw * rl / rw
    mass = np.array(
        [
            [
                iw * ll / rw + mw * rw * ll + ml * rw * lbl,
                0,
                ml * lwl * lbl - p["I_ll"],
                0,
                0,
            ],
            [
                0,
                iw * lr / rw + mw * rw * lr + ml * rw * lbr,
                0,
                ml * lwr * lbr - p["I_lr"],
                0,
            ],
            [
                c7,
                c7,
                -(ml * rw * lwl + mb * rw * ll / 2),
                -(ml * rw * lwr + mb * rw * lr / 2),
                0,
            ],
            [c11, c11, ml * lwl * lc, ml * lwr * lc, -p["I_b"]],
            [c16, -c16, iz * ll / (2 * rl), -iz * lr / (2 * rl), 0],
        ]
    )
    gravity = np.zeros((5, 3))
    gravity[0, 0] = -(ml * lwl + mb * ll / 2) * g
    gravity[1, 1] = -(ml * lwr + mb * lr / 2) * g
    gravity[3, 2] = -mb * g * lc
    inputs = np.array(
        [
            [1 + ll / rw, 0, -1, 0],
            [0, 1 + lr / rw, 0, -1],
            [-1, -1, 0, 0],
            [lc / rw, lc / rw, 1, 1],
            [rl / rw, -rl / rw, 0, 0],
        ]
    )
    acceleration = np.linalg.solve(mass, np.column_stack((gravity, inputs)))
    coordinates = np.eye(5)
    coordinates[0] = [rw / 2, rw / 2, 0, 0, 0]
    coordinates[1] = [-rw / (2 * rl), rw / (2 * rl), -ll / (2 * rl), lr / (2 * rl), 0]
    acceleration = coordinates @ acceleration
    a, b = np.zeros((10, 10)), np.zeros((10, 4))
    a[::2, 1::2] = np.eye(5)
    a[1::2, 4::2] = acceleration[:, :3]
    b[1::2] = acceleration[:, 3:]
    return a, b


class VmcModelling(RobotModelling):
    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model)
        self._models: dict[float, VmcModel] = {}

    def linearize(self, height: float) -> VmcModel:
        if height not in self._models:
            point = self.operating_point(height)
            data = mujoco.MjData(self.model)
            data.qpos[:] = point.qpos
            data.ctrl[:] = point.torque
            mujoco.mj_forward(self.model, data)
            params = equivalent_parameters(self.model, data)
            a, b = equivalent_dynamics(params)
            self._models[height] = VmcModel(
                a, b, 0.0, STATE_NAMES, INPUT_NAMES, point, params
            )
        return self._models[height]
