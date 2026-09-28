import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent


@dataclass(frozen=True)
class DemoPaths:
    xml: Path = ROOT / "MJCF" / "demo" / "demo.xml"
    lqr_yaml: Path = ROOT / "configs" / "lqr.yaml"
    mpc_yaml: Path = ROOT / "configs" / "mpc.yaml"


@dataclass(frozen=True)
class DemoJoints:
    left_front: str = "left_front_joint"
    left_rear: str = "left_rear_joint"
    right_front: str = "right_front_joint"
    right_rear: str = "right_rear_joint"
    left_wheel: str = "left_wheel_joint"
    right_wheel: str = "right_wheel_joint"

    @property
    def leg(self) -> tuple[str, str, str, str]:
        return self.left_front, self.left_rear, self.right_front, self.right_rear

    @property
    def wheels(self) -> tuple[str, str]:
        return self.left_wheel, self.right_wheel


@dataclass(frozen=True)
class DemoActuators:
    left_front: str = "left_front_motor"
    left_rear: str = "left_rear_motor"
    right_front: str = "right_front_motor"
    right_rear: str = "right_rear_motor"
    left_wheel: str = "left_wheel_motor"
    right_wheel: str = "right_wheel_motor"


@dataclass(frozen=True)
class DemoSensors:
    left_front_pos: str = "left_front_joint_pos"
    left_rear_pos: str = "left_rear_joint_pos"
    right_front_pos: str = "right_front_joint_pos"
    right_rear_pos: str = "right_rear_joint_pos"
    left_wheel_pos: str = "left_wheel_joint_pos"
    right_wheel_pos: str = "right_wheel_joint_pos"
    left_front_vel: str = "left_front_joint_vel"
    left_rear_vel: str = "left_rear_joint_vel"
    right_front_vel: str = "right_front_joint_vel"
    right_rear_vel: str = "right_rear_joint_vel"
    left_wheel_vel: str = "left_wheel_joint_vel"
    right_wheel_vel: str = "right_wheel_joint_vel"
    quat: str = "quat"
    acc: str = "acc"
    gyro: str = "gyro"


@dataclass(frozen=True)
class DemoEncoderOffsets:
    left_phi1: float = -2.68
    left_phi4: float = -0.62
    right_phi1: float = -2.68
    right_phi4: float = -0.62


@dataclass(frozen=True)
class DemoVmcGeometry:
    l1: float = 0.16
    l2: float = 0.169
    l3: float = 0.169
    l4: float = 0.16
    l5: float = 0.0


@dataclass(frozen=True)
class DemoControl:
    target_l0: float = 0.1
    target_phi0: float = math.pi / 2
    target_s: float = 0.0
    target_velocity: float = 0.0
    target_yaw: float = 0.0
    target_yaw_rate: float = 0.0
    lqr_start: float = 0.01
    stand_kp: float = 5.0
    stand_kd: float = 0.5
    leg_force_kp: float = 1000.0
    leg_force_ki: float = 1000.0
    leg_force_kd: float = 40.0
    leg_force_limit: float = 80.0
    leg_force_integral_limit: float = 60.0
    viewer_fps: float = 60.0


PATHS = DemoPaths()
JOINTS = DemoJoints()
ACTUATORS = DemoActuators()
SENSORS = DemoSensors()
OFFSETS = DemoEncoderOffsets()
VMC_GEOMETRY = DemoVmcGeometry()
CONTROL = DemoControl()
KEY_COMMANDS = {
    "0": (0.0, 0.0, 0.0),
    "1": (1.0, 0.0, 0.0),
    "2": (-1.0, 0.0, 0.0),
    "3": (0.0, -1.0, 0.0),
    "4": (0.0, 1.0, 0.0),
    "5": (0.0, 0.0, 1.0),
    "6": (0.0, 0.0, -1.0),
}
