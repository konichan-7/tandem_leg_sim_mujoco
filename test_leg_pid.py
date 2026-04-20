import math
import time

import mujoco
import mujoco.viewer
import numpy as np

L1, L2, L3, L4, L5 = 0.215, 0.254, 0.254, 0.215, 0.0


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def getPhi(phi1, phi4, l1, l2, l3, l4, l5):
    try:
        x_B = -l5 / 2 + math.cos(phi1) * l1
        y_B = math.sin(phi1) * l1
        x_D = l5 / 2 + math.cos(phi4) * l4
        y_D = math.sin(phi4) * l4
        A_0 = 2 * l2 * (x_D - x_B)
        B_0 = 2 * l2 * (y_D - y_B)
        l_BD = ((x_D - x_B) ** 2 + (y_D - y_B) ** 2) ** 0.5
        C_0 = l2**2 + l_BD**2 - l3**2

        val = A_0**2 + B_0**2 - C_0**2
        if val < 0:
            val = 0
        phi2 = wrap(2 * math.atan2(B_0 + val**0.5, A_0 + C_0))

        x_C = -l5 / 2 + l1 * math.cos(phi1) + l2 * math.cos(phi2)
        y_C = l1 * math.sin(phi1) + l2 * math.sin(phi2)
        phi3 = wrap(math.atan2(y_C - y_D, x_C - x_D))
        l_0 = (x_C**2 + y_C**2) ** 0.5
        phi_0 = math.atan2(y_C, x_C)
        return phi2, phi3, l_0, phi_0
    except Exception:
        return 0, 0, 0.15, 1.57


def ik(L0, phi0, l1, l2, l3, l4, l5):
    xC = L0 * math.cos(phi0)
    yC = L0 * math.sin(phi0)

    dist_AC = math.sqrt((xC + l5 / 2) ** 2 + yC**2)
    cos_alpha = (l1**2 + dist_AC**2 - l2**2) / (2 * l1 * dist_AC)
    alpha = math.acos(np.clip(cos_alpha, -1, 1))
    angle_AC = math.atan2(yC, xC + l5 / 2)
    phi1 = angle_AC + alpha

    dist_EC = math.sqrt((xC - l5 / 2) ** 2 + yC**2)
    cos_beta = (l4**2 + dist_EC**2 - l3**2) / (2 * l4 * dist_EC)
    beta = math.acos(np.clip(cos_beta, -1, 1))
    angle_EC = math.atan2(yC, xC - l5 / 2)
    phi4 = angle_EC - beta
    return phi1, phi4


def Mat_JRM(phi0, phi1, phi2, phi3, phi4, L0, l1, l4):
    denom = math.sin(phi3 - phi2)
    return np.array(
        [
            [
                l1 * math.sin(phi0 - phi3) * math.sin(phi1 - phi2) / denom,
                l1 * math.sin(phi1 - phi2) * math.cos(phi0 - phi3) / (denom * L0),
            ],
            [
                l4 * math.sin(phi0 - phi2) * math.sin(phi3 - phi4) / denom,
                l4 * math.sin(phi3 - phi4) * math.cos(phi0 - phi2) / (denom * L0),
            ],
        ],
        dtype=np.float64,
    )


class PID_control:
    def __init__(self, kp, ki, kd, target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0
        self.last_error = 0

    def position_pid(self, current, dt):
        error = self.target - current
        self.integral += error * dt
        self.integral = np.clip(self.integral, -500, 500)
        derivative = (error - self.last_error) / dt
        self.last_error = error
        return np.clip(
            self.kp * error + self.ki * self.integral + self.kd * derivative,
            -200,
            200,
        )


class CascadePID:
    def __init__(self, p_kp, p_ki, p_kd, v_kp, v_ki, v_kd, target, v_limit, out_limit):
        self.pos_pid = PID_control(p_kp, p_ki, p_kd, target)
        self.vel_pid = PID_control(v_kp, v_ki, v_kd, 0.0)
        self.v_limit = v_limit
        self.out_limit = out_limit

    @property
    def target(self):
        return self.pos_pid.target

    @target.setter
    def target(self, val):
        self.pos_pid.target = val

    def compute(self, pos, vel, dt):
        target_v = self.pos_pid.position_pid(pos, dt)
        target_v = np.clip(target_v, -self.v_limit, self.v_limit)

        self.vel_pid.target = target_v
        out = self.vel_pid.position_pid(vel, dt)
        return np.clip(out, -self.out_limit, self.out_limit)


def main():
    model = mujoco.MjModel.from_xml_path("MJCF/balance_bot.xml")
    data = mujoco.MjData(model)

    l_front_idx = model.joint("Left_front_joint").qposadr[0]
    l_rear_idx = model.joint("Left_rear_joint").qposadr[0]
    r_front_idx = model.joint("Right_front_joint").qposadr[0]
    r_rear_idx = model.joint("Right_rear_joint").qposadr[0]

    data.qpos[2] = 0.4
    data.qpos[l_front_idx] = 1.0
    data.qpos[l_rear_idx] = -1.0
    data.qpos[r_front_idx] = -1.0
    data.qpos[r_rear_idx] = 1.0

    last_print_time = 0

    print("Started leg PID test. Base is nailed in the air.")
    print("Will track 0.15m for 1s, then a sine wave L0 target.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            # 强行钉住浮动基座(悬空状态)
            data.qpos[:3] = [0, 0, 0.4]
            data.qpos[3:7] = [1, 0, 0, 0]
            data.qvel[:6] = 0

            # if data.time > 1.0:
            #     current_target_L0 = 0.15 + 0.05 * math.sin(
            #         2 * math.pi * 0.5 * (data.time - 1.0)
            #     )
            #     current_target_phi0 = math.pi / 2 + 0.2 * math.cos(
            #         2 * math.pi * 0.5 * (data.time - 1.0)
            #     )
            # else:
            current_target_L0 = 0.15 + 0.05 * math.sin(2 * math.pi * 1.0 * data.time)
            current_target_phi0 = math.pi / 2 + 0.2 * math.cos(
                2 * math.pi * 0.5 * data.time
            )

            # 使用 IK 直接解算关节角并下发
            p1, p4 = ik(current_target_L0, current_target_phi0, L1, L2, L3, L4, L5)

            q_rear_target = wrap(math.pi - p1)
            q_front_target = wrap(p4)

            # 强行设置 qpos 与 qvel 使得关节为绝对的位置模式
            data.qpos[l_rear_idx] = q_rear_target
            data.qpos[l_front_idx] = q_front_target
            data.qpos[r_rear_idx] = wrap(p1 - math.pi)
            data.qpos[r_front_idx] = wrap(-p4)

            data.qvel[model.joint("Left_front_joint").dofadr[0]] = 0.0
            data.qvel[model.joint("Left_rear_joint").dofadr[0]] = 0.0
            data.qvel[model.joint("Right_front_joint").dofadr[0]] = 0.0
            data.qvel[model.joint("Right_rear_joint").dofadr[0]] = 0.0

            phi_l1 = wrap(math.pi - data.qpos[l_rear_idx])
            phi_l4 = wrap(data.qpos[l_front_idx])
            phi_r1 = wrap(math.pi + data.qpos[r_rear_idx])
            phi_r4 = wrap(-data.qpos[r_front_idx])

            p2l, p3l, L0_l, phi0_l = getPhi(phi_l1, phi_l4, L1, L2, L3, L4, L5)
            p2r, p3r, L0_r, phi0_r = getPhi(phi_r1, phi_r4, L1, L2, L3, L4, L5)

            if data.time - last_print_time > 0.1:
                print(
                    f"t={data.time:.2f} cmd_L0={current_target_L0:.4f} FK_L0={L0_l:.4f} err_L0={current_target_L0-L0_l:.4f} | "
                    f"cmd_phi0={current_target_phi0:.4f} FK_phi0={phi0_l:.4f} err_phi0={current_target_phi0-phi0_l:.4f}"
                )
                last_print_time = data.time

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
