# Tandem Leg MuJoCo

基于 MuJoCo 的闭环轮腿机器人仿真，包含 demo 模型、LQR 控制器和 MPC 控制器。

## 环境

```bash
git clone --recurse-submodules <repository-url>
cd tandem_leg_sim_mujoco
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

macOS 上运行 MuJoCo viewer 时使用 `mjpython`：

```bash
.venv/bin/mjpython lqr_controller.py
.venv/bin/mjpython mpc_controller.py
```

LQR 默认读取 `configs/lqr.yaml`，MPC 默认读取 `configs/mpc.yaml`，两者共用 `MJCF/demo/demo.xml`。相机为自由视角，可用鼠标移动；运行时按住 `1/2` 给定 $\pm0.8\,\mathrm{m/s}$ 前进或后退速度，按住 `3/4` 给定左右转 $4.0\,\mathrm{rad/s}$ 角速度，松开后目标指令立即归零，空格键暂停或继续。前进和转向键可以组合使用；控制器按 YAML 中的 `linear_acceleration` 和 `yaw_acceleration` 将目标规划成斜坡，并用 `yaw_tracking_error_limit` 限制偏航角速度参考与实测值的差，避免持续转向时挤占平衡控制的力矩裕量。

## 验证

```bash
.venv/bin/mjpython verify_leg_ik.py
```

`verify_leg_ik.py` 在可达工作空间内采样目标腿长和角度，验证逆运动学与正运动学互为逆映射。

## 目录

```text
MJCF/demo/          demo 模型和网格
configs/            控制参数
sp_lqr/             状态空间模型、LQR 和 MPC 实现
utils/              运动学、VMC、PID 和 MuJoCo I/O
lqr_controller.py   LQR 仿真入口
mpc_controller.py   MPC 仿真入口
verify_leg_ik.py    腿部运动学验证
```
