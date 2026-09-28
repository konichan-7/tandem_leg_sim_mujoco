# Tandem Leg MuJoCo

基于 MuJoCo 的闭环轮腿机器人仿真，包含 demo 模型、LQR 控制器和 MPC 控制器。

机器人、关节、执行器和传感器定义在 `MJCF/demo/demo.xml` 中；Python 控制器读取仿真状态，计算电机力矩，再调用 MuJoCo 推进一步物理状态。本项目只使用传统控制算法，不包含强化学习和状态估计。

## 第三方库

| 库 | 在本项目中的作用 |
|---|---|
| MuJoCo | 加载 MJCF 模型，计算刚体动力学、接触和传感器数据，并显示交互式仿真窗口 |
| NumPy | 处理状态向量、控制向量、雅可比矩阵和矩阵运算 |
| SciPy | 离散化状态空间模型、求解 Riccati 方程和拟合变腿长模型 |
| python-control | 根据状态空间模型计算 LQR 反馈增益 |
| OSQP | 将 MPC 写成带约束二次规划问题并在线求解 |
| SymPy | 推导轮腿机器人线性化动力学矩阵 |
| PyYAML | 从 `configs/lqr.yaml` 和 `configs/mpc.yaml` 加载模型及控制参数 |
| GLFW | 处理仿真窗口、自由相机和键盘输入 |

## 核心算法

- **五连杆运动学**：由关节角计算腿长和腿角，也可以由目标腿长、腿角反解关节角。
- **VMC（虚拟模型控制）**：通过腿部雅可比矩阵，把期望支撑力和虚拟髋关节力矩映射为实际关节力矩。
- **腿长 PID**：维持左右腿目标长度，并提供机器人站立所需的轴向支撑力。
- **LQR**：基于线性化全身模型，同时反馈轮速、偏航、腿角和机体俯仰状态来保持平衡。
- **MPC**：在有限预测时域内使用同一套全身模型预测未来状态，通过 OSQP 求解满足轮毂和关节力矩约束的控制量。
- **速度斜坡与参考治理**：将键盘目标平滑转换为速度参考，并限制偏航跟踪误差，避免阶跃输入和持续转向破坏平衡。

每个仿真周期的基本数据流如下：

```text
MuJoCo 传感器 → 状态向量 → LQR/MPC → 轮毂力矩、虚拟髋关节力矩
                   腿部状态 → PID → 轴向支撑力
虚拟髋关节力矩 + 轴向支撑力 → VMC → 关节力矩 → MuJoCo 执行器
```

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

LQR 默认读取 `configs/lqr.yaml`，MPC 默认读取 `configs/mpc.yaml`，两者共用 `MJCF/demo/demo.xml`。相机为自由视角，可用鼠标移动；运行时按住 `1/2` 给定 $\pm0.8\,\mathrm{m/s}$ 前进或后退速度，按住 `3/4` 给定左右转 $4.0\,\mathrm{rad/s}$ 角速度，按住 `5/6` 升高或降低腿长，松开按键即停止对应输入，空格键暂停或继续。按键可以组合使用；控制器按 YAML 中的 `linear_acceleration` 和 `yaw_acceleration` 将目标规划成斜坡，并用 `yaw_tracking_error_limit` 限制偏航角速度参考与实测值的差，避免持续转向时挤占平衡控制的力矩裕量。

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
