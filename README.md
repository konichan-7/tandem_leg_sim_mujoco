# Tandem Leg MuJoCo

闭环轮腿机器人仿真，包含 LQR、MPC，以及独立的平地 locomotion policy。模型为 `MJCF/demo/demo.xml`；原始 MJCF 保存在 `MJCF/demo/demo_original.xml`。四组闭链约束及真实左右差异保留，几何修正见 [模型说明](docs/closed_chain_model_alignment.md)。

## 环境与运行

```bash
git clone --recurse-submodules <repository-url>
cd tandem_leg_sim_mujoco
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ./tandem_mjlab
```

macOS 的 MuJoCo 窗口使用 `mjpython`。

### 正式 policy

```bash
.venv/bin/mjpython -m tandem_mjlab.sim2sim --keyboard
```

默认前后1m/s、转向2rad/s。按住 **1/2** 前进/后退、**3/4** 右转/左转，可组合，松开减速归零；Ctrl+C退出。静止腿动作平滑已包含在模型内。

发布版为 `tandem_mjlab/policies/locomotion.onnx`，同版本 PT、协议、评估与已知限制见 [发布说明](tandem_mjlab/policies/README.md)。+2m/s直线航向仍未达到现有门槛。

### 传统控制

```bash
.venv/bin/mjpython lqr_controller.py
.venv/bin/mjpython mpc_controller.py
.venv/bin/mjpython verify_leg_ik.py
```

LQR/MPC 分别读取 `configs/lqr.yaml`、`configs/mpc.yaml`。状态反馈产生轮力矩及虚拟髋力矩，腿长PID产生支撑力，经VMC映射至关节力矩。MPC使用OSQP求解受限预测控制，详见 [MPC说明](MPC.md)。

按住1/2前后移动、3/4左右转、5/6改变腿长，松开停止对应输入，空格暂停/继续；具体速度、腿长范围和斜坡由YAML配置。

## 目录

| 路径 | 内容 |
|---|---|
| `MJCF/demo/` | 原始与运行MJCF、STL网格 |
| `configs/`、`utils/` | 控制参数、运动学、VMC、PID和MuJoCo I/O |
| `sp_lqr/` | 状态空间模型与控制子模块 |
| `tandem_mjlab/` | 唯一正式policy、部署、续训和评估 |
| `docs/` | 闭链模型说明 |

[RL入口与代码边界](tandem_mjlab/README.md)。旧模型、训练阶段编排和重复评估入口已移除；新作业与轨迹输出到被Git忽略的 `outputs/`。
