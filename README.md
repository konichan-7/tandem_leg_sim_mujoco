# leg_hero 闭链轮腿机器人

基于 MuJoCo 的传统力矩控制仿真，包含左右闭链腿、驱动轮和独立 yaw/pitch 云台。程序通过 `main.py` 统一启动，支持选择机器人 MJCF、控制器和地形。

| 控制器 | 建模与控制方式 |
|---|---|
| `lqr`（默认） | 10 维等效摆杆模型，4 路虚拟力矩反馈，配合腿长 PID 和 VMC 输出底盘电机力矩 |
| `whole_body_lqr` | 完整多体动力学的 24 维局部约简模型，直接输出 6 路底盘电机力矩 |

两版均使用独立云台 PD，并保留各自的跳跃策略。whole-body 模型包含云台动力学，但底盘 LQR 不控制云台电机。控制器直接使用仿真状态，不包含强化学习或状态估计。

## 安装与启动

使用 Python 3.12，在项目根目录执行：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/mjpython main.py
```

macOS 交互窗口使用 `mjpython`。Linux 将运行命令的解释器替换为 `.venv/bin/python`；Windows 使用 `py -3.12 -m venv .venv` 创建环境，再通过 `.venv\Scripts\python.exe` 安装依赖及运行 `main.py`。

```bash
# 24 维底盘 LQR，平地
.venv/bin/mjpython main.py --controller whole_body_lqr

# 加载 20 cm 台阶地形
.venv/bin/mjpython main.py --controller whole_body_lqr --terrain step

# 选择同构机器人、地形和控制配置
.venv/bin/mjpython main.py --model /path/to/robot.xml \
  --controller whole_body_lqr --terrain /path/to/terrain.xml \
  --config /path/to/control.yaml

# 无窗口站立仿真，运行 5 秒仿真时间
.venv/bin/mjpython main.py --controller lqr --headless --duration 5
```

| 参数 | 默认值 / 可选项 |
|---|---|
| `--model` | `MJCF/leg_hero.xml`，可指定同构机器人 MJCF |
| `--controller` | `lqr` 或 `whole_body_lqr` |
| `--terrain` | `flat`、`step` 或自定义地形 MJCF 路径 |
| `--config` | 默认加载所选控制器对应的 YAML |
| `--headless` | 不创建窗口，执行零运动指令仿真 |
| `--duration` | 无窗口运行时长，默认 5 s，必须为有限正数 |

两种控制配置不能混用。自定义模型需满足现有角色命名、闭链拓扑和直接力矩电机约定；初始轮下需有 z=0 的水平支撑面。加载器替换机器人内名为 `floor` 的地面，自定义地形的测高几何体应使用 group 1。详细接口约束见技术报告。

## 按键

| 按键 | 操作 |
|---|---|
| 1 / 2 | 前进 / 后退 |
| 3 / 4 | 负 / 正偏航 |
| 5 / 6 | 升高 / 降低腿高 |
| Shift | 保持云台世界指向并旋转底盘；松开后减速、对齐云台方向 |
| 空格 | 按住下蹲，松开起跳 |
| P | 暂停 / 继续 |

## 目录与文档

```text
main.py                 统一仿真入口
controllers/            chassis.py 底盘基类，两版控制器、云台与跳跃控制
modelling/              base.py 通用基座，kinematics.py，lqr_design.py，两版建模
configs/                两种控制器的 YAML 配置
MJCF/                   机器人模型、网格与 terrains/ 地形
utils/                  PID、斜坡、路径和交互窗口
tests/                  建模与控制器测试
compare_lqr.py          两种架构的对照实验
docs/
  whole_body_lqr.md     动力学、状态空间、两种建模比较与控制框图
  jump.md               跳跃阶段、支撑判定和空中力矩分配
```

- [Whole-body LQR 技术报告](docs/whole_body_lqr.md)：完整多体动力学、闭链消元、24 维状态空间、独立云台闭环、10 维解析模型及接口约定。
- [跳跃控制说明](docs/jump.md)：两版策略差异、阶段切换、轨迹和力矩分配。

根目录保留本 README 作为使用入口；详细技术文档集中在 `docs/`。模型与网格已包含在仓库中，当前运行不依赖 `sp_lqr` 子模块。

## 验证

```bash
.venv/bin/mjpython -m unittest discover -s tests -v
.venv/bin/mjpython compare_lqr.py --output reports/lqr_comparison
```

测试覆盖闭链与 VMC、线性预测、同构模型替换、配平稳定性、扰动恢复、云台独立性、跳跃阶段机及入口组合。对照实验运行 16 组场景、32 次试验，输出 JSON/CSV。

最近一次入口整理在 macOS ARM64 上通过八项测试，32 次对照均完成；结果与适用范围见 [报告验证章节](docs/whole_body_lqr.md#13-验证历史结果与未解决问题)。窗口交互和其他平台尚未复验。20 cm 台阶越障仍为已知未解决问题，地形加载和原地跳跃成功不代表越障通过。
