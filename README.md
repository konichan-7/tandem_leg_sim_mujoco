# leg_hero 闭链轮腿机器人

MuJoCo 模型及传统 LQR 力矩控制，支持左右闭链腿、驱动轮和 yaw/pitch 云台。当前工作分支为 `leg_hero`。

## 启动控制器

```bash
.venv/bin/mjpython lqr_controller.py
```

左右腿质量、质心和惯量已以左腿为基准镜像对称。启动时从 MJCF 自动求站立平衡点和 LQR 增益，直接初始化到闭链、重力和接触平衡状态。云台默认保持相对机身零位，由独立的 `GimbalController` 控制 yaw/pitch。

| 按键 | 功能 |
| --- | --- |
| 1 / 2 | 前进 / 后退，默认 ±0.8 m/s |
| 3 / 4 | 负 / 正偏航，默认 ∓1.2 rad/s |
| 5 / 6 | 升高 / 降低腿高，默认 ±0.05 m/s，范围 0.20–0.28 m |
| 左 / 右 Shift | 按住时锁定云台世界系指向，底盘以 +8 rad/s 旋转 |
| 空格 | 暂停 / 继续 |

原有键盘映射及 viewer 沿用 `demo` 分支：按住生效，松开清零对应方向输入，支持组合键。前进和转向按配置的加速度减速，腿高松键后保持目标值。速度和腿高范围使用 hero 的配置。

新增 Shift 按住模式：捕获按下时云台的世界系指向，云台反向补偿底盘旋转。此时平移目标为零，覆盖 `1–4` 的运动指令，`5/6` 仍可调腿高。底盘沿 6 rad/s² 斜坡升至 8 rad/s，参考起转时间约 1.33 秒；松开两个 Shift 后，云台继续保持同一世界系指向；底盘先按原斜坡减速，再以不超过普通转向速度 1.2 rad/s 就近对齐云台朝向。归位期间暂停 `1–4` 的运动输入，完成后恢复；再次按下 Shift 可直接重新进入旋转模式。

```bash
.venv/bin/mjpython verify_lqr.py
```

配置见 [configs/lqr.yaml](configs/lqr.yaml)，控制方法、参考 main 分支的适配说明及测试结果见 [LQR.md](LQR.md)。

代码按 `demo` 分支组织：

```text
demo.py                  路径、显示配置和按键映射
lqr_controller.py        参数解析与启动入口
configs/lqr.yaml         control、command、lqr 参数
utils/
  __init__.py            导出 DemoLqrController
  lqr_control.py         控制器、斜坡规划、力矩控制与仿真步
  gimbal_controller.py   云台 yaw/pitch 参考、世界指向保持与力矩输出
  math_tools.py          闭链约简、LQR 求解、move_towards
  mujoco_io.py           MuJoCo 平衡姿态与逆动力学
  viewer.py              demo 键盘及显示逻辑，增加左右 Shift 状态读取
verify_lqr.py            无界面仿真验证
```

入口与 `demo` 原文件一致。viewer 通过 `command` 更新目标，仿真调用链为 `step → control → lqr_control → update_command`；云台控制器使用同一次全系统 LQR 求解中对应云台的增益行，保留底盘与云台的动力学耦合。闭链动力学仍由 hero 的 MJCF 推导。

## 模型与环境

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

查看无控制器的原始模型：

```bash
.venv/bin/mjpython -m mujoco.viewer --mjcf MJCF/leg_hero.xml
```

无控制器时机器人会倒下。重新生成 MJCF 和检查模型：

```bash
.venv/bin/mjpython tools/build_mjcf.py
.venv/bin/mjpython tools/check_mjcf.py
```

几何来源、四处闭链、关节轴方向、碰撞设定和 yaw 惯量估算见 [MJCF/README.md](MJCF/README.md)。质量、惯量、杆长、轮径与关节轴均以 MJCF 为准，控制配置不重复保存这些参数。
