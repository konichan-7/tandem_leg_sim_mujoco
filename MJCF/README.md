# leg_hero 闭链仿真模型

`leg_hero.xml` 从同目录的 `leg_hero.urdf` 生成，保留原始坐标系、质量、质心、惯量、关节轴和初始装配姿态。唯一缺失的动力学数据是 `yaw_link` 的质量和惯量，当前使用网格估算，待标定。

在项目根目录启动：

```bash
.venv/bin/mjpython -m mujoco.viewer --mjcf MJCF/leg_hero.xml
```

当前目录已建立 `.venv` 并安装依赖。迁移到其他电脑时：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

macOS 图形界面使用 `mjpython`。模型没有平衡控制器，零力矩运行时会倒下；这不是闭链失效。关节位置为 URDF 装配姿态下的零位，左右腿原始姿态并不对称，轮心高度约差 10.3 mm，未人为修改为对称站立姿态。

## 闭链拓扑

沿用参考 `demo.xml` 的连接关系，闭合点重新从当前 STL 的销孔圆心取得，没有复制参考机器人的尺寸。

| 约束 | 第一刚体 | 第二刚体 |
| --- | --- | --- |
| `left_loop1` | `left_front_link` | `left_rear_child2_link` |
| `left_loop2` | `left_front_child1_link` | `left_rear_child3_link` |
| `right_loop1` | `right_front_link` | `right_rear_child2_link` |
| `right_loop2` | `right_front_child1_link` | `right_rear_child3_link` |

每处使用一对同轴 site 和 `<equality><connect .../></equality>`。树结构中的转动关节已经限制腿部平面运动，所以闭合点约束可形成所需的平面铰接，无需 weld 锁死相对转动。四个 connect 共 12 个标量约束，其雅可比在初始姿态的秩为 8；模型的 22 个广义速度扣除闭链约束后，有 14 个独立速度自由度：浮动机身 6、腿部 4、车轮 2、云台 2。

生成脚本的 `LOOP_PINS` 是 STL 圆孔边界拟合得到的后支链局部 X/Z 坐标（米）：

| 侧别 | `rear_child2_link` 闭合孔 | `rear_child3_link` 末端孔 |
| --- | --- | --- |
| 左 | `(-0.09627728, 0.09463449)` | `(-0.09336046, -0.02480701)` |
| 右 | `(-0.11299873, 0.07386668)` | `(-0.08881104, -0.03800210)` |

轴向 Y 坐标选在后连杆质心所在平面；另一端 site 由 URDF 装配变换求得，初始位置严格重合。这些位置与前支链独立拟合的销孔轴一致至约 2.1 微米，右侧微小差异来自 URDF 的 `right_rear_child3_joint` 原点截断精度。更改几何后，需要重新确定销孔位置。

## 力矩接口

`data.ctrl` 顺序如下，`gear=1`，单位 N·m。正方向遵循 URDF 关节轴，不保证左右相同指令对应相同物理转向。

| 索引 | 电机 | 关节局部轴 |
| --- | --- | --- |
| 0 | `left_front_motor` | `+Y` |
| 1 | `left_rear_motor` | `+Y` |
| 2 | `right_front_motor` | `-Y` |
| 3 | `right_rear_motor` | `-Y` |
| 4 | `left_wheel_motor` | `-Y` |
| 5 | `right_wheel_motor` | `+Y` |
| 6 | `yaw_motor` | `+Z` |
| 7 | `pitch_motor` | `-Y` |

被动连杆关节没有电机。URDF 将一个关节误命名为 `right_rear_child2_link`，MJCF 中统一为 `right_rear_child2_joint`，刚体名不变。URDF 的 continuous 关节保持不限位，未虚构机械限位或电机最大力矩。提供八路电机关节位置/速度传感器和机身 IMU site 的四元数、角速度、加速度输出。

## 仿真假设与待标定数据

- **yaw 质量和惯量待标定**：原 URDF 全为 0。估算时在内存中合并近重合顶点、去除重复/退化面并修正面朝向，再按均匀密度 `2700 kg/m³` 积分。估算质量为 `1.229551 kg`，总质量约 `18.148293 kg`。网格仍非水密，且实际装配材料并不均匀，因此这些只是运行仿真的近似参数；标定后应替换 `yaw_link/inertial`。生成脚本重跑会覆盖 XML。
- **碰撞**：保留机器人与地面接触，关闭机器人内部自碰撞。车轮使用 STL 径向最大距离和轴向包围范围得到的圆柱，半径约 `0.06 m`；其他刚体使用网格凸包碰撞，凹形部件的接触属于近似。显示 group 3 可查看轮子碰撞体和闭合 site。
- **数值参数**：时间步 `0.001 s`、Newton 求解器、implicitfast 积分器；闭链 `solref="0.003 1"`。关节黏性阻尼 `0.02 N·m·s/rad`，地面滑动摩擦系数 `0.8`，均为仿真初值，非实测参数。
- **资源**：`base_link.STL` 有 963887 个三角面，超出当前 MuJoCo STL 解码器的 200000 面限制。生成的 `base_link.obj` 保留三角几何，XML 引用 OBJ；原 STL 和 URDF 不变。其余网格继续使用原 STL。所有刚体显式指定惯量，不从碰撞凸包推断惯量。

## 重新生成与验证

```bash
.venv/bin/mjpython tools/build_mjcf.py
.venv/bin/mjpython tools/check_mjcf.py
```

验证脚本核对 URDF 质量、惯量、关节轴以及闭链约束秩，并运行三组各 5 秒的仿真。MuJoCo 3.14.0 的结果：

| 工况 | 最大闭合点误差 |
| --- | --- |
| 自由机身、零力矩、落地 | 0.295 mm |
| 固定机身、关节力矩激励 | 0.218 mm |
| 自由机身、力矩激励、落地 | 0.323 mm |

初始闭合误差低于 `3e-16 m`，三组仿真均无 MuJoCo 警告或非有限状态。这些是模型运行检查，不代表平衡控制性能或实机参数已验证。

MJCF 约束语义参见 [MuJoCo connect 文档](https://mujoco.readthedocs.io/en/latest/XMLreference.html#equality-connect)。
