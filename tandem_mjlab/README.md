# Demo 平地 policy

正式版本为 `policies/locomotion.onnx`，配套 `policies/locomotion.pt` 用于续训。当前版本包含有界差速修正和静止腿动作平滑，协议统一为 27 维观测、五帧历史、6 维动作。旧模型和阶段实验入口已移除。

## 运行

在仓库根目录安装和运行：

```bash
.venv/bin/python -m pip install -e ./tandem_mjlab
.venv/bin/mjpython -m tandem_mjlab.sim2sim --keyboard
```

macOS 按住 **1/2** 前进/后退、**3/4** 右转/左转，可组合；松开减速归零。默认速度 1 m/s、转向 2 rad/s，指令斜坡为 1.5 m/s² 和 6 rad/s²。终端 Ctrl+C 退出。无窗口评估可使用普通 Python。

```bash
.venv/bin/mjpython -m tandem_mjlab.sim2sim --duration 20 --velocity 1 --ramp
.venv/bin/mjpython -m tandem_mjlab.function.locomotion.evaluate --output outputs/evaluation --workers 4
```

评估默认包含 30 个静止、直行、自转、组合转弯和指令切换工况。`--final` 增加 60s 静止、4个60s直行和5个20ms动作延迟工况；`--baseline <results.json>` 检查前后响应回归。生成的轨迹、报告和作业目录不进入 Git。

## 续训与导出

CUDA 训练环境安装：

```bash
python -m pip install -e './tandem_mjlab[train]'
python -m tandem_mjlab.function.locomotion.train --output outputs/finetune --num-envs 65536 --iterations 20 --mini-batches 8
python -m tandem_mjlab.export --checkpoint outputs/finetune/model_19.pt --output outputs/finetune/policy.onnx
```

续训默认加载正式 PT，冻结原动作网络、历史编码器与静止平滑系数，仅更新差速修正网络和 critic，重新建立优化器。`--checkpoint` 指定同协议的新 checkpoint；`--randomize` 开启小范围动力学随机化和0–20ms动作延迟。训练完成后需单独评估，训练入口不会替换发布文件。

65536 是本次 RTX4090 24GB 的实测规模，其他机器按显存调整 `--num-envs` 与 `--mini-batches`。本机 macOS 用于 native MuJoCo/ONNX 部署，CUDA 训练在 Linux 服务器验证。

## 代码边界

| 文件 | 职责 |
|---|---|
| `model.py`、`nominal_state.json` | MJCF、标称姿态、PD、限幅和仿真参数 |
| `policy.py` | 历史编码器、动作网络、差速修正、静止平滑与 ONNX 网络 |
| `function/locomotion/task.py`、`commands.py` | 环境、奖励、终止条件和指令采样 |
| `function/locomotion/train.py`、`evaluate.py` | 续训与 native 轨迹验收 |
| `sim2sim.py`、`keyboard.py` | 部署、键盘和轨迹记录 |
| `export.py`、`runner.py`、`source.py` | 导出、保存和训练源码快照 |
| `registration.py` | 唯一任务 `Mjlab-Tandem-WheelLeg-Flat` |
| `policies/` | 唯一发布版本及评估证据 |

模型和闭链修复见 [模型说明](../docs/closed_chain_model_alignment.md)，策略协议与已知限制见 [发布说明](policies/README.md)。
