# 维护范围

- 只维护 demo 平地 locomotion 的发布、续训与评估；训练和部署共享 MJCF、观测协议和网络动作处理。
- `policies/` 仅保留当前正式 ONNX、匹配的续训 PT、元数据和发布评估，不保留旧模型或实验目录。
- `function/locomotion/` 只放任务、指令、训练与评估；修改任务同步更新 registration.py 和 README。
- 训练作业、轨迹和临时输出放在被 Git 忽略的 outputs/ 或仓库外。源码快照由 source.py 收集。
- 发布状态和性能验收分别记录，保留当前版本的已知限制。
