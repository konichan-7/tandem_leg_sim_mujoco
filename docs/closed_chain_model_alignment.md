# 模型与闭链约束

原始 MJCF 保存在 `MJCF/demo/demo_original.xml`。训练、发布 policy 和传统控制入口使用 `MJCF/demo/demo.xml`。运行模型仅修正四个前腿闭链 site 的局部 y 坐标；质量、质心、惯量、关节轴、阻尼、左右结构差异及四组 equality 保持原样。加载器不在运行时修改锚点。

| site | 原始 y，m | 运行 y，m |
|---|---:|---:|
| left_front_site1 | 0.0278 | 0.0188 |
| left_front_site2 | 0 | −0.0118 |
| right_front_site1 | −0.0278 | −0.0268 |
| right_front_site2 | 0 | +0.0118 |

原始点对存在无法通过绕 y 轴的腿关节运动消除的横向间隙。闭链约束的广义力为

\[
C(q)=p_1(q)-p_2(q),\qquad \dot C=J_C(q)v,\qquad Q_{eq}=J_C^{\mathsf T}\lambda.
\]

若约束点之间有固定几何偏差，整体旋转会产生

\[
\delta C=\delta\theta\times C,\qquad \tau_{eq}=C\times\lambda.
\]

因此错误锚点会带来非物理的约束力矩。修正后的锚点对齐历史训练的有效坐标，不以左右对称为目标。

2026-10-07几何检查：与历史有效锚点的最大坐标差1.735e−17m，标称闭链间隙1.19e−9m，采样最大间隙4.493e−8m。原始模型初态约−2.74Nm的寄生偏航约束力矩被消除。当前发布版的运动性能单独记录在 [policy说明](../tandem_mjlab/policies/README.md)，几何修正不等同于策略验收通过。

原始模型SHA256：`41f336f33b7a0d0163dba6621166d4c75a2bb08a117d87e411b652b050fe9963`。

运行模型SHA256：`b884291eba8f9a87205a0a42c1d1117e49709b14e3b6be66de86f0c736db5c8e`。
