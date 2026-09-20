# 汽车底部车身面板冲击数据集中文说明

本汽车底部车身面板冲击数据集（Automotive Underbody Panel
Impact Dataset）包含三种汽车
结构几何上的独立冲击有限元仿真。每种几何包含 500 个约束 LHS 工况，
每个工况保存 17 个对齐时刻的完整节点三维位移场和壳单元
von Mises 等效应力场。

## 参数与单位

仿真采用 tonne--mm--s--N 一致单位制：坐标和位移为 mm，时间为 s，
速度为 mm/s，质量为 tonne，密度为 tonne/mm^3，应力和弹性模量为 MPa。

- 冲击速度范围为 1732.05--5196.15 mm/s。
- 质量比 `mu` 范围为 0.75--1.25，并满足
  `impactor_mass = 0.01 tonne * mu`、
  `impactor_density = 5.205e-5 tonne/mm^3 * mu`。
- `theta` 为相对全局 +Z 的极角，范围 0--15 度；`phi` 为全局 XY
  平面内从 +X 指向 +Y 的方位角，范围 0--360 度。
- 速度向量为
  `v = s [sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)]`。
- E 和 nu 是刚性冲击球的材料参数，三个离散组合分别为
  `(70000 MPa, 0.33)`、`(110000 MPa, 0.34)` 和
  `(210000 MPa, 0.30)`。

冲击位置不是三个连续坐标的独立采样。候选位置是距离面板拓扑外边界
至少 80 mm 的壳单元质心；两个归一化 LHS 位置坐标映射到最近且未使用的
候选质心，Z 坐标取该质心的实际高度。

## 仿真设置

冲击体是半径 12.5 mm 的刚性球壳，初始间隙 5.0 mm，采用 ELFORM=2、
SHRF=0.833333、NIP=3、厚度 0.1 mm 和 `*MAT_RIGID`。面板拓扑外边界
节点的六个自由度全部固定。冲击接触采用
`*CONTACT_AUTOMATIC_SURFACE_TO_SURFACE_ID`，静、动摩擦系数均为 0.15；
全局 +Z 方向施加 9810 mm/s^2 的体加速度。仿真终止时间为 0.03 s。

求解使用 ANSYS v221 中的 `lsdyna_sp.exe`，对应 LS-DYNA SMP single
precision R12，运行参数为 `ncpu=8`、`memory=400m`。面板材料和壳截面
沿用上游 Version 3 模型；完整 keyword 卡不包含在本紧凑数据发布中。

## LHS 与时间采样

LHS 包含位置 X/Y、速度大小、质量比、theta、phi 和材料类别七个维度，
并从 128 个候选设计中选择归一化最小样本间距最大的设计。三种几何共享
参数范围和仿真规则，但使用不同的候选网格位置和随机种子；其中
`floorfrontdriver` 是分阶段构建的设计，而不是一次生成的单批设计。

D3PLOT 名义输出间隔为 0.0002 s。紧凑转换每十个原始状态保留一帧，
并额外加入最终状态；全部公开工况的索引均为
`[0,10,20,...,150,151]`。前 16 帧名义间隔为 0.002 s，最后一帧可能与
前一帧非常接近，实际计算应使用每个样本保存的 `time`。

峰值时间 `t*` 只在 17 个保留状态中对有效节点的位移模长取最大值，
因此是离散时间，不是连续求解轨迹的精确峰值；应力标签取相同状态。

## 数据规模

| 几何 | case数 | 节点数 | 壳单元数 | 位移张量 | 应力张量 |
|---|---:|---:|---:|---|---|
| `floorfrontdriver` | 500 | 7,408 | 7,374 | `[7408,17,3]` | `[7374,17]` |
| `floorfrontR` | 500 | 12,011 | 12,055 | `[12011,17,3]` | `[12055,17]` |
| `trunkfloor` | 500 | 14,440 | 14,589 | `[14440,17,3]` | `[14589,17]` |

三种几何是彼此独立的数据。相同 case ID 不代表同一次物理仿真，
不能作为跨几何物理配对样本。

## 应力定义

`effective_stress` 是通过 LS-PrePost 导出的壳单元 von Mises 等效应力。
LS-PrePost 的 `etime 9` 对应 `Effective Stress (v-m), ip#max`：对每个壳单元
和每个保留时刻，在全部厚度积分点的 von Mises 应力中取最大值。数据不保留
取得最大值的积分点编号。应力单位为 MPa，张量形状为 `[Ne,17]`。

## 使用方法

本仓库按工况保存独立 `.pt` 文件，不需要 Git LFS 或 ZIP 解包。
数据内容与 1.1.1 版本逐字节一致，发布布局为 `git-cases-v1`。
`manifest.csv` 的 `path` 是当前路径，`shard`/`member` 仅记录原归档来源。
读取方法：

```bash
python scripts/load_case.py --dataset-root . \
  --geometry floorfrontdriver --case case001
```

全量验证：

```bash
python scripts/validate_dataset.py --dataset-root . --verify-checksums
```

本发布包中的 `floorfrontR` 数据经过完整质量核查；不要与更早的内部构建
混用。本仓库采用 MIT License，详细来源、字段和局限请参阅英文
`README.md`、`DATASHEET.md`、`metadata/SIMULATION_PROTOCOL.md`、
`metadata/LHS_DESIGN.md`、`metadata/TEMPORAL_SAMPLING.md` 和
`THIRD_PARTY_NOTICES.md`。
