# CHANGELOG — v3.20260710

## 概述

v3 在 v2 的 RF成本表面 + A* 管线基础上，新增 IPSO-SA 和 DBO 两种群体智能路径规划算法，
形成三算法（A* vs IPSO-SA vs DBO）对比框架。

## 新增

### 通用框架
- `src/base_planner.py`: 群体智能路径规划基类
  - 路径编码: M 个中间路标点 (lat, lon), 首尾固定
  - 适应度 = 平均成本 + 硬约束惩罚 + 回退惩罚 + 长度惩罚
  - 种群初始化: 直线 + A*热启动 + 走廊随机
  - 硬约束修复: 落入禁区 → 搜索最近可行点 (半径30像素)
  - 路标→路径: 等距插值 (30m) → 平滑 → 质量门控

### IPSO-SA 规划器
- `src/ipso_sa_planner.py`: 改进粒子群 + 模拟退火
  - PSO: 惯性权重线性递减 (0.9→0.4), c1=c2=2.0
  - SA: 温度指数衰减 (alpha=0.95), 以 exp(-Δf/T) 接受劣解
  - 申报书基准对照组

### DBO 规划器
- `src/dbo_planner.py`: 蜣螂优化算法 (Xue & Shen, 2022)
  - 四行为组: 滚球(探索) + 育雏(局部开发) + 觅食(精细搜索) + 偷窃(多样性)
  - 区域半径随迭代线性收缩

### 三算法对比
- `validate_v3.py`: 统一入口, 复用 v2 预处理+成本建模
  - Step1: 选择 10 条测试线路
  - Step2: 公共预处理 (v2 Phase 1-3)
  - Step3: 逐条三算法验证
  - Step4: 汇总对比 + 统计

## 技术参数

| 参数 | IPSO-SA | DBO |
|------|---------|-----|
| 种群规模 | 30 | 30 |
| 最大迭代 | 200 | 200 |
| 走廊宽度 | 15km | 15km |
| 路标点密度 | 1.5/km | 1.5/km |
| 早停代数 | 50 | 50 |

## 兼容性

- 复用 v2 的 preprocessing, cost_model, output 全部模块
- 复用 shared/data_acquisition.py
- 输出格式与 v2 一致 (per-case JSON + SHP + GeoJSON)
