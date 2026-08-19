# DEVELOPMENT.md — 项目交接文档（工作流公共记忆）

> 供双 Agent 工作流（.workflow/）中的架构师/工程师启动时读取。
> 修改业务代码前必读；本文件随项目演进持续更新。

## 1. 项目是什么

台湾输电线路智能路径规划系统（大创项目）。用 GIS 空间分析 + 随机森林/CNN 成本建模 + 启发式/AI 搜索，
为台湾地区自动规划输电线路路径。管线：数据获取 → 预处理 → 成本建模 → 路径搜索 → 输出可视化。

## 2. 目录与版本演进

| 目录 | 定位 | 说明 |
|------|------|------|
| `v1_20260525/` | 宽松约束原型 | 6 条线路，单一硬约束 |
| `v2_20260525/` | 严格质量门控（稳定版） | 7 项门控，10/10 通过 |
| `v3_20260710/` | 三算法对比框架 | A* / IPSO-SA / DBO，RF 成本面 |
| `v4_20260718/` | CNN 增强版 | U-Net 成本预测 + MLP 启发式 |
| `V5_final/` | **当前交付主线** | VIN-Grad / PPO / A* 三方法 + 真值验收 ≤8% |
| `shared/` | 公共模块 | data_acquisition.py 等 |
| `docs/` | 文档/结果汇总 | changelog/ 与 proposals/ 为本地文件（git 排除） |
| `data/` | 数据 | git 排除 |

## 3. 环境事实

- 项目 Python 3.13；依赖：numpy / scipy / geopandas / rasterio / scikit-learn / matplotlib / torch
- 本机解释器参考：
  - 系统 Python：`D:\python\python.exe`（3.12.5，已装 numpy 1.26.4 / pytest 9.1.1）
  - 受管 Python：`C:\Users\86133\.workbuddy\binaries\python\versions\3.13.12\python.exe`
- **语法编译/测试**可用任意解释器（py_compile 不依赖第三方包）；
  **运行重脚本**（ai_path_planning*.py）需完整依赖 + 数据文件 + GPU（可选，RTX 4060）
- 运行完整管线需台湾 DEM/SHP 数据（在 `data/` 或 D:\大创\data），工作区不一定有 → 重脚本运行验证由用户在有数据的机器上做

## 4. 测试资源

- 无 pytest 测试套件（研究项目，未建 test_*.py）
- `./tools/test_backend.sh`：py_compile 全量语法 + shared 冒烟（可 PYTHON= 指定解释器）
- `./tools/run_all_tests.sh`：全量汇总（test_frontend.sh / test_typescript.sh 未实现，自动跳过 N/A）
- 冒烟 import 说明：`shared/data_acquisition.py` 顶层 `import config` 依赖项目外 config 模块，
  import 失败多为环境问题而非代码问题

## 5. 版本规范

- **无版本号常量**（非 Web 服务）；版本 = git 提交 + docs/changelog/ 里程碑
- 版本标注（脚本头 docstring 的 "vX.2026xxxx"）只由**用户**指派，工程师/架构师不得擅自改
- ⚠️ 已知版本标注混乱（交付物历史问题）：`V5_final/README.md` 是旧 v3 文档；
  `V5_final/scripts/ai_path_planning_v5.py` 文件头自称为 `ai_path_planning_v4.py`/"v4 升级版"。
  V5 实质 = v4 三方法代码 + 真值验收框架打包。读到 v4 字样不要误判版本。

## 6. 已知缺陷 / 注意事项

- ✅ `V5_final/scripts/ai_path_planning.py` 的语法错误**已于 2026-08-19 修复**（原为远端交付物
  自带：`export_route` 的 GeoJSON 回退分支在 `"features": [{` 处被截断，已补全为完整
  FeatureCollection + json.dump 写出）。该文件是旧版脚本副本，非 V5 主脚本
  （主脚本 `ai_path_planning_v5.py` 不受影响）。
- 版本标注混乱见上节；处理前与用户确认。
- CLAUDE.md 项目规则：代码 snake_case / 类型标注 / Google docstring / 4 空格缩进；
  commit 后必须更新 docs/changelog/ 和桌面备份；改动前确认「本地修复 vs 推送 GitHub」。

## 7. 工作流约定（.workflow/）

- 状态机：`planning → coding → testing → done/failed`，phase 记录在 .workflow/status.json
- 三种运行模式（任选）：
  1. **双终端手动**：`./tools/start_architect.sh` + `./tools/start_engineer.sh`
  2. **单会话直跑**：一个 agent 全程维护状态机
  3. **全自动协调器**：`./tools/orchestrator_auto.example.sh`（参数/约束见脚本头注释）
- 时间戳用 `date '+%Y-%m-%d %H:%M'`；迭代上限 status.json `max_iterations`（默认 3）
- 运行期文件（status.json / task.md / plan.md / test_report.md）已 gitignore，不提交；
  提示词与 tools/ 脚本建议提交，保证团队成员/CI 拿到一致规则
