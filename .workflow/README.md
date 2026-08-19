# 双 Agent 工作流 — Transmission-line-route-planning 适配版

模板来源：`桌面/agent-workflow-template/`（通用可移植版，含完整使用说明）。
本目录是应用到本项目的适配结果，**本项目为纯 Python 研究项目，无前端**。

## 占位符对照（本项目已填）

| 占位符 | 本项目值 |
|---|---|
| `<PROJECT_NAME>` | Transmission-line-route-planning |
| `<PROJECT_ROOT>` | 项目根（.workflow/ 上一级） |
| `<BACKEND_STACK>` / `<BACKEND_DIR>` | Python 3.13 / `shared/` + 各版本 `scripts/` |
| `<FRONTEND_STACK>` / `<FRONTEND_DIR>` | 无前端（已删除相关章节） |
| 测试入口 | `./tools/test_backend.sh`（py_compile + 冒烟） |
| 版本常量 | 无（已改为 changelog 里程碑治理） |

## 与模板的差异（适配点）

1. **无前端**：engineer/architect 提示词删除了前端目录与测试；
   `run_all_tests.sh` 对未实现的 `test_frontend.sh` / `test_typescript.sh` 自动跳过（N/A）。
2. **测试策略**：无 pytest 套件，`test_backend.sh` = py_compile 全量语法 + shared 冒烟（尽力而为）。
3. **版本治理**：无版本号常量 → 改为 git + docs/changelog/ 里程碑，禁止擅自写版本标注。
4. **已修复历史缺陷**：`V5_final/scripts/ai_path_planning.py` 远端自带语法错误已于
   2026-08-19 修复（export_route GeoJSON 分支补全），现 48/48 全量编译通过。
5. **新成员**：`tools/orchestrator_auto.example.sh` 全自动协调器（第三模式）。

## 三模式速查

- 双终端：`./tools/start_architect.sh` 和 `./tools/start_engineer.sh`（各开一个终端）
- 单会话：一个 agent 全程维护 status.json 闭环
- 全自动：`ORCH_SKIP_TEST=0 ./tools/orchestrator_auto.example.sh`（会真实调 claude，谨慎）

## 首次使用

1. 把本次需求写进 `.workflow/task.md`
2. 将 status.json 置为 planning（`sed -i 's/"phase": *"[^"]*"/"phase": "planning"/' .workflow/status.json`）
3. 按所选模式启动

详细约定见 `DEVELOPMENT.md`（架构/测试/版本/已知缺陷）与模板 `README.md`。
