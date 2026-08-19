#!/bin/bash
# 双终端工作流 — 终端 B：工程师
# 职责：按方案写代码 / 按测试报告修 bug（coding）
# 用法：./tools/start_engineer.sh
# 模型可覆盖（不指定则用 claude 当前默认模型，支持任意模型）：ENGINEER_MODEL=<任意模型> ./tools/start_engineer.sh
set -e
cd "$(dirname "$0")/.."

# 模型名不做硬编码：ENGINEER_MODEL 未设置时用 claude 默认模型（任何已配置模型均可跑）
MODEL="${ENGINEER_MODEL:-}"
PROMPT=".workflow/engineer_prompt.md"

echo "═══════════════════════════════════════════════"
echo "  🔧  工程师 终端（模型: ${MODEL:-claude 默认}）"
echo "═══════════════════════════════════════════════"
echo "  · 你负责 coding（按方案写代码、修 bug）"
echo "  · 启动后 claude 自动加载角色提示词，并读 status.json 判断当前阶段"
echo "  · 若当前不是 coding 阶段：会提示你去架构师终端，不要越权动手"
echo "  · 干完活按提示词更新 status.json 并告诉你下一步切到哪个终端"
echo "  · 注意：只有 status.json 的 phase=coding 时你才动手；否则等架构师交接"
echo ""

if [ ! -f "$PROMPT" ]; then
  echo "❌ 找不到 $PROMPT（请在项目根目录运行）" >&2
  exit 1
fi

if [ -n "$MODEL" ]; then
  exec claude --model "$MODEL" "$(cat "$PROMPT")"
else
  exec claude "$(cat "$PROMPT")"
fi
