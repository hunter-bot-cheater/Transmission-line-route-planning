#!/bin/bash
# 双终端工作流 — 终端 A：架构师 + 测试员
# 职责：需求分析 / 方案设计（planning）/ 测试验收（testing）/ 版本指派
# 用法：./tools/start_architect.sh
# 模型可覆盖（不指定则用 claude 当前默认模型，支持任意模型）：ARCHITECT_MODEL=<任意模型> ./tools/start_architect.sh
set -e
cd "$(dirname "$0")/.."

# 模型名不做硬编码：ARCHITECT_MODEL 未设置时用 claude 默认模型（任何已配置模型均可跑）
MODEL="${ARCHITECT_MODEL:-}"
PROMPT=".workflow/architect_prompt.md"

echo "═══════════════════════════════════════════════"
echo "  🛡️  架构师 + 测试员 终端（模型: ${MODEL:-claude 默认}）"
echo "═══════════════════════════════════════════════"
echo "  · 你负责 planning（写方案）和 testing（跑测试验收）"
echo "  · 启动后 claude 自动加载角色提示词，并读 status.json 判断当前阶段"
echo "  · 若当前是 coding 阶段：会提示你去工程师终端，不要越权动手"
echo "  · 干完活按提示词更新 status.json 并告诉你下一步切到哪个终端"
echo "  · 新任务：把 .workflow/task.md 改成需求，并执行下面命令置为 planning："
echo "      sed -i 's/\"phase\": *\"[^\"]*\"/\"phase\": \"planning\"/' .workflow/status.json"
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
