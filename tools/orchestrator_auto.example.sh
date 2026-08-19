#!/bin/bash
# orchestrator_auto.example.sh — 全自动单进程协调器（第三模式 · 示例）
# =============================================================================
# 与 start_architect.sh / start_engineer.sh 的双终端手动模式互补：
# 一个进程内驱动 架构师→工程师→测试 闭环，适合小任务 / CI / 不想开两个终端时。
#
# 用法：
#   ./tools/orchestrator_auto.example.sh
#
# 环境变量可覆盖（全部可选，均有默认值）：
#   ORCH_MODEL=<模型名>        指定模型（默认 claude 当前默认模型）
#   ORCH_MAX_TURNS=<N>         单步 agent 最大轮数（默认 60）
#   ORCH_STEP_TIMEOUT=<秒>     单步超时（默认 900s；macOS 无 timeout 命令时可设 0 关闭）
#   ORCH_TEST_CMD=<命令>       测试命令（默认 ./tools/run_all_tests.sh）
#   ORCH_SKIP_TEST=<1>         跳过测试直接按 status.test_passed 收尾（调试用）
#
# 设计要点（防"静默失败"）：
#   * 每步 agent 执行后核对 status.json 的 phase 是否真的推进，没推进 = 失败即停
#   * 每轮结束打印 status 快照；失败时打印 test_report 尾部关键行，绝不无声退出
#   * 迭代上限读 status.json 的 max_iterations，超限置 phase=failed 并退出 1
#   * 协调器只负责"调度 + 校验 + 汇报"，不替 agent 做任何规划/编码/测试决策
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || { echo "❌ 找不到项目根目录（请在项目根运行 ./tools/orchestrator_auto.example.sh）" >&2; exit 1; }

STATUS=".workflow/status.json"
REPORT=".workflow/test_report.md"
ARCH_PROMPT=".workflow/architect_prompt.md"
ENG_PROMPT=".workflow/engineer_prompt.md"

# ---- 参数（环境变量覆盖） ----
ORCH_MODEL="${ORCH_MODEL:-}"
ORCH_MAX_TURNS="${ORCH_MAX_TURNS:-60}"
ORCH_STEP_TIMEOUT="${ORCH_STEP_TIMEOUT:-900}"
ORCH_TEST_CMD="${ORCH_TEST_CMD:-./tools/run_all_tests.sh}"
ORCH_SKIP_TEST="${ORCH_SKIP_TEST:-0}"

# ---- JSON 读取（python3 优先，回退 grep/sed，避免依赖 jq） ----
json_get() { # $1=file  $2=key
  local f="$1" k="$2"
  if command -v python3 >/dev/null 2>&1; then
    python3 -c 'import json,sys
try:
    v=json.load(open(sys.argv[1]))[sys.argv[2]]
    print(str(v).lower() if isinstance(v,bool) else v)
except Exception: sys.exit(1)' "$f" "$k" 2>/dev/null && return 0
  fi
  # grep 回退（无 python3 时；支持带引号字符串与裸值 bool/数字）
  grep -o "\"$k\"[[:space:]]*:[[:space:]]*[^,}]*" "$f" 2>/dev/null | head -1 | sed -E 's/^[^:]*:[[:space:]]*//; s/^"//; s/"$//; s/[[:space:]]*$//'
}

status_snapshot() {
  echo "── status.json 快照 ──"
  cat "$STATUS" 2>/dev/null || echo "(status.json 不可读!)"
}

die() { echo "❌ $1" >&2; status_snapshot; [ -f "$REPORT" ] && { echo "── test_report.md 尾部（关键行）──"; tail -n 40 "$REPORT"; }; exit 1; }

run_agent() { # $1=角色名  $2=提示词路径
  local role="$1" prompt="$2"
  [ -f "$prompt" ] || die "找不到提示词 $prompt"
  echo "── [$role] 开始（模型: ${ORCH_MODEL:-默认}，max_turns=$ORCH_MAX_TURNS）──"
  local cmd
  if [ -n "$ORCH_MODEL" ]; then
    cmd=(claude --model "$ORCH_MODEL" -p "$(cat "$prompt")" --max-turns "$ORCH_MAX_TURNS")
  else
    cmd=(claude -p "$(cat "$prompt")" --max-turns "$ORCH_MAX_TURNS")
  fi
  if [ "$ORCH_STEP_TIMEOUT" -gt 0 ] && command -v timeout >/dev/null 2>&1; then
    timeout "$ORCH_STEP_TIMEOUT" "${cmd[@]}" || { [ $? -eq 124 ] && die "[$role] 超时(>${ORCH_STEP_TIMEOUT}s)"; die "[$role] agent 调用失败"; }
  else
    "${cmd[@]}" || die "[$role] agent 调用失败（超时控制已关闭）"
  fi
  echo "── [$role] 结束 ──"
}

[ -f "$STATUS" ] || die "找不到 $STATUS（请确认 .workflow/ 已就位）"

phase="$(json_get "$STATUS" phase)";   [ -z "$phase" ]  && die "status.json 缺少 phase 字段"
iteration="$(json_get "$STATUS" iteration)"; iteration="${iteration:-1}"
max_iter="$(json_get "$STATUS" max_iterations)"; max_iter="${max_iter:-3}"

echo "══ 协调器启动：phase=$phase  iteration=$iteration/$max_iter  ══"
[ "$phase" = "done" ] && { echo "✅ 任务已是 done 状态，无需执行"; exit 0; }
[ "$phase" = "failed" ] && { echo "❌ 任务已是 failed 状态，需人工介入"; exit 1; }

for round in $(seq 1 "$max_iter"); do
  echo ""
  echo "════ 第 $round 轮（phase=$phase）════"
  case "$phase" in
    planning)
      run_agent "架构师·方案" "$ARCH_PROMPT"
      new_phase="$(json_get "$STATUS" phase)"
      [ "$new_phase" != "coding" ] && die "架构师结束后 phase 应为 coding，实际=$new_phase（状态未推进，疑似 agent 未按提示词更新）"
      phase="$new_phase" ;;
    coding)
      run_agent "工程师·编码" "$ENG_PROMPT"
      new_phase="$(json_get "$STATUS" phase)"
      [ "$new_phase" != "testing" ] && die "工程师结束后 phase 应为 testing，实际=$new_phase（状态未推进）"
      phase="$new_phase" ;;
    testing)
      echo "── [测试] $ORCH_TEST_CMD ──"
      test_rc=0
      if [ "$ORCH_SKIP_TEST" = "1" ]; then
        echo "(ORCH_SKIP_TEST=1，跳过实际测试)"
      else
        bash "$ORCH_TEST_CMD"; test_rc=$?
      fi
      tp="$(json_get "$STATUS" test_passed)"
      if [ "$test_rc" -eq 0 ] && [ "$tp" = "true" ]; then
        echo "✅ 测试通过，任务完成"
        echo "── 置 phase=done ──"
        sed -i 's/"phase"[[:space:]]*:[[:space:]]*"[^"]*"/"phase": "done"/' "$STATUS"
        exit 0
      fi
      # 未通过：要么回 coding 修，要么超限 failed
      if [ "$iteration" -ge "$max_iter" ]; then
        sed -i 's/"phase"[[:space:]]*:[[:space:]]*"[^"]*"/"phase": "failed"/' "$STATUS"
        die "达到迭代上限 $max_iter（测试仍失败），已置 phase=failed，需人工介入"
      fi
      iteration=$((iteration + 1))
      echo "── 测试未通过，第 $iteration/$max_iter 轮，置回 coding ──"
      # 注意：status 里 iteration 可能已被架构师在 testing 阶段递增，这里统一以协调器为准覆盖
      sed -i -e 's/"phase"[[:space:]]*:[[:space:]]*"[^"]*"/"phase": "coding"/' \
             -e "s/\"iteration\"[[:space:]]*:[[:space:]]*[0-9]*/\"iteration\": $iteration/" "$STATUS"
      phase="coding" ;;
    done)
      echo "✅ 任务已 done"; exit 0 ;;
    failed)
      echo "❌ 任务已 failed，需人工介入"; exit 1 ;;
    *)
      die "未知 phase=$phase" ;;
  esac
  status_snapshot
done

die "循环耗尽（$max_iter 轮）仍未收敛，请人工介入"
