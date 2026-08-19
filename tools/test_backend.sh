#!/bin/bash
# test_backend.sh — Transmission-line-route-planning 后端（Python）测试
# =============================================================================
# 本项目无 pytest 测试套件（纯研究项目），测试策略分两层：
#   1) py_compile 全量语法编译（硬性）：项目内所有 *.py 必须可编译
#   2) 冒烟 import（尽力而为）：shared/data_acquisition.py 依赖项目外 config 模块，
#      环境不满足时输出 WARN 但不 FATAL（属环境问题，非代码问题）
# 用法： ./tools/test_backend.sh
# 可用环境变量：PYTHON=<解释器路径>（默认自动探测：python3 > python）
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || { echo "❌ 找不到项目根目录"; exit 1; }

if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY=python
fi

echo "== 测试解释器: $PY（可用 PYTHON= 覆盖） =="
"$PY" --version 2>&1 || { echo "❌ 找不到 Python 解释器"; exit 1; }

PASS=0
FAIL=0

echo ""
echo "── [1/3] py_compile 全量语法编译 ──"
# 排除 .git / __pycache__
mapfile -t PYS < <(find . -name "*.py" -not -path "./.git/*" -not -path "*__pycache__*" 2>/dev/null | sort)
for f in "${PYS[@]}"; do
    if "$PY" -m py_compile "$f" 2>/dev/null; then
        PASS=$((PASS + 1))
    else
        echo "  ❌ 语法错误: $f"
        FAIL=$((FAIL + 1))
    fi
done
echo "  编译通过: $PASS / $((PASS + FAIL))"

echo ""
echo "── [2/3] 冒烟 import：shared.data_acquisition ──"
if "$PY" -c "import sys; sys.path.insert(0, '.'); import shared.data_acquisition" 2>/tmp/tlrp_import_err.txt; then
    echo "  ✅ shared.data_acquisition import OK"
else
    echo "  ⚠️  import 失败（多为环境依赖，非代码问题，见下）："
    tail -2 /tmp/tlrp_import_err.txt | sed 's/^/     /'
fi
rm -f /tmp/tlrp_import_err.txt

echo ""
echo "── [3/3] 汇总 ──"
echo "通过: $PASS  失败: $FAIL"
echo ""
[ "$FAIL" -eq 0 ] && { echo "✅ 语法测试全部通过"; exit 0; } || { echo "❌ 存在语法错误，请修复后重跑"; exit 1; }
