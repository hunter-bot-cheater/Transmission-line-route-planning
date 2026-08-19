#!/bin/bash
# 一键跑全部测试套件（后端 pytest + 前端 vitest + TS 类型检查）
# 缺失的测试脚本（如纯后端项目无前端）自动跳过并标记 N/A，不算失败。

echo "===== 运行全部测试 ====="
echo ""

cd "$(dirname "$0")/.." || { echo "找不到项目根目录"; exit 1; }

PASS=0
FAIL=0
SKIP=0

run_test() {
    local name="$1"
    local script="$2"
    echo "--- $name ---"
    if [ ! -f "$script" ]; then
        echo "⏭️  跳过（未实现 $script）"
        SKIP=$((SKIP + 1))
        echo ""
        return
    fi
    if bash "$script"; then
        echo "✅ 通过"
        PASS=$((PASS + 1))
    else
        echo "❌ 失败"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# 后端单元测试（脚本内部 cd 到后端目录，才能 import 相应包）
run_test "后端 pytest" "tools/test_backend.sh"

# 前端单元测试（脚本内部 cd 到前端目录；纯后端项目可留空不实现）
run_test "前端 vitest" "tools/test_frontend.sh"

# 前端 TS 类型检查（vue-tsc --noEmit；无前端则留空不实现）
run_test "前端 TS 类型检查" "tools/test_typescript.sh"

echo "===== 测试汇总 ====="
echo "通过: $PASS"
echo "跳过: $SKIP"
echo "失败: $FAIL"
echo ""

[ $FAIL -eq 0 ] && exit 0 || exit 1
