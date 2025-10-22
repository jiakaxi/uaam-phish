#!/usr/bin/env bash
# Shell script to install Git hooks
# 使用方法: ./.github/hooks/install-hooks.sh

set -euo pipefail

echo "🔧 安装 Git Hooks..."

# 检查是否在 Git 仓库中
if [ ! -d ".git" ]; then
    echo "❌ 错误: 不在 Git 仓库根目录"
    exit 1
fi

# 复制 pre-commit hook
SOURCE_HOOK=".github/hooks/pre-commit"
TARGET_HOOK=".git/hooks/pre-commit"

if [ -f "$SOURCE_HOOK" ]; then
    cp "$SOURCE_HOOK" "$TARGET_HOOK"
    chmod +x "$TARGET_HOOK"
    echo "✅ pre-commit hook 已安装到 .git/hooks/"
else
    echo "❌ 错误: 找不到 $SOURCE_HOOK"
    exit 1
fi

echo ""
echo "🎉 Git Hooks 安装完成!"
echo "现在每次 commit 前会自动运行:"
echo "  - ruff check (代码检查)"
echo "  - black --check (格式检查)"
echo "  - pytest (运行测试)"
