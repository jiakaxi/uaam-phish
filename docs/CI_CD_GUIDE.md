# CI/CD 自动化流程指南

## 📋 概述

项目已配置完整的 CI/CD 流程，包括：
- ✅ 代码质量检查（Ruff + Black）
- ✅ 单元测试（Pytest）
- ✅ 配置验证（OmegaConf）
- ✅ 依赖安全检查（pip-audit）
- ✅ 自动代码格式化
- ✅ Pre-commit hooks

---

## 🔧 GitHub Actions Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**触发条件：**
- Push 到 `main` 或 `dev` 分支
- Pull Request 到 `main` 或 `dev` 分支

**包含的检查：**

#### a) 代码质量检查
```yaml
lint:
  - Ruff 检查（代码规范）
  - Black 格式检查
```

#### b) 单元测试
```yaml
test:
  - 多版本 Python 测试（3.9, 3.10, 3.11）
  - 代码覆盖率报告
  - 上传到 Codecov
```

#### c) 数据验证
```yaml
validate-data:
  - 验证数据 schema
  - 检查数据完整性
```

#### d) 配置验证
```yaml
validate-configs:
  - 验证所有 YAML 配置文件
  - 检查 Hydra 配置
```

#### e) 文档检查
```yaml
docs-check:
  - 检查 README.md
  - 检查必要文档存在性
```

#### f) 安全检查
```yaml
security:
  - pip-audit 依赖安全审计
  - 检查已知漏洞
```

---

### 2. 自动格式化 Workflow (`.github/workflows/auto-format.yml`)

**触发条件：**
- Pull Request 创建或更新

**功能：**
- 自动运行 Ruff 修复
- 自动运行 Black 格式化
- 自动提交格式化后的代码

**使用：**
1. 创建 Pull Request
2. GitHub Actions 自动格式化代码
3. 查看自动提交的更改
4. 合并到主分支

---

## 🪝 Pre-commit Hooks

### 安装 Pre-commit

```bash
# 安装 pre-commit
pip install pre-commit

# 安装 hooks 到 .git/hooks/
pre-commit install
```

### 手动运行

```bash
# 对所有文件运行
pre-commit run --all-files

# 只运行特定 hook
pre-commit run ruff --all-files
pre-commit run black --all-files
pre-commit run pytest --all-files
```

### 跳过 Pre-commit（不推荐）

```bash
# 跳过所有 hooks
git commit --no-verify

# 跳过特定 hook
SKIP=pytest git commit
```

### 配置的 Hooks

1. **Ruff** - Python linter
   - 自动修复简单问题
   - 检查代码规范

2. **Black** - 代码格式化
   - 统一代码风格
   - PEP 8 标准

3. **文件检查**
   - 删除行尾空格
   - 添加文件结尾换行
   - 检查 YAML/JSON 语法
   - 检查大文件（>10MB）
   - 检测合并冲突
   - 检测私钥泄露

4. **Pytest** - 运行测试
   - 提交前运行测试
   - 快速失败机制

---

## 🚀 开发工作流

### 本地开发

```bash
# 1. 创建新分支
git checkout -b feature/my-feature

# 2. 开发代码
vim src/my_module.py

# 3. 运行 pre-commit 检查
pre-commit run --all-files

# 4. 如果有错误，修复后重新提交
git add .
git commit -m "feat: 添加新功能"

# 5. 推送到远程
git push origin feature/my-feature
```

### Pull Request 流程

```bash
# 1. 在 GitHub 创建 Pull Request
# 2. 等待 CI 检查完成
# 3. 查看自动格式化提交（如有）
# 4. 修复任何失败的检查
# 5. 请求代码审查
# 6. 合并到主分支
```

---

## 📊 CI 状态徽章

在 README.md 中添加状态徽章：

```markdown
![CI](https://github.com/username/uaam-phish/workflows/CI/badge.svg)
![Code Coverage](https://codecov.io/gh/username/uaam-phish/branch/main/graph/badge.svg)
```

---

## 🔍 故障排除

### 问题 1: CI 检查失败

**Lint 错误：**
```bash
# 本地运行 ruff 检查
ruff check .

# 自动修复
ruff check --fix .

# Black 格式化
black .
```

**测试失败：**
```bash
# 本地运行测试
pytest tests/ -v

# 查看详细错误
pytest tests/ -v --tb=long

# 运行特定测试
pytest tests/test_data.py -v
```

**配置错误：**
```bash
# 验证配置文件
python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/config.yaml')"
```

### 问题 2: Pre-commit 太慢

```bash
# 跳过 pytest（临时）
SKIP=pytest git commit

# 或者编辑 .pre-commit-config.yaml
# 注释掉 pytest hook
```

### 问题 3: 自动格式化冲突

```bash
# 拉取最新的自动格式化提交
git pull origin your-branch

# 解决冲突后重新提交
git add .
git commit -m "resolve conflicts"
```

---

## ⚙️ 高级配置

### 自定义 Ruff 规则

编辑 `pyproject.toml` 或 `ruff.toml`:

```toml
[tool.ruff]
line-length = 100
select = ["E", "F", "I"]
ignore = ["E501"]
```

### 自定义 Black 配置

编辑 `pyproject.toml`:

```toml
[tool.black]
line-length = 100
target-version = ['py310']
```

### 添加新的 CI 检查

编辑 `.github/workflows/ci.yml`:

```yaml
  my-custom-check:
    name: 自定义检查
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: 运行自定义脚本
        run: python scripts/my_check.py
```

---

## 📚 最佳实践

### 1. 提交前检查

```bash
# 运行完整检查
pre-commit run --all-files
pytest tests/
```

### 2. 小而频繁的提交

```bash
# 好的提交
git commit -m "feat: 添加 URL 编码器"
git commit -m "fix: 修复数据加载 bug"
git commit -m "docs: 更新 README"

# 避免大的混合提交
# ❌ git commit -m "各种更改"
```

### 3. 使用语义化提交信息

```
feat: 新功能
fix: Bug 修复
docs: 文档更新
style: 代码格式
refactor: 重构
test: 测试相关
chore: 构建/工具配置
```

### 4. 保持 CI 绿色

- ✅ 确保所有检查通过后再合并
- ✅ 及时修复失败的检查
- ✅ 不要跳过重要检查

---

## 🎯 下一步

### 添加更多检查

1. **类型检查**
   ```bash
   pip install mypy
   mypy src/
   ```

2. **文档生成**
   ```bash
   pip install sphinx
   sphinx-build docs/ docs/_build/
   ```

3. **性能测试**
   ```bash
   pip install pytest-benchmark
   pytest tests/ --benchmark-only
   ```

### 集成其他服务

- [ ] Codecov - 代码覆盖率
- [ ] SonarCloud - 代码质量
- [ ] Dependabot - 依赖更新
- [ ] CodeQL - 安全扫描

---

## 🔗 参考资源

- [GitHub Actions 文档](https://docs.github.com/actions)
- [Pre-commit 文档](https://pre-commit.com/)
- [Ruff 文档](https://docs.astral.sh/ruff/)
- [Black 文档](https://black.readthedocs.io/)

---

**维护者:** UAAM-Phish Team
**最后更新:** 2025-10-22
