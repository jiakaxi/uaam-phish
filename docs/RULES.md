# Project Rules (UAAM-Phish)

## Objectives
- 可复现实验：同一配置应可在不同机器复现指标（±容差）。
- 质量闸门：未通过 CI（ruff + black + pytest）禁止合并。
- AI 协作：先写 specs 再让 AI 生成代码；生成后必须通过 lint/test。

## Workflow
1. 每个模块先补 `docs/specs/<module>.md`（问题→I/O→API→测试清单）。
2. 先写失败测试（TDD）→ 让 AI 完成实现 → 本地 `make lint test`。
3. 开 PR：描述动机/变更/测试截图/风险/对现有脚本的影响。
4. CI 绿灯 + 至少 1 人评审后合并到 `dev`，里程碑打 Tag。

## Code Style
- Python ≥ 3.10；类型标注、docstring（Args/Returns/Raises）。
- 命名：snake_case；类名 PascalCase；常量 UPPER_CASE。
- 禁止魔法数，统一经由配置（OmegaConf）。

## Prompts（统一结构）
- 背景（模块目标/约束/依赖）
- 需生成（函数/类/脚本）
- 接口（签名/返回/异常）
- 质量门槛（通过哪些 tests、时间复杂度、日志打点）

## Logging & Seed
- 所有入口调用 `set_global_seed(seed)`。
- 重要阶段打日志：数据统计、训练参数、阈值、关键指标。

## Data & DVC
- 原始与处理数据一律由 DVC 管理；禁止直接 push 大文件。
- 改动数据流程时，更新 `dvc.yaml` 与 `docs/DATA_README.md`。

## Commit / PR
- Conventional Commits：`feat|fix|docs|refactor|chore: ...`
- PR 标题清晰、附运行截图或指标对比。
