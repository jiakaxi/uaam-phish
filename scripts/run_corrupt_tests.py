#!/usr/bin/env python
"""
批量运行腐败数据测试脚本（Python 版本）
运行完整的 L/M/H × 3 模态 = 9 个测试

使用方法:
    # 主腐败评测（L/M/H × 3 模态 = 9 个测试）
    python scripts/run_corrupt_tests.py \
        --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
        --test-type corrupt \
        --modalities url html img \
        --levels L M H

    # IID 轻噪声（0.1/0.3/0.5 × 3 模态 = 9 个测试）
    python scripts/run_corrupt_tests.py \
        --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
        --test-type iid \
        --modalities url html img \
        --levels 0.1 0.3 0.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from src.utils.logging import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量运行腐败数据测试 - 完整套件 (L/M/H × 3 模态 = 9 个测试)"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="IID 训练目录（包含 checkpoints/best.ckpt）",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["url", "html", "img"],
        default=["url", "html", "img"],
        help="要测试的模态（默认：所有三模态）",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["L", "M", "H", "0.1", "0.3", "0.5"],
        default=["L", "M", "H"],
        help="腐败强度级别（默认：L M H，主腐败评测）",
    )
    parser.add_argument(
        "--test-type",
        choices=["corrupt", "iid"],
        default="corrupt",
        help="测试类型：corrupt（主腐败评测 L/M/H）或 iid（轻噪声 0.1/0.3/0.5）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认：experiments/corrupt_eval_<model_name>）",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="随机种子（默认：从 experiment-dir 自动发现）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不执行",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="遇到错误时继续运行剩余测试",
    )
    return parser.parse_args()


def get_test_combinations(
    modalities: List[str], levels: List[str], test_type: str
) -> List[Tuple[str, str, str]]:
    """生成所有测试组合 (modality, level, csv_path)"""
    combinations = []
    corrupt_root = Path("workspace/data/corrupt")

    for modality in modalities:
        for level in levels:
            if test_type == "corrupt":
                # 主腐败评测：L/M/H
                csv_path = (
                    corrupt_root / modality / f"test_corrupt_{modality}_{level}.csv"
                )
            else:
                # IID 轻噪声：0.1/0.3/0.5
                csv_path = (
                    corrupt_root
                    / "iid"
                    / modality
                    / f"test_corrupt_{modality}_{level}.csv"
                )

            if csv_path.exists():
                combinations.append((modality, level, str(csv_path)))
            else:
                log.warning(f"CSV 文件不存在: {csv_path}")

    return combinations


def find_experiment_config(experiment_dir: Path) -> str:
    """从实验目录查找实验配置名称"""
    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        return cfg.run.get("name", "s0_iid_earlyconcat")
    return "s0_iid_earlyconcat"


def find_checkpoint(experiment_dir: Path) -> Path:
    """查找 checkpoint 文件

    查找顺序：
    1. experiment_dir/checkpoints/
    2. experiment_dir/lightning_logs/version_*/checkpoints/
    3. 从 config.yaml 读取 output_dir，查找 outputs/.../checkpoints/
    4. 在 outputs/ 目录中按时间戳查找最近的匹配实验
    """
    # 方法1：尝试在 checkpoints 目录中查找
    checkpoints_dir = experiment_dir / "checkpoints"
    if checkpoints_dir.exists():
        ckpt_files = list(checkpoints_dir.glob("*.ckpt"))
        if ckpt_files:
            best_ckpt = [f for f in ckpt_files if "best" in f.name.lower()]
            if best_ckpt:
                return best_ckpt[0]
            return ckpt_files[0]

    # 方法2：尝试在 lightning_logs 中查找
    lightning_logs = experiment_dir / "lightning_logs"
    if lightning_logs.exists():
        for version_dir in sorted(lightning_logs.glob("version_*"), reverse=True):
            ckpt_dir = version_dir / "checkpoints"
            if ckpt_dir.exists():
                ckpt_files = list(ckpt_dir.glob("*.ckpt"))
                if ckpt_files:
                    best_ckpt = [f for f in ckpt_files if "best" in f.name.lower()]
                    if best_ckpt:
                        return best_ckpt[0]
                    return ckpt_files[0]

    # 方法3：从 config.yaml 读取 output_dir
    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(config_path)
            if "paths" in cfg and "output_dir" in cfg.paths:
                output_dir = Path(cfg.paths.output_dir)
                if output_dir.exists():
                    ckpt_dir = output_dir / "checkpoints"
                    if ckpt_dir.exists():
                        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
                        if ckpt_files:
                            best_ckpt = [
                                f for f in ckpt_files if "best" in f.name.lower()
                            ]
                            if best_ckpt:
                                return best_ckpt[0]
                            return ckpt_files[0]
        except Exception as e:
            log.warning(f"读取 config.yaml 失败: {e}")

    # 方法4：在 outputs/ 目录中查找（按实验名称和时间戳匹配）
    exp_name = experiment_dir.name
    # 提取时间戳部分（假设格式为 name_YYYYMMDD_HHMMSS）
    if "_" in exp_name:
        parts = exp_name.split("_")
        if len(parts) >= 3:
            date_part = parts[-2]  # YYYYMMDD
            time_part = parts[-1]  # HHMMSS
            # 转换为 outputs 目录格式：YYYY-MM-DD/HH-MM-SS
            if len(date_part) == 8 and len(time_part) == 6:
                date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                time_str = f"{time_part[:2]}-{time_part[2:4]}-{time_part[4:6]}"
                outputs_dir = Path("outputs") / date_str / time_str
                if outputs_dir.exists():
                    # 在 outputs 目录中查找 checkpoint
                    for ckpt_file in outputs_dir.rglob("**/*.ckpt"):
                        if "best" in ckpt_file.name.lower():
                            return ckpt_file
                    # 如果没有 best，返回第一个找到的
                    ckpt_files = list(outputs_dir.rglob("**/*.ckpt"))
                    if ckpt_files:
                        return ckpt_files[0]

    # 方法5：在 outputs/ 目录中按实验名称查找最近的
    outputs_root = Path("outputs")
    if outputs_root.exists():
        # 查找所有包含实验名称的目录
        matching_dirs = []
        for date_dir in outputs_root.glob("*"):
            if not date_dir.is_dir():
                continue
            for time_dir in date_dir.glob("*"):
                if not time_dir.is_dir():
                    continue
                # 检查是否有匹配的 checkpoint
                for ckpt_file in time_dir.rglob("**/*.ckpt"):
                    matching_dirs.append((time_dir, ckpt_file))

        if matching_dirs:
            # 按修改时间排序，返回最新的
            matching_dirs.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
            best_ckpt = [x[1] for x in matching_dirs if "best" in x[1].name.lower()]
            if best_ckpt:
                return best_ckpt[0]
            return matching_dirs[0][1]

    raise FileNotFoundError(
        f"在 {experiment_dir} 中未找到 checkpoint 文件。\n"
        f"请检查以下位置：\n"
        f"  1. {experiment_dir}/checkpoints/\n"
        f"  2. {experiment_dir}/lightning_logs/version_*/checkpoints/\n"
        f"  3. outputs/ 目录（根据实验时间戳）\n"
        f"或者手动指定 checkpoint 路径。"
    )


def run_test(
    experiment: str,
    checkpoint_path: Path,
    csv_path: str,
    modality: str,
    level: str,
    dry_run: bool = False,
) -> bool:
    """运行单个测试"""
    run_name = f"corrupt_{modality}_{level}"

    # 使用 checkpoint 路径作为 ckpt_path
    # 注意：train_hydra.py 中的 trainer.test() 会使用 ckpt_path 参数
    # 我们需要通过配置覆盖来指定 checkpoint 路径
    # 使用绝对路径或相对于项目根目录的路径
    # train_hydra.py 的装饰器指定了 config_path="../configs"，所以需要从 scripts/ 目录运行
    # 或者修改为从项目根目录运行，使用绝对路径
    project_root = Path(__file__).parent.parent

    # 将 Windows 路径中的反斜杠转换为正斜杠，避免 Hydra 将其解释为转义符
    csv_path_normalized = str(csv_path).replace("\\", "/")
    checkpoint_path_normalized = str(checkpoint_path).replace("\\", "/")

    # 构建命令，对于包含特殊字符（如 =）的 override，需要用引号包裹整个 override
    # 在 subprocess 中，将包含 = 的 override 作为单个字符串传递，Python 会自动处理引号
    # 注意：对于包含 = 的值（如 checkpoint 文件名），整个 override 需要用引号包裹
    # 使用 + 前缀添加新配置项（如果配置中不存在）
    cmd = [
        "python",
        str(project_root / "scripts" / "train_hydra.py"),
        f"experiment={experiment}",
        "trainer.max_epochs=0",
        f'datamodule.test_csv="{csv_path_normalized}"',  # 用引号包裹路径
        f'run.name="{run_name}"',
        f'+trainer.ckpt_path="{checkpoint_path_normalized}"',  # 使用 + 添加新配置，用引号包裹包含 = 的文件名
        "--config-path",
        str(project_root / "configs"),
        "--config-name",
        "config",
    ]

    log.info(f">> 运行测试: {modality.upper()}-{level}")
    log.info(f"   CSV: {csv_path}")
    log.info(f"   Checkpoint: {checkpoint_path}")
    log.info(f"   命令: {' '.join(cmd)}")

    if dry_run:
        return True

    try:
        # 确保在项目根目录运行
        project_root = Path(__file__).parent.parent
        subprocess.run(
            cmd, check=True, capture_output=True, text=True, cwd=str(project_root)
        )
        log.info("   [OK] 完成")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"   [FAIL] 失败: {e}")
        if e.stderr:
            log.error(f"   错误输出: {e.stderr}")
        return False


def main() -> None:
    args = parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        log.error(f"实验目录不存在: {experiment_dir}")
        sys.exit(1)

    # 自动发现实验配置和 checkpoint
    experiment_name = find_experiment_config(experiment_dir)
    checkpoint_path = find_checkpoint(experiment_dir)

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 从实验目录名提取模型名称
        model_name = (
            experiment_dir.name.split("_")[0]
            if "_" in experiment_dir.name
            else "default"
        )
        output_dir = Path(f"experiments/corrupt_eval_{model_name}")

    log.info("=" * 70)
    log.info("腐败数据批量测试 - 完整套件")
    log.info("=" * 70)
    log.info(f">> 实验目录: {experiment_dir}")
    log.info(f">> 实验配置: {experiment_name}")
    log.info(f">> Checkpoint: {checkpoint_path}")
    log.info(f">> 测试类型: {args.test_type}")
    log.info(f">> 模态: {', '.join(args.modalities)}")
    log.info(f">> 强度级别: {', '.join(args.levels)}")
    log.info(f">> 输出目录: {output_dir}")
    log.info("=" * 70)

    # 生成所有测试组合
    combinations = get_test_combinations(args.modalities, args.levels, args.test_type)
    total_tests = len(combinations)

    if total_tests == 0:
        log.error("未找到任何有效的测试组合。请检查 CSV 文件是否存在。")
        sys.exit(1)

    log.info(
        f"\n>> 测试计划: {len(args.modalities)} 模态 × {len(args.levels)} 强度 = {total_tests} 个测试"
    )
    log.info("=" * 70)

    # 运行所有测试
    failed_tests = []
    for i, (modality, level, csv_path) in enumerate(combinations, 1):
        log.info(f"\n[{i}/{total_tests}] {modality.upper()}-{level}")

        success = run_test(
            experiment_name,
            checkpoint_path,
            csv_path,
            modality,
            level,
            args.dry_run,
        )

        if not success:
            failed_tests.append((modality, level))
            if not args.continue_on_error:
                log.error("\n测试失败，停止运行。")
                sys.exit(1)

    # 总结
    log.info("\n" + "=" * 70)
    if failed_tests:
        log.warning(f">> 完成: {total_tests - len(failed_tests)}/{total_tests} 成功")
        log.warning(f">> 失败: {len(failed_tests)} 个测试")
        for mod, level in failed_tests:
            log.warning(f"  - {mod.upper()}-{level}")
        sys.exit(1)
    else:
        log.info(f">> 所有 {total_tests} 个测试完成！")
        log.info("=" * 70)
        log.info("测试覆盖：")
        for modality in args.modalities:
            log.info(f"  - {modality.upper()}: {', '.join(args.levels)}")
        log.info("")
        log.info("现在可以运行结果收集脚本：")
        log.info(
            f"  python scripts/test_corrupt_data.py \\\n"
            f"    --experiment-dir {experiment_dir} \\\n"
            f"    --output-dir {output_dir} \\\n"
            f"    --test-type {args.test_type} \\\n"
            f"    --modalities {' '.join(args.modalities)} \\\n"
            f"    --levels {' '.join(args.levels)}"
        )
        log.info("=" * 70)


if __name__ == "__main__":
    main()
