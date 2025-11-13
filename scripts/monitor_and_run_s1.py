"""
监控当前S1训练并自动启动后续任务
非交互式版本，适合后台运行
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import json

# 训练任务列表 (跳过第一个，因为已经在运行)
EXPERIMENTS = [
    {
        "name": "S1_IID_seed43",
        "cmd": "python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=43",
    },
    {
        "name": "S1_IID_seed44",
        "cmd": "python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=44",
    },
    {
        "name": "S1_BrandOOD_seed42",
        "cmd": "python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=42",
    },
    {
        "name": "S1_BrandOOD_seed43",
        "cmd": "python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=43",
    },
    {
        "name": "S1_BrandOOD_seed44",
        "cmd": "python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=44",
    },
]

LOG_FILE = "workspace/training_progress.log"


def log(message):
    """记录日志到文件和控制台"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"

    # 将输出重定向到UTF-8文件，避免控制台编码问题
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

    # 控制台输出（可能有编码问题，但不影响日志文件）
    try:
        print(log_msg)
    except UnicodeEncodeError:
        # 如果无法打印，只记录时间戳
        print(f"[{timestamp}] (log written to file)")


def find_latest_s1_experiment():
    """查找最新的S1实验目录"""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return None

    matching_dirs = sorted(
        [
            d
            for d in experiments_dir.iterdir()
            if d.is_dir() and "s1_" in d.name.lower()
        ],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    return matching_dirs[0] if matching_dirs else None


def is_training_complete(exp_dir):
    """检查训练是否完成"""
    if not exp_dir or not exp_dir.exists():
        return False

    summary_file = exp_dir / "SUMMARY.md"
    metrics_file = exp_dir / "results" / "metrics_final.json"

    return summary_file.exists() and metrics_file.exists()


def get_training_progress(exp_dir):
    """从日志中获取训练进度"""
    if not exp_dir or not exp_dir.exists():
        return None

    log_file = exp_dir / "logs" / "train.log"
    if not log_file.exists():
        return None

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # 找到包含"Epoch"的最后一行
        epoch_lines = [
            line for line in lines if "Epoch " in line and "train/loss" in line
        ]
        if epoch_lines:
            last_epoch_line = epoch_lines[-1]
            # 提取epoch数字
            import re

            match = re.search(r"Epoch (\d+):", last_epoch_line)
            if match:
                return int(match.group(1))

        return None
    except Exception as e:
        log(f"解析进度出错: {e}")
        return None


def wait_for_current_training():
    """等待当前训练完成"""
    log("=" * 70)
    log("检测当前训练状态...")

    current_exp = find_latest_s1_experiment()
    if not current_exp:
        log("未找到正在运行的S1实验")
        return

    log(f"找到实验目录: {current_exp.name}")

    if is_training_complete(current_exp):
        log("✅ 当前训练已完成")
        return

    log("⏳ 当前训练正在进行中，等待完成...")
    log("检查间隔: 5分钟")

    check_interval = 300  # 5分钟
    last_epoch = -1

    while not is_training_complete(current_exp):
        time.sleep(check_interval)

        current_epoch = get_training_progress(current_exp)
        if current_epoch is not None and current_epoch != last_epoch:
            log(f"训练进度: Epoch {current_epoch}/20 完成")
            last_epoch = current_epoch
        else:
            log(f"检查训练状态... (目录: {current_exp.name})")

    log(f"✅ 训练完成: {current_exp.name}")
    log("=" * 70)


def run_experiment(exp_config, index, total):
    """运行单个实验"""
    log("=" * 70)
    log(f"启动实验 {index}/{total}: {exp_config['name']}")
    log(f"命令: {exp_config['cmd']}")
    log("=" * 70)

    start_time = time.time()

    try:
        result = subprocess.run(
            exp_config["cmd"], shell=True, capture_output=True, text=True
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            log(f"✅ 实验 {exp_config['name']} 完成！耗时: {elapsed/60:.1f} 分钟")

            # 查找生成的实验目录
            exp_dir = find_latest_s1_experiment()
            if exp_dir:
                log(f"实验目录: {exp_dir}")

                # 读取最终指标
                metrics_file = exp_dir / "results" / "metrics_final.json"
                if metrics_file.exists():
                    with open(metrics_file, "r") as f:
                        metrics = json.load(f)
                        log(f"  AUROC: {metrics.get('final_test_auroc', 'N/A'):.4f}")
                        log(f"  ACC: {metrics.get('final_test_acc', 'N/A'):.4f}")
                        log(f"  Epochs: {metrics.get('total_epochs', 'N/A')}")

            return True
        else:
            log(f"❌ 实验 {exp_config['name']} 失败！返回码: {result.returncode}")
            log(f"错误输出: {result.stderr[:500]}")  # 只记录前500字符
            return False

    except Exception as e:
        elapsed = time.time() - start_time
        log(f"❌ 实验 {exp_config['name']} 出错: {e}")
        log(f"耗时: {elapsed/60:.1f} 分钟")
        return False


def main():
    """主函数"""
    log("")
    log("=" * 70)
    log("S1 实验自动训练监控脚本")
    log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"日志文件: {LOG_FILE}")
    log("=" * 70)

    # 等待当前训练完成
    wait_for_current_training()

    # 顺序运行剩余实验
    log("\n开始运行剩余5个实验...")
    total_start = time.time()
    results = []

    for i, exp_config in enumerate(EXPERIMENTS, start=2):  # 从2开始因为seed=42是第1个
        success = run_experiment(exp_config, i, 6)
        results.append({"name": exp_config["name"], "success": success, "index": i})

        if not success:
            log(f"⚠️  实验 {exp_config['name']} 失败，继续运行下一个")

        # 短暂等待
        time.sleep(10)

    # 总结
    total_elapsed = time.time() - total_start
    log("")
    log("=" * 70)
    log("所有实验完成！")
    log(f"总耗时: {total_elapsed/3600:.2f} 小时")
    log(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 70)

    # 结果汇总
    log("\n实验结果汇总:")
    log("1/6 - S1_IID_seed42 ✅ (已完成)")
    for result in results:
        icon = "✅" if result["success"] else "❌"
        log(f"{result['index']}/6 - {result['name']} {icon}")

    successful = 1 + sum(1 for r in results if r["success"])
    log(f"\n总成功: {successful}/6")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n用户中断，退出脚本")
        sys.exit(1)
    except Exception as e:
        log(f"\n脚本出错: {e}")
        import traceback

        log(traceback.format_exc())
        sys.exit(1)
