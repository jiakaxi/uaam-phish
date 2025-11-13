"""
自动运行S1实验的6个训练任务
监控当前训练，完成后自动启动下一个
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import json

# 训练任务列表
EXPERIMENTS = [
    {
        "name": "S1_IID_seed42",
        "cmd": "python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=42",
    },
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


def find_latest_experiment_dir(pattern="s1_"):
    """查找最新的实验目录"""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return None

    matching_dirs = sorted(
        [
            d
            for d in experiments_dir.iterdir()
            if d.is_dir() and pattern in d.name.lower()
        ],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )

    return matching_dirs[0] if matching_dirs else None


def is_training_complete(exp_dir):
    """检查训练是否完成（通过SUMMARY.md和metrics_final.json的存在）"""
    if not exp_dir or not exp_dir.exists():
        return False

    summary_file = exp_dir / "SUMMARY.md"
    metrics_file = exp_dir / "results" / "metrics_final.json"

    return summary_file.exists() and metrics_file.exists()


def get_experiment_status(exp_dir):
    """获取实验的详细状态"""
    if not exp_dir or not exp_dir.exists():
        return {"status": "not_found", "message": "实验目录不存在"}

    summary_file = exp_dir / "SUMMARY.md"
    metrics_file = exp_dir / "results" / "metrics_final.json"
    log_file = exp_dir / "logs" / "train.log"

    status = {
        "exp_dir": str(exp_dir),
        "has_summary": summary_file.exists(),
        "has_metrics": metrics_file.exists(),
        "has_logs": log_file.exists(),
    }

    if log_file.exists():
        # 读取最后几行日志
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            status["log_last_lines"] = lines[-5:] if len(lines) >= 5 else lines

    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
                status["final_metrics"] = {
                    "test_auroc": metrics.get("final_test_auroc"),
                    "test_acc": metrics.get("final_test_acc"),
                    "total_epochs": metrics.get("total_epochs"),
                }
        except Exception as e:
            status["metrics_error"] = str(e)

    if is_training_complete(exp_dir):
        status["status"] = "complete"
        status["message"] = "训练已完成"
    else:
        status["status"] = "running"
        status["message"] = "训练进行中"

    return status


def run_experiment(exp_config, start_index):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 启动实验 {start_index + 1}/6: {exp_config['name']}"
    )
    print(f"命令: {exp_config['cmd']}")
    print(f"{'='*70}\n")

    # 记录开始时间
    start_time = time.time()

    try:
        # 运行训练命令
        result = subprocess.run(
            exp_config["cmd"],
            shell=True,
            capture_output=False,  # 让输出直接显示到控制台
            text=True,
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(
                f"\n✅ 实验 {exp_config['name']} 完成！耗时: {elapsed_time/60:.1f} 分钟"
            )
            return True
        else:
            print(f"\n❌ 实验 {exp_config['name']} 失败！返回码: {result.returncode}")
            return False

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ 实验 {exp_config['name']} 出错: {e}")
        print(f"耗时: {elapsed_time/60:.1f} 分钟")
        return False


def main():
    """主函数：顺序运行所有实验"""
    print(f"\n{'='*70}")
    print("S1 实验自动训练脚本")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总共 {len(EXPERIMENTS)} 个实验")
    print(f"{'='*70}\n")

    # 检查是否已有训练在运行
    latest_exp = find_latest_experiment_dir()
    if latest_exp:
        print(f"检测到最新的实验目录: {latest_exp}")
        status = get_experiment_status(latest_exp)
        print(f"状态: {status['message']}")

        if status["status"] == "running":
            print("\n⚠️  检测到有训练正在进行中")
            response = input("是否等待当前训练完成后继续？(y/n): ")
            if response.lower() != "y":
                print("退出脚本")
                return

            # 等待当前训练完成
            print("\n等待当前训练完成...")
            check_interval = 300  # 5分钟检查一次
            while not is_training_complete(latest_exp):
                time.sleep(check_interval)
                status = get_experiment_status(latest_exp)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 训练进行中...")

            print("\n✅ 当前训练已完成！")

    # 顺序运行所有实验
    total_start_time = time.time()
    results = []

    for i, exp_config in enumerate(EXPERIMENTS):
        success = run_experiment(exp_config, i)
        results.append({"name": exp_config["name"], "success": success, "index": i + 1})

        if not success:
            print(f"\n⚠️  实验 {exp_config['name']} 失败，但继续运行下一个实验")

        # 短暂等待，确保文件系统同步
        time.sleep(10)

    # 总结
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*70}")
    print("所有实验完成！")
    print(f"总耗时: {total_elapsed/3600:.2f} 小时")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    # 结果汇总
    print("\n实验结果汇总:")
    for result in results:
        status_icon = "✅" if result["success"] else "❌"
        print(f"{status_icon} {result['index']}/6 - {result['name']}")

    successful_count = sum(1 for r in results if r["success"])
    print(f"\n成功: {successful_count}/{len(EXPERIMENTS)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断，退出脚本")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n脚本出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
