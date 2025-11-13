"""
完整S1实验自动化流程
1. 监控当前训练（seed=42）
2. 自动启动后续5个实验
3. 完成后自动运行Phase 4结果分析
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import json

# 实验配置
EXPERIMENTS = [
    {
        "id": 1,
        "name": "S1_IID_seed42",
        "status": "running",
        "dir_pattern": "s1_iid_lateavg_20251112_155335",
    },
    {
        "id": 2,
        "name": "S1_IID_seed43",
        "cmd": "python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=43",
    },
    {
        "id": 3,
        "name": "S1_IID_seed44",
        "cmd": "python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=44",
    },
    {
        "id": 4,
        "name": "S1_BrandOOD_seed42",
        "cmd": "python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=42",
    },
    {
        "id": 5,
        "name": "S1_BrandOOD_seed43",
        "cmd": "python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=43",
    },
    {
        "id": 6,
        "name": "S1_BrandOOD_seed44",
        "cmd": "python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=44",
    },
]

LOG_FILE = Path("workspace/full_automation.log")
STATUS_FILE = Path("workspace/automation_status.json")


def log(message, level="INFO"):
    """记录日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] [{level}] {message}"

    print(log_msg)

    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")


def save_status(status):
    """保存当前状态"""
    STATUS_FILE.parent.mkdir(exist_ok=True)
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)


def load_status():
    """加载状态"""
    if STATUS_FILE.exists():
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"current_experiment": 1, "completed": [], "failed": []}


def find_experiment_dir(pattern=None):
    """查找实验目录"""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        return None

    if pattern:
        # 查找特定目录
        target = experiments_dir / pattern
        return target if target.exists() else None

    # 查找最新的S1目录
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
    if not exp_dir or not isinstance(exp_dir, Path):
        exp_dir = Path(exp_dir) if exp_dir else None

    if not exp_dir or not exp_dir.exists():
        return False

    summary = exp_dir / "SUMMARY.md"
    metrics = exp_dir / "results" / "metrics_final.json"

    return summary.exists() and metrics.exists()


def get_training_progress(exp_dir):
    """获取训练进度"""
    if not exp_dir or not isinstance(exp_dir, Path):
        exp_dir = Path(exp_dir) if exp_dir else None

    if not exp_dir or not exp_dir.exists():
        return {"status": "not_found"}

    log_file = exp_dir / "logs" / "train.log"
    if not log_file.exists():
        return {"status": "starting", "message": "日志文件未创建"}

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        # 查找最后的epoch信息
        epoch_lines = [
            line for line in lines if "Epoch " in line and "train/loss" in line
        ]
        if epoch_lines:
            import re

            last_line = epoch_lines[-1]
            match = re.search(r"Epoch (\d+):", last_line)
            if match:
                current_epoch = int(match.group(1))
                return {
                    "status": "training",
                    "current_epoch": current_epoch,
                    "total_epochs": 20,
                    "progress": f"{current_epoch}/20",
                    "message": f"训练中: Epoch {current_epoch}/20",
                }

        return {"status": "running", "message": "训练已开始"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def wait_for_experiment_1():
    """监控实验1直到完成"""
    log("=" * 70)
    log("步骤 1/3: 监控当前训练 (S1 IID seed=42)")
    log("=" * 70)

    exp1_dir = find_experiment_dir(EXPERIMENTS[0].get("dir_pattern"))
    if not exp1_dir:
        log("未找到实验1的目录，尝试查找最新S1目录...", "WARN")
        exp1_dir = find_experiment_dir()

    if not exp1_dir:
        log("错误: 未找到实验1的目录", "ERROR")
        return False

    log(f"实验目录: {exp1_dir}")

    if is_training_complete(exp1_dir):
        log("实验1已经完成！")
        return True

    log("实验1正在运行中，开始监控...")
    log("检查间隔: 3分钟")

    check_interval = 180  # 3分钟
    last_progress = None

    while not is_training_complete(exp1_dir):
        progress = get_training_progress(exp1_dir)

        if progress != last_progress:
            log(f"状态: {progress.get('message', '训练中...')}")
            last_progress = progress

        time.sleep(check_interval)

    log("=" * 70)
    log("实验1完成！")
    log("=" * 70)
    return True


def run_experiment(exp_config):
    """运行单个实验"""
    exp_id = exp_config["id"]
    exp_name = exp_config["name"]
    cmd = exp_config["cmd"]

    log("=" * 70)
    log(f"启动实验 {exp_id}/6: {exp_name}")
    log(f"命令: {cmd}")
    log("=" * 70)

    start_time = time.time()

    try:
        # 运行训练
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            log(f"实验 {exp_name} 完成！耗时: {elapsed/3600:.2f} 小时")

            # 查找生成的实验目录
            exp_dir = find_experiment_dir()
            if exp_dir and is_training_complete(exp_dir):
                log(f"实验目录: {exp_dir}")

                # 读取指标
                metrics_file = exp_dir / "results" / "metrics_final.json"
                if metrics_file.exists():
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    log(f"  AUROC: {metrics.get('final_test_auroc', 'N/A'):.4f}")
                    log(f"  ACC: {metrics.get('final_test_acc', 'N/A'):.4f}")
                    log(f"  F1: {metrics.get('final_test_f1', 'N/A'):.4f}")

            return True, exp_dir
        else:
            log(f"实验 {exp_name} 失败！返回码: {result.returncode}", "ERROR")
            if result.stderr:
                log(f"错误信息: {result.stderr[:500]}", "ERROR")
            return False, None

    except Exception as e:
        elapsed = time.time() - start_time
        log(f"实验 {exp_name} 出错: {e}", "ERROR")
        return False, None


def run_remaining_experiments():
    """运行剩余5个实验"""
    log("")
    log("=" * 70)
    log("步骤 2/3: 运行剩余5个实验")
    log("=" * 70)

    status = load_status()
    results = []

    for exp in EXPERIMENTS[1:]:  # 跳过第一个
        exp_id = exp["id"]

        # 检查是否已完成
        if exp_id in status["completed"]:
            log(f"实验 {exp['name']} 已完成，跳过")
            results.append(
                {"id": exp_id, "name": exp["name"], "success": True, "skipped": True}
            )
            continue

        # 运行实验
        success, exp_dir = run_experiment(exp)
        results.append(
            {
                "id": exp_id,
                "name": exp["name"],
                "success": success,
                "dir": str(exp_dir) if exp_dir else None,
            }
        )

        # 更新状态
        if success:
            status["completed"].append(exp_id)
        else:
            status["failed"].append(exp_id)
        status["current_experiment"] = exp_id
        save_status(status)

        # 短暂等待
        time.sleep(10)

    return results


def run_phase4_analysis():
    """运行Phase 4结果分析"""
    log("")
    log("=" * 70)
    log("步骤 3/3: Phase 4 结果分析")
    log("=" * 70)

    # 检查脚本是否存在
    eval_script = Path("scripts/evaluate_s0.py")
    summary_script = Path("scripts/summarize_s0_results.py")

    if not eval_script.exists():
        log("警告: evaluate_s0.py 不存在，跳过结果提取", "WARN")
    else:
        log("提取评估结果...")
        try:
            result = subprocess.run(
                "python scripts/evaluate_s0.py --runs_dir experiments",
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                log("评估结果提取完成")
            else:
                log(f"评估结果提取失败: {result.stderr[:500]}", "WARN")
        except Exception as e:
            log(f"评估结果提取出错: {e}", "WARN")

    if not summary_script.exists():
        log("警告: summarize_s0_results.py 不存在，跳过汇总生成", "WARN")
    else:
        log("生成S0/S1汇总表格...")
        try:
            result = subprocess.run(
                "python scripts/summarize_s0_results.py",
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                log("S0/S1汇总表格生成完成")
            else:
                log(f"汇总表格生成失败: {result.stderr[:500]}", "WARN")
        except Exception as e:
            log(f"汇总表格生成出错: {e}", "WARN")

    log("Phase 4 完成")


def main():
    """主函数"""
    # 确保在正确的工作目录
    project_root = Path(__file__).parent.parent.resolve()
    os.chdir(project_root)

    # 设置PYTHONPATH确保模块能正确导入
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    os.environ["PYTHONPATH"] = str(project_root)

    log("")
    log("=" * 70)
    log("S1 实验完整自动化流程")
    log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"工作目录: {os.getcwd()}")
    log(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    log("=" * 70)
    log("")
    log("任务:")
    log("  1. 监控当前训练 (S1 IID seed=42)")
    log("  2. 自动启动后续5个实验")
    log("  3. 完成后进行Phase 4结果分析")
    log("")

    total_start = time.time()

    try:
        # 步骤1: 等待实验1完成
        if not wait_for_experiment_1():
            log("实验1监控失败，退出", "ERROR")
            return 1

        # 更新状态
        status = load_status()
        status["completed"].append(1)
        save_status(status)

        # 步骤2: 运行剩余实验
        results = run_remaining_experiments()

        # 步骤3: Phase 4分析
        run_phase4_analysis()

        # 最终总结
        total_elapsed = time.time() - total_start
        log("")
        log("=" * 70)
        log("所有任务完成！")
        log(f"总耗时: {total_elapsed/3600:.2f} 小时")
        log(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log("=" * 70)

        # 结果汇总
        log("")
        log("实验结果汇总:")
        log("  1/6 - S1_IID_seed42: 完成")

        for result in results:
            status_icon = "完成" if result["success"] else "失败"
            skipped = " (已存在)" if result.get("skipped") else ""
            log(f"  {result['id']}/6 - {result['name']}: {status_icon}{skipped}")

        successful = 1 + sum(1 for r in results if r["success"])
        log(f"\n总成功: {successful}/6")

        log("")
        log("输出文件:")
        log(f"  - 日志: {LOG_FILE}")
        log(f"  - 状态: {STATUS_FILE}")
        log("  - 实验目录: experiments/s1_*")

        return 0

    except KeyboardInterrupt:
        log("\n用户中断", "WARN")
        return 1
    except Exception as e:
        log(f"\n严重错误: {e}", "ERROR")
        import traceback

        log(traceback.format_exc(), "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
