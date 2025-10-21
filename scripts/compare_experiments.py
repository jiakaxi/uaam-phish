#!/usr/bin/env python
"""
实验对比工具
快速对比多个实验的结果
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd


def load_experiment_metrics(exp_dir: Path) -> Dict:
    """加载实验指标"""
    metrics_file = exp_dir / "results" / "metrics_final.json"
    config_file = exp_dir / "config.yaml"
    
    result = {
        "实验名称": exp_dir.name,
        "实验目录": str(exp_dir),
    }
    
    # 加载指标
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            metrics = data.get('metrics', {})
            
            # 提取关键指标
            for key, value in metrics.items():
                clean_key = key.replace('/', '_').replace('test_', '')
                if isinstance(value, float):
                    result[clean_key] = round(value, 4)
                else:
                    result[clean_key] = value
    else:
        result["状态"] = "❌ 指标文件缺失"
    
    # 加载配置（可选）
    if config_file.exists():
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(config_file)
            result["模型"] = cfg.model.pretrained_name
            result["学习率"] = cfg.train.lr
            result["批量大小"] = cfg.train.bs
            result["Dropout"] = cfg.model.dropout
        except Exception:
            pass
            
    return result


def compare_experiments(exp_dirs: List[Path], output_file: str = None) -> pd.DataFrame:
    """
    对比多个实验
    
    Args:
        exp_dirs: 实验目录列表
        output_file: 输出文件路径（可选）
        
    Returns:
        对比结果 DataFrame
    """
    results = []
    
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            print(f"⚠️  实验目录不存在: {exp_dir}")
            continue
            
        try:
            metrics = load_experiment_metrics(exp_dir)
            results.append(metrics)
        except Exception as e:
            print(f"⚠️  加载实验失败 {exp_dir.name}: {e}")
    
    if not results:
        print("❌ 没有成功加载任何实验")
        return None
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 按 F1 或 AUROC 排序（如果存在）
    if 'f1' in df.columns:
        df = df.sort_values('f1', ascending=False)
    elif 'auroc' in df.columns:
        df = df.sort_values('auroc', ascending=False)
    
    # 打印结果
    print("\n" + "=" * 100)
    print("实验对比结果")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)
    
    # 保存到文件
    if output_file:
        output_path = Path(output_file)
        if output_path.suffix == '.csv':
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif output_path.suffix == '.xlsx':
            df.to_excel(output_path, index=False)
        elif output_path.suffix == '.md':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# 实验对比结果\n\n")
                f.write(df.to_markdown(index=False))
        else:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 对比结果已保存到: {output_path}")
    
    return df


def find_latest_experiments(base_dir: Path, n: int = 5) -> List[Path]:
    """查找最近的 N 个实验"""
    exp_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    exp_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return exp_dirs[:n]


def find_best_experiment(base_dir: Path, metric: str = 'f1') -> Path:
    """查找最佳实验"""
    best_exp = None
    best_value = -float('inf')
    
    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir():
            continue
            
        metrics_file = exp_dir / "results" / "metrics_final.json"
        if not metrics_file.exists():
            continue
            
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                metrics = data.get('metrics', {})
                
                # 查找指定指标
                value = None
                for key in metrics:
                    if metric in key.lower():
                        value = metrics[key]
                        break
                
                if value is not None and value > best_value:
                    best_value = value
                    best_exp = exp_dir
        except Exception:
            continue
    
    return best_exp


def main():
    parser = argparse.ArgumentParser(description="实验对比工具")
    parser.add_argument("--base_dir", default="experiments", help="实验根目录")
    parser.add_argument("--exp_names", nargs='+', help="要对比的实验名称列表")
    parser.add_argument("--latest", type=int, help="对比最近的 N 个实验")
    parser.add_argument("--all", action="store_true", help="对比所有实验")
    parser.add_argument("--output", help="输出文件路径 (.csv, .xlsx, .md)")
    parser.add_argument("--metric", default="f1", help="排序指标")
    parser.add_argument("--find_best", action="store_true", help="查找最佳实验")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"❌ 实验目录不存在: {base_dir}")
        return
    
    # 查找最佳实验
    if args.find_best:
        best_exp = find_best_experiment(base_dir, args.metric)
        if best_exp:
            print(f"🏆 最佳实验 (按 {args.metric}): {best_exp.name}")
            metrics = load_experiment_metrics(best_exp)
            print("\n指标:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        else:
            print(f"❌ 未找到有效的实验结果")
        return
    
    # 确定要对比的实验
    exp_dirs = []
    
    if args.exp_names:
        # 指定实验名称
        for name in args.exp_names:
            exp_dir = base_dir / name
            if exp_dir.exists():
                exp_dirs.append(exp_dir)
            else:
                # 尝试模糊匹配
                matched = [d for d in base_dir.iterdir() 
                          if d.is_dir() and name in d.name]
                if matched:
                    exp_dirs.extend(matched)
                else:
                    print(f"⚠️  未找到实验: {name}")
    
    elif args.latest:
        # 最近的 N 个实验
        exp_dirs = find_latest_experiments(base_dir, args.latest)
        print(f"📊 对比最近的 {len(exp_dirs)} 个实验:")
        for exp in exp_dirs:
            print(f"  - {exp.name}")
        print()
    
    elif args.all:
        # 所有实验
        exp_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        print(f"📊 对比所有 {len(exp_dirs)} 个实验\n")
    
    else:
        # 默认：最近的 5 个实验
        exp_dirs = find_latest_experiments(base_dir, 5)
        print(f"📊 对比最近的 {len(exp_dirs)} 个实验 (使用 --latest N 或 --all 修改):")
        for exp in exp_dirs:
            print(f"  - {exp.name}")
        print()
    
    if not exp_dirs:
        print("❌ 没有找到要对比的实验")
        print("提示:")
        print("  - 使用 --exp_names exp1 exp2 指定实验")
        print("  - 使用 --latest 10 对比最近 10 个实验")
        print("  - 使用 --all 对比所有实验")
        return
    
    # 执行对比
    df = compare_experiments(exp_dirs, output_file=args.output)
    
    # 显示最佳实验
    if df is not None and len(df) > 0:
        print(f"\n🏆 当前对比中的最佳实验:")
        best_row = df.iloc[0]
        print(f"  实验: {best_row['实验名称']}")
        if args.metric in df.columns:
            print(f"  {args.metric}: {best_row[args.metric]}")


if __name__ == "__main__":
    main()

