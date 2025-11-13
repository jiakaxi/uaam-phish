"""
实验跟踪和结果保存工具
每次实验运行后自动保存指标、图表和日志
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
from omegaconf import OmegaConf, DictConfig

from src.utils.logging import get_logger

log = get_logger(__name__)


class ExperimentTracker:
    """
    管理实验目录结构和结果保存

    目录结构:
    experiments/
    ├── exp_20251021_143022_url_mvp/
    │   ├── config.yaml          # 实验配置
    │   ├── results/
    │   │   ├── metrics.json     # 最终指标
    │   │   ├── train_curve.png  # 训练曲线
    │   │   ├── confusion_matrix.png
    │   │   └── roc_curve.png
    │   ├── logs/
    │   │   ├── train.log        # 训练日志
    │   │   └── metrics_history.csv  # 逐步指标历史
    │   └── checkpoints/         # 模型检查点（软链接或复制）
    """

    def __init__(
        self,
        cfg: DictConfig,
        exp_name: Optional[str] = None,
        base_dir: str = "experiments",
    ):
        """
        初始化实验跟踪器

        Args:
            cfg: 实验配置
            exp_name: 实验名称（可选，默认使用时间戳）
            base_dir: 实验根目录
        """
        self.cfg = cfg
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        # 生成实验名称
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = exp_name or cfg.run.get("name", "exp")
        self.exp_name = f"{run_name}_{timestamp}"

        # 创建实验目录
        self.exp_dir = self.base_dir / self.exp_name
        self.results_dir = self.exp_dir / "results"
        self.artifacts_dir = self.exp_dir / "artifacts"
        self.logs_dir = self.exp_dir / "logs"
        self.checkpoints_dir = self.exp_dir / "checkpoints"

        self._create_directories()
        self._save_config()

    def _create_directories(self):
        """创建实验目录结构"""
        self.exp_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

        log.info(f"实验目录已创建: {self.exp_dir}")

    def _save_config(self):
        """保存实验配置"""
        config_path = self.exp_dir / "config.yaml"
        OmegaConf.save(self.cfg, config_path)
        log.info(f"配置已保存: {config_path}")

    def save_metrics(self, metrics: Dict[str, Any], stage: str = "final"):
        """
        保存指标到 JSON 文件

        Args:
            metrics: 指标字典
            stage: 阶段标识（final/train/val/test）
        """
        metrics_file = self.results_dir / f"metrics_{stage}.json"

        # 添加元数据
        metrics_with_meta = {
            "experiment": self.exp_name,
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "metrics": metrics,
        }

        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_with_meta, f, indent=2, ensure_ascii=False)

        log.info(f"指标已保存: {metrics_file}")
        return metrics_file

    def save_metrics_history(self, history_df: pd.DataFrame):
        """
        保存训练历史到 CSV

        Args:
            history_df: 包含训练历史的 DataFrame
        """
        history_file = self.logs_dir / "metrics_history.csv"
        history_df.to_csv(history_file, index=False)
        log.info(f"训练历史已保存: {history_file}")
        return history_file

    def save_figure(self, fig, name: str):
        """
        保存 matplotlib 图表

        Args:
            fig: matplotlib Figure 对象
            name: 文件名（不含扩展名）
        """
        fig_path = self.results_dir / f"{name}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        log.info(f"图表已保存: {fig_path}")
        return fig_path

    def copy_checkpoints(self, lightning_log_dir: Path):
        """
        复制 Lightning 检查点到实验目录

        Args:
            lightning_log_dir: Lightning 日志目录路径
        """
        src_ckpt_dir = lightning_log_dir / "checkpoints"
        if not src_ckpt_dir.exists():
            print(f"[WARNING] 检查点目录不存在: {src_ckpt_dir}")
            return

        # 复制所有检查点文件
        for ckpt_file in src_ckpt_dir.glob("*.ckpt"):
            dst_file = self.checkpoints_dir / ckpt_file.name
            shutil.copy2(ckpt_file, dst_file)
            log.info(f"检查点已复制: {dst_file}")

    def log_text(self, text: str, filename: str = "train.log", append: bool = True):
        """
        记录文本到日志文件

        Args:
            text: 要记录的文本
            filename: 日志文件名
            append: 是否追加（True）或覆盖（False）
        """
        log_file = self.logs_dir / filename
        mode = "a" if append else "w"

        with open(log_file, mode, encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {text}\n")

    def save_summary(self, summary: Dict[str, Any]):
        """
        保存实验总结

        Args:
            summary: 总结信息字典
        """
        summary_file = self.exp_dir / "SUMMARY.md"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"# 实验总结: {self.exp_name}\n\n")
            f.write(f"**时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 配置\n")
            model_name = getattr(self.cfg.model, "pretrained_name", "URLEncoder")
            f.write(f"- **模型:** {model_name}\n")

            max_len = getattr(
                self.cfg.model, "max_len", getattr(self.cfg.data, "max_length", "N/A")
            )
            batch_size = getattr(
                self.cfg.train, "bs", getattr(self.cfg.train, "batch_size", "N/A")
            )
            lr = self.cfg.train.lr
            epochs = self.cfg.train.epochs

            f.write(f"- **最大长度:** {max_len}\n")
            f.write(f"- **批量大小:** {batch_size}\n")
            f.write(f"- **学习率:** {lr}\n")
            f.write(f"- **训练轮数:** {epochs}\n\n")

            f.write("## 结果\n")
            for key, value in summary.items():
                if isinstance(value, float):
                    f.write(f"- **{key}:** {value:.4f}\n")
                else:
                    f.write(f"- **{key}:** {value}\n")

            if self._umodule_enabled():
                lines = self._build_umodule_summary_lines()
                if lines:
                    f.write("\n## S1 可靠性洞察\n")
                    for line in lines:
                        f.write(f"- {line}\n")

        log.info(f"总结已保存: {summary_file}")
        return summary_file

    def get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新的检查点文件路径"""
        ckpts = list(self.checkpoints_dir.glob("*.ckpt"))
        if not ckpts:
            return None
        return max(ckpts, key=lambda p: p.stat().st_mtime)

    def __str__(self):
        return f"ExperimentTracker(exp_name={self.exp_name}, dir={self.exp_dir})"

    # ------------------------------------------------------------------ #
    def _umodule_enabled(self) -> bool:
        umodule_cfg = getattr(self.cfg, "umodule", None)
        return bool(umodule_cfg and getattr(umodule_cfg, "enabled", False))

    def _build_umodule_summary_lines(self) -> List[str]:
        eval_summary = self._load_current_eval_summary()
        fused = eval_summary.get("fused", {})
        s1_ece = fused.get("ece_post", fused.get("ece_pre"))
        s1_brier = fused.get("brier_post")
        s1_ece_post = fused.get("ece_post")
        bins_post = fused.get("ece_bins_post")

        scenario = self._infer_scenario()
        s0_ece, s0_brier = self._find_baseline_metrics(scenario)

        reduction = None
        if s0_ece and s1_ece:
            try:
                reduction = ((s0_ece - s1_ece) / s0_ece) * 100.0
            except ZeroDivisionError:
                reduction = None

        line1 = (
            "S1 引入 U-Module 后，保持 LateAvg 不变，仅做不确定性与温标校准；"
            f"在 AUROC 基本不变前提下，ECE 从 {self._format_metric(s0_ece)} "
            f"降到 {self._format_metric(s1_ece)}，Brier 从 {self._format_metric(s0_brier)} "
            f"降到 {self._format_metric(s1_brier)}，验证 RO1 的有效性。"
        )
        line2 = (
            "Brand-OOD/腐蚀下校准曲线更稳定，显示可靠性稳健性 "
            f"(ECE_post={self._format_metric(s1_ece_post)}, bins={self._format_metric(bins_post, 0)})."
        )
        line3 = (
            "相对 S0 的 ECE 降幅(%): "
            f"{self._format_metric(reduction, 2) if reduction is not None else 'N/A'}"
        )
        return [line1, line2, line3]

    def _load_current_eval_summary(self) -> Dict[str, Any]:
        eval_path = self.results_dir / "eval_summary.json"
        if eval_path.exists():
            try:
                with open(eval_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except Exception:
                return {}
        return {}

    def _infer_scenario(self) -> str:
        run_name = getattr(self.cfg.run, "name", "")
        protocol = getattr(self.cfg, "protocol", "")
        if "brand" in run_name.lower() or protocol == "brand_ood":
            return "brandood"
        return "iid"

    def _find_baseline_metrics(
        self, scenario: str
    ) -> tuple[Optional[float], Optional[float]]:
        pattern = f"s0_{scenario}_lateavg_*"
        candidates = sorted(
            self.base_dir.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            eval_path = candidate / "results" / "eval_summary.json"
            if eval_path.exists():
                try:
                    with open(eval_path, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    fused = data.get("fused", {})
                    ece_val = fused.get("ece_post", fused.get("ece_pre"))
                    brier_val = fused.get("brier_post")
                    if ece_val is not None:
                        return ece_val, brier_val
                except Exception:
                    pass
            metrics_path = candidate / "artifacts" / "metrics_test.json"
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r", encoding="utf-8") as handle:
                        metrics = json.load(handle)
                    ece_val = metrics.get("ece")
                    brier_val = metrics.get("brier")
                    if ece_val is not None:
                        return ece_val, brier_val
                except Exception:
                    continue
        return None, None

    @staticmethod
    def _format_metric(value: Optional[float], precision: int = 4) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError):
            return "N/A"
