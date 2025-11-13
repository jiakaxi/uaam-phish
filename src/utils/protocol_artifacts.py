"""
Artifact generation utilities (thesis Sec. 4.6.4).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Set non-interactive backend to avoid Tkinter thread conflicts
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend

import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logging import get_logger
from src.utils.metrics import ece as compute_ece_metric, brier_score


log = get_logger(__name__)


class ArtifactsWriter:
    def __init__(
        self,
        module,
        output_dir: Path | str,
        split_metadata: Optional[Dict] = None,
    ) -> None:
        self.module = module
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.split_metadata = split_metadata or {}
        self._splits_written = False

    # ------------------------------------------------------------------ #
    def update_split_metadata(self, split_metadata: Dict) -> None:
        self.split_metadata = split_metadata or {}
        self._splits_written = False

    # ------------------------------------------------------------------ #
    def save_stage_artifacts(self, preds_list: List[Dict], stage: str) -> None:
        if not preds_list:
            log.warning(
                "No predictions provided for stage '%s'; skip artifacts.", stage
            )
            return

        df_preds = self._build_predictions_dataframe(preds_list)
        self._write_predictions(df_preds, stage)

        metrics = self._compute_metrics(df_preds, stage)
        self._write_metrics(metrics, stage)
        self._plot_roc(df_preds, stage)
        self._plot_reliability(df_preds, stage)
        self._maybe_write_splits()

    # ------------------------------------------------------------------ #
    def _build_predictions_dataframe(self, preds_list: List[Dict]) -> pd.DataFrame:
        ids: List = []
        y_true_chunks: List[torch.Tensor] = []
        logits_chunks: List[torch.Tensor] = []
        prob_chunks: List[torch.Tensor] = []
        extra_cols: Dict[str, List[torch.Tensor]] = defaultdict(list)

        for batch_preds in preds_list:
            batch_ids = self._to_list(batch_preds.get("id"))
            batch_true = (
                self._ensure_tensor(batch_preds["y_true"]).view(-1).to(torch.long)
            )
            batch_logits = (
                self._ensure_tensor(batch_preds["logit"]).view(-1).to(torch.float32)
            )
            batch_prob = (
                self._ensure_tensor(batch_preds["prob"]).view(-1).to(torch.float32)
            )

            if not batch_ids:
                batch_ids = [None] * batch_true.shape[0]
            ids.extend(batch_ids)

            y_true_chunks.append(batch_true)
            logits_chunks.append(batch_logits)
            prob_chunks.append(batch_prob)

            extras = batch_preds.get("extras") or {}
            for key, value in extras.items():
                tensor = self._ensure_tensor(value).view(-1).to(torch.float32)
                if tensor.numel() != batch_true.shape[0]:
                    raise ValueError(
                        f"Extras column '{key}' length {tensor.numel()} "
                        f"!= batch size {batch_true.shape[0]}"
                    )
                extra_cols[key].append(tensor)

        y_true = torch.cat(y_true_chunks).view(-1).cpu().numpy()
        logits = torch.cat(logits_chunks).view(-1).cpu().numpy()
        probs = torch.cat(prob_chunks).view(-1).cpu().numpy()
        y_pred = (probs > 0.5).astype(int)

        data = {
            "sample_id": ids,
            "y_true": y_true,
            "logit": logits,
            "prob": probs,
            "y_pred": y_pred,
        }
        for key, tensors in extra_cols.items():
            data[key] = torch.cat(tensors).view(-1).cpu().numpy()

        return pd.DataFrame(data)

    def _write_predictions(self, df: pd.DataFrame, stage: str) -> None:
        stage_path = self.output_dir / f"predictions_{stage}.csv"
        df.to_csv(stage_path, index=False)
        log.info(">> Saved predictions: %s", stage_path)
        if stage == "test":
            final_path = self.output_dir / "predictions.csv"
            df.to_csv(final_path, index=False)

    def _compute_metrics(self, df: pd.DataFrame, stage: str) -> Dict[str, float]:
        metrics = {
            "accuracy": float(accuracy_score(df["y_true"], df["y_pred"])),
            "auroc": float(roc_auc_score(df["y_true"], df["prob"])),
            "f1_macro": float(f1_score(df["y_true"], df["y_pred"], average="macro")),
            "brier": float(brier_score(df["prob"].to_numpy(), df["y_true"].to_numpy())),
            "positive_class": "phishing",
        }
        try:
            ece_value, stats = compute_ece_metric(
                df["prob"].to_numpy(),
                df["y_true"].to_numpy(),
                n_bins=self._get_ece_bins(),
            )
            metrics["ece"] = float(ece_value)
            metrics["ece_bins_used"] = int(stats["bins_used"])
            metrics["ece_low_sample_warning"] = bool(stats.get("ece_reason"))
            if stats.get("ece_reason"):
                metrics["ece_reason"] = stats["ece_reason"]
            metrics["n_samples"] = int(stats.get("n_samples", len(df)))
            metrics[f"n_{stage}"] = int(stats.get("n_samples", len(df)))
        except Exception as exc:
            log.warning("Failed to compute ECE: %s", exc)
        metrics["artifacts"] = self._stage_artifact_paths(stage)
        return metrics

    def _write_metrics(self, metrics: Dict[str, float], stage: str) -> None:
        stage_path = self.output_dir / f"metrics_{stage}.json"
        with open(stage_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        log.info(">> Saved metrics: %s", stage_path)
        if stage == "test":
            final_path = self.output_dir / "metrics.json"
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

    def _plot_roc(self, df: pd.DataFrame, stage: str) -> None:
        fpr, tpr, _ = roc_curve(df["y_true"], df["prob"])
        auc = roc_auc_score(df["y_true"], df["prob"])

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({stage})")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        stage_path = self.output_dir / f"roc_curve_{stage}.png"
        plt.savefig(stage_path, dpi=150)
        plt.close()
        log.info(">> Saved ROC curve: %s", stage_path)
        if stage == "test":
            final_path = self.output_dir / "roc_curve.png"
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.3f}")
            plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (test)")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(final_path, dpi=150)
            plt.close()

    def _plot_reliability(self, df: pd.DataFrame, stage: str) -> None:
        if stage == "test" and self._has_umodule():
            return
        try:
            from sklearn.calibration import calibration_curve
        except ImportError:
            log.warning("sklearn.calibration unavailable; skip reliability diagram.")
            return

        try:
            ece_value, stats = compute_ece_metric(
                df["prob"].to_numpy(),
                df["y_true"].to_numpy(),
                n_bins=self._get_ece_bins(),
            )
            ece_bins = stats["bins_used"]
        except Exception:
            ece_value, ece_bins = 0.0, self._get_ece_bins()

        prob_true, prob_pred = calibration_curve(
            df["y_true"], df["prob"], n_bins=ece_bins, strategy="uniform"
        )
        plt.figure(figsize=(6, 5))
        plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed probability")
        plt.title(f"Reliability Diagram ({stage}) - ECE={ece_value:.4f}")
        plt.legend(loc="upper left")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        stage_path = self.output_dir / f"reliability_before_ts_{stage}.png"
        plt.savefig(stage_path, dpi=150)
        plt.close()
        log.info(">> Saved reliability diagram: %s", stage_path)
        if stage == "test":
            final_path = self.output_dir / "reliability_before_ts.png"
            plt.figure(figsize=(6, 5))
            plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Model")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
            plt.xlabel("Predicted probability")
            plt.ylabel("Observed probability")
            plt.title(f"Reliability Diagram (test) - ECE={ece_value:.4f}")
            plt.legend(loc="upper left")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(final_path, dpi=150)
            plt.close()

    def _maybe_write_splits(self) -> None:
        if self._splits_written or not self.split_metadata:
            return
        splits_path = self.output_dir / "data_splits.json"
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(self.split_metadata, f, indent=2, ensure_ascii=False)
        self._splits_written = True
        log.info(">> Saved split metadata: %s", splits_path)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Iterable):
            return torch.as_tensor(value)
        raise TypeError(f"Unsupported value type for tensor conversion: {type(value)}")

    @staticmethod
    def _to_list(value) -> List:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return value.cpu().tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _get_ece_bins(self) -> int:
        metrics_cfg = getattr(self.module, "metrics_cfg", None)
        if metrics_cfg and hasattr(metrics_cfg, "ece_bins"):
            return int(getattr(metrics_cfg, "ece_bins"))
        return 15

    def _has_umodule(self) -> bool:
        return bool(getattr(self.module, "umodule_enabled", False))

    def _stage_artifact_paths(self, stage: str) -> Dict[str, str]:
        paths = {
            "predictions": f"predictions_{stage}.csv",
            "metrics": f"metrics_{stage}.json",
            "roc_curve": f"roc_curve_{stage}.png",
        }
        default_reliability = f"reliability_before_ts_{stage}.png"
        if (self.output_dir / default_reliability).exists():
            paths["reliability_before"] = default_reliability
        custom_pre = f"reliability_pre_{stage}.png"
        custom_post = f"reliability_post_{stage}.png"
        for name, key in (
            (custom_pre, "reliability_pre"),
            (custom_post, "reliability_post"),
        ):
            if (self.output_dir / name).exists():
                paths[key] = name
        return paths
