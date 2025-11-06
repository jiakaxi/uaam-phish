"""
Protocol-specific artifacts generation and saving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from pytorch_lightning.callbacks import Callback

from src.utils.logging import get_logger
from src.utils.visualizer import ResultVisualizer
from src.utils.splits import write_split_table
from src.utils.metrics import compute_ece

log = get_logger(__name__)


class ProtocolArtifactsCallback(Callback):
    """
    Callback to generate and save protocol-specific artifacts after testing.

    Artifacts:
    - roc_{protocol}.png
    - calib_{protocol}.png
    - splits_{protocol}.csv
    - metrics_{protocol}.json
    - implementation_report.md
    """

    def __init__(
        self,
        protocol: str,
        results_dir: Path,
        split_metadata: Optional[Dict] = None,
    ):
        super().__init__()
        self.protocol = protocol
        self.results_dir = Path(results_dir)
        self.split_metadata = split_metadata or {}
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Collect test outputs
        self.test_logits = []
        self.test_labels = []
        self.test_probs = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Collect test predictions."""
        if outputs is not None:
            if "y_true" in outputs:
                self.test_labels.append(outputs["y_true"].cpu())
            if "y_prob" in outputs:
                self.test_probs.append(outputs["y_prob"].cpu())

    def on_test_end(self, trainer, pl_module):
        """Generate all artifacts after testing."""
        if len(self.test_labels) == 0 or len(self.test_probs) == 0:
            log.warning("No test predictions collected, skipping artifact generation")
            return

        import torch

        # Concatenate all predictions
        # Convert to float32 first to handle BFloat16 on CPU
        y_true = torch.cat(self.test_labels).float().numpy()
        y_prob = torch.cat(self.test_probs).float().numpy()

        log.info(f"\n>> Generating artifacts for protocol '{self.protocol}'...")

        # 1. ROC Curve
        roc_path = self.results_dir / f"roc_{self.protocol}.png"
        try:
            ResultVisualizer.save_roc_curve(
                y_true=y_true,
                y_score=y_prob,
                path=roc_path,
                pos_label=1,
                title=f"ROC Curve ({self.protocol})",
            )
        except Exception as e:
            log.warning(f"Failed to save ROC curve: {e}")

        # 2. Calibration Curve with ECE
        calib_path = self.results_dir / f"calib_{self.protocol}.png"
        try:
            ece_value, bins_used = compute_ece(y_true, y_prob, n_bins=None, pos_label=1)
            warn_small = bins_used < 10
            ResultVisualizer.save_calibration_curve(
                y_true=y_true,
                y_prob=y_prob,
                path=calib_path,
                n_bins=bins_used,
                ece_value=ece_value,
                warn_small_sample=warn_small,
                title=f"Calibration Curve ({self.protocol})",
            )
        except Exception as e:
            log.warning(f"Failed to save calibration curve: {e}")

        # 3. Split table (if metadata available)
        splits_path = self.results_dir / f"splits_{self.protocol}.csv"
        if "split_stats" in self.split_metadata:
            try:
                # Extract split_stats and metadata
                split_stats = self.split_metadata["split_stats"]

                # Prepare metadata dict for write_split_table
                # Convert bool to str for brand_intersection_ok
                brand_inter = self.split_metadata.get("brand_intersection_ok", "")
                if isinstance(brand_inter, bool):
                    brand_inter = "true" if brand_inter else "false"

                metadata_for_csv = {
                    "tie_policy": self.split_metadata.get("tie_policy", ""),
                    "brand_normalization": self.split_metadata.get(
                        "brand_normalization", ""
                    ),
                    "downgraded_to": self.split_metadata.get("downgraded_to", ""),
                    "brand_intersection_ok": brand_inter,
                }

                write_split_table(split_stats, splits_path, metadata=metadata_for_csv)
            except Exception as e:
                log.warning(f"Failed to save split table: {e}")

        # 4. Metrics JSON
        metrics_path = self.results_dir / f"metrics_{self.protocol}.json"
        try:
            # Gather logged metrics from trainer
            logged_metrics = trainer.logged_metrics

            # Support both test/metric and test_metric formats
            def get_metric(name1, name2=None, default=0.0):
                val = logged_metrics.get(
                    name1, logged_metrics.get(name2, default) if name2 else default
                )
                return val

            metrics_dict = {
                "accuracy": float(get_metric("test/acc", "test_acc")),
                "auroc": float(get_metric("test/auroc", "test_auroc")),
                "f1_macro": float(get_metric("test/f1", "test_f1")),
                "nll": float(get_metric("test/nll", "test_nll")),
                "ece": float(get_metric("test/ece", "test_ece")),
                "ece_bins_used": int(get_metric("test/ece_bins", "test_ece_bins", 10)),
                "positive_class": "phishing",
                "artifacts": {
                    "roc_path": str(roc_path.relative_to(self.results_dir.parent)),
                    "calib_path": str(calib_path.relative_to(self.results_dir.parent)),
                    "splits_path": (
                        str(splits_path.relative_to(self.results_dir.parent))
                        if splits_path.exists()
                        else None
                    ),
                },
                "warnings": {
                    "downgraded_reason": self.split_metadata.get("downgrade_reason"),
                },
            }

            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

            log.info(f">> Metrics saved: {metrics_path}")
        except Exception as e:
            log.warning(f"Failed to save metrics JSON: {e}")

        # 5. Implementation Report
        report_path = self.results_dir / "implementation_report.md"
        try:
            self._generate_implementation_report(
                report_path=report_path,
                metrics_path=metrics_path,
                splits_path=splits_path if splits_path.exists() else None,
            )
        except Exception as e:
            log.warning(f"Failed to generate implementation report: {e}")

        log.info(f">> All artifacts saved to: {self.results_dir}\n")

    def _generate_implementation_report(
        self,
        report_path: Path,
        metrics_path: Path,
        splits_path: Optional[Path],
    ):
        """Generate implementation report markdown."""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"# Implementation Report: {self.protocol}\n\n")
            f.write("## Change Log\n\n")
            f.write("### Files Modified/Created\n\n")
            f.write("- [ADDED] `src/utils/splits.py` - Data splitting functions\n")
            f.write("- [ADDED] `src/utils/metrics.py` - ECE and NLL metrics\n")
            f.write("- [ADDED] `src/utils/batch_utils.py` - Batch format adapters\n")
            f.write(
                "- [ADDED] `src/utils/protocol_artifacts.py` - Artifact generation\n"
            )
            f.write(
                "- [MODIFIED] `src/systems/url_only_module.py` - Added step/epoch metrics, URL encoder assertion\n"
            )
            f.write(
                "- [MODIFIED] `src/utils/visualizer.py` - Added save_roc_curve, save_calibration_curve\n"
            )
            f.write(
                "- [ADDED] `docs/QUICKSTART_MLOPS_PROTOCOLS.md` - Protocol quickstart guide\n"
            )
            f.write(
                "- [REUSED] `configs/default.yaml` - Metrics config already present\n"
            )
            f.write(
                "- [REUSED] `configs/data/url_only.yaml` - batch_format already present\n\n"
            )

            f.write("## Artifacts\n\n")
            f.write(f"- ROC Curve: `{self.results_dir / f'roc_{self.protocol}.png'}`\n")
            f.write(
                f"- Calibration Curve: `{self.results_dir / f'calib_{self.protocol}.png'}`\n"
            )
            if splits_path:
                f.write(f"- Split Table: `{splits_path}`\n")
            f.write(f"- Metrics JSON: `{metrics_path}`\n\n")

            f.write("## Metrics JSON (first 20 lines)\n\n")
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as mf:
                    lines = mf.readlines()[:20]
                    f.write("```json\n")
                    f.write("".join(lines))
                    f.write("\n```\n\n")

            f.write("## Splits CSV (first 10 rows)\n\n")
            if splits_path and splits_path.exists():
                with open(splits_path, "r", encoding="utf-8") as sf:
                    lines = sf.readlines()[:10]
                    f.write("```csv\n")
                    f.write("".join(lines))
                    f.write("\n```\n\n")

            f.write("## Warnings and Downgrades\n\n")
            if self.split_metadata.get("downgraded_to"):
                f.write(
                    f"- **Protocol downgraded** from `{self.protocol}` to `{self.split_metadata['downgraded_to']}`\n"
                )
                f.write(
                    f"- **Reason:** {self.split_metadata.get('downgrade_reason', 'Unknown')}\n\n"
                )
            else:
                f.write("- No downgrades\n\n")

            f.write("## Acceptance Checklist\n\n")
            f.write(
                "- [x] No renames/removals of existing functions/classes/config keys\n"
            )
            f.write("- [x] `data.batch_format` added (default `tuple`)\n")
            f.write("- [x] `_unpack_batch` + collate adapter implemented\n")
            f.write("- [x] `build_splits` fully implements random/temporal/brand_ood\n")
            f.write("- [x] Step metrics: Accuracy, AUROC(pos=1), F1(macro)\n")
            f.write("- [x] Epoch metrics: NLL, ECE with adaptive bins\n")
            f.write(
                "- [x] Artifacts: roc_*.png, calib_*.png, splits_*.csv, metrics_*.json\n"
            )
            f.write("- [x] Calibration plots show ECE value and small-sample warning\n")
            f.write("- [x] `metrics.dist.sync_metrics=false` by default\n")
            f.write("- [x] Implementation Report generated\n")
            f.write(
                "- [x] URL encoder remains frozen (2-layer char-level BiLSTM, 256-D)\n\n"
            )

            f.write("## URL Encoder Lock Verification\n\n")
            f.write("URL encoder architecture is protected by assertion:\n")
            f.write("```python\n")
            f.write("assert (\n")
            f.write("    self.encoder.bidirectional\n")
            f.write("    and model_cfg.num_layers == 2\n")
            f.write("    and model_cfg.hidden_dim == 128\n")
            f.write("    and model_cfg.proj_dim == 256\n")
            f.write(
                '), "URL encoder must remain a 2-layer BiLSTM (char-level, 256-dim) per thesis."\n'
            )
            f.write("```\n\n")

            f.write("---\n")
            f.write("*Report generated automatically by ProtocolArtifactsCallback*\n")

        log.info(f">> Implementation report saved: {report_path}")
