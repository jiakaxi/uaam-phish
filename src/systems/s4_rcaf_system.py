"""
S4 RCAF Full System - Adaptive Fusion with Learned Lambda_c.

This system implements the complete S4 RCAF (Reliability-Consistency Adaptive Fusion)
mechanism where lambda_c is learned per-sample via a small attention network.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score

from src.models.url_encoder import URLEncoder
from src.models.html_encoder import HTMLEncoder
from src.models.visual_encoder import VisualEncoder
from src.modules.u_module import UModule
from src.modules.c_module import CModule
from src.modules.fusion.adaptive_fusion import AdaptiveFusion
from src.utils.logging import get_logger


log = get_logger(__name__)


class S4RCAFSystem(pl.LightningModule):
    """
    S4 RCAF Full System with learned lambda_c for adaptive fusion.

    Key differences from S3:
    - lambda_c is learned per-sample via LambdaGate, not a fixed hyperparameter
    - Training uses adaptive fusion (not LateAvg), ensuring gradient flow to gate
    - Monitors lambda_c statistics (mean, std) for collapse detection
    """

    def __init__(
        self,
        # Encoder params (same as S0)
        url_vocab_size: int = 128,
        url_embed_dim: int = 64,
        url_hidden_dim: int = 128,
        url_num_layers: int = 2,
        url_max_len: int = 200,
        html_model_name: str = "bert-base-uncased",
        html_hidden_dim: int = 768,
        html_projection_dim: int = 256,
        html_freeze_bert: bool = False,
        visual_model_name: str = "resnet50",
        visual_pretrained: bool = True,
        visual_projection_dim: int = 256,
        visual_freeze_backbone: bool = False,
        # Training params
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        pos_weight: Optional[float] = None,
        # Config
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        # Validate required modules
        assert cfg is not None, "cfg is required for S4RCAFSystem"
        modules_cfg = getattr(cfg, "modules", None)
        assert modules_cfg is not None, "cfg.modules is required"
        assert getattr(modules_cfg, "use_umodule", False), "S4 requires U-Module"
        assert getattr(modules_cfg, "use_cmodule", False), "S4 requires C-Module"

        # Enable TF32 for faster training on Ampere+ GPUs
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # ===== Encoders (复用 S0 架构) =====
        self.url_encoder = URLEncoder(
            vocab_size=url_vocab_size,
            embedding_dim=url_embed_dim,
            hidden_dim=url_hidden_dim,
            num_layers=url_num_layers,
            bidirectional=True,
            dropout=dropout,
            proj_dim=256,
        )
        self.html_encoder = HTMLEncoder(
            bert_model=html_model_name,
            hidden_dim=html_hidden_dim,
            output_dim=html_projection_dim,
            dropout=dropout,
            freeze_bert=html_freeze_bert,
        )
        self.visual_encoder = VisualEncoder(
            pretrained=visual_pretrained,
            freeze_backbone=visual_freeze_backbone,
            embedding_dim=visual_projection_dim,
            dropout=dropout,
        )

        # Classification heads (logits output)
        self.url_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))
        self.html_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))
        self.visual_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))

        # ===== Trustworthiness Modules =====
        # U-Module (Reliability)
        umodule_cfg = getattr(cfg, "umodule", None)
        self.u_module = UModule(
            mc_iters=getattr(umodule_cfg, "mc_iters", 10),
            dropout_p=dropout,
            init_temperature=getattr(umodule_cfg, "temperature_init", 1.0),
            lambda_u=getattr(umodule_cfg, "lambda_u", 1.0),
            learnable=getattr(umodule_cfg, "learnable", False),
        )

        # C-Module (Consistency)
        c_module_cfg = getattr(modules_cfg, "c_module", None) if modules_cfg else None
        c_module_thresh = getattr(c_module_cfg, "thresh", 0.6) if c_module_cfg else 0.6

        # Gather metadata sources (CSV files with url_text, html_text, etc.)
        metadata_sources = self._gather_metadata_sources()
        log.info(f"[S4] Gathered {len(metadata_sources)} metadata sources for C-Module")
        for src in metadata_sources:
            log.debug(f"[S4] Metadata source: {src}")

        self.c_module = CModule(
            model_name=(
                getattr(
                    c_module_cfg,
                    "model_name",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )
                if c_module_cfg
                else "sentence-transformers/all-MiniLM-L6-v2"
            ),
            thresh=c_module_thresh,
            brand_lexicon_path=(
                getattr(c_module_cfg, "brand_lexicon_path", None)
                if c_module_cfg
                else None
            ),
            use_ocr=getattr(c_module_cfg, "use_ocr", False) if c_module_cfg else False,
            metadata_sources=metadata_sources,
        )

        # ===== Adaptive Fusion Module =====
        fusion_cfg = getattr(cfg, "fusion", None) if hasattr(cfg, "fusion") else None
        if fusion_cfg is None:
            # Fallback to system-level fusion config
            fusion_cfg = (
                getattr(getattr(cfg, "system", None), "fusion", None)
                if hasattr(cfg, "system")
                else None
            )

        hidden_dim = getattr(fusion_cfg, "hidden_dim", 16) if fusion_cfg else 16
        temperature = getattr(fusion_cfg, "temperature", 2.0) if fusion_cfg else 2.0
        self.lambda_regularization = (
            getattr(fusion_cfg, "lambda_regularization", 0.01) if fusion_cfg else 0.01
        )
        self.warmup_epochs = (
            getattr(fusion_cfg, "warmup_epochs", 0) if fusion_cfg else 0
        )

        self.adaptive_fusion = AdaptiveFusion(
            num_modalities=3,
            num_classes=2,
            hidden_dim=hidden_dim,
            temperature=temperature,
        )

        # ===== Loss and Metrics =====
        self._pos_weight = float(pos_weight) if pos_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.train_f1 = F1Score(task="binary")

        self.val_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary")

        self.test_acc = Accuracy(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_f1 = F1Score(task="binary")

        # ===== Logging Buffers =====
        self.modalities = ("url", "html", "visual")

        # Training buffers for lambda_c monitoring
        self.train_lambda_c_buffer: List[torch.Tensor] = []

        # Test buffers for scenario-specific analysis
        self.test_lambda_buffer: List[torch.Tensor] = []
        self.test_alpha_buffer: List[torch.Tensor] = []
        self.test_scenario_buffer: List[str] = []
        self.test_pred_buffer: List[torch.Tensor] = []
        self.test_label_buffer: List[torch.Tensor] = []
        self.test_sample_id_buffer: List[str] = []

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoders and adaptive fusion.

        Returns dict with keys:
            - probs: Fused probabilities [B, 2]
            - alpha_m: Fusion weights [B, 3]
            - lambda_c: Adaptive consistency weights [B, 3]
            - U_m: Unified trust scores [B, 3]
            - logits_url/html/visual: Per-modality logits
        """
        # Encode each modality
        url_emb = self.url_encoder(batch["url"])
        html_emb = self.html_encoder(
            batch["html"]["input_ids"], batch["html"]["attention_mask"]
        )
        vis_emb = self.visual_encoder(batch["visual"])

        # Get logits
        url_logits = self.url_head(url_emb).squeeze(-1)  # [B]
        html_logits = self.html_head(html_emb).squeeze(-1)
        visual_logits = self.visual_head(vis_emb).squeeze(-1)

        # NaN check on logits before sigmoid
        if torch.isnan(url_logits).any():
            log.error(
                f"[Forward] URL logits NaN! mean={url_logits[~torch.isnan(url_logits)].mean() if (~torch.isnan(url_logits)).any() else 'all_nan'}"
            )
        if torch.isnan(html_logits).any():
            log.error(
                f"[Forward] HTML logits NaN! mean={html_logits[~torch.isnan(html_logits)].mean() if (~torch.isnan(html_logits)).any() else 'all_nan'}"
            )
        if torch.isnan(visual_logits).any():
            log.error(
                f"[Forward] Visual logits NaN! mean={visual_logits[~torch.isnan(visual_logits)].mean() if (~torch.isnan(visual_logits)).any() else 'all_nan'}"
            )

        # Convert logits to probabilities (binary classification)
        # Replace NaN logits with 0 (neutral) before sigmoid
        url_logits_clean = torch.nan_to_num(
            url_logits, nan=0.0, posinf=10.0, neginf=-10.0
        )
        html_logits_clean = torch.nan_to_num(
            html_logits, nan=0.0, posinf=10.0, neginf=-10.0
        )
        visual_logits_clean = torch.nan_to_num(
            visual_logits, nan=0.0, posinf=10.0, neginf=-10.0
        )

        p_url = torch.sigmoid(url_logits_clean)
        p_html = torch.sigmoid(html_logits_clean)
        p_visual = torch.sigmoid(visual_logits_clean)

        probs_url = torch.stack([1 - p_url, p_url], dim=-1)  # [B, 2]
        probs_html = torch.stack([1 - p_html, p_html], dim=-1)
        probs_visual = torch.stack([1 - p_visual, p_visual], dim=-1)

        # Get reliability scores (U-Module)
        r_url = self._compute_reliability(url_logits, "url")
        r_html = self._compute_reliability(html_logits, "html")
        r_visual = self._compute_reliability(visual_logits, "visual")
        r_m = torch.stack([r_url, r_html, r_visual], dim=1)  # [B, 3]

        # Get consistency scores (C-Module) - process per sample
        c_m = self._compute_consistency_batch(batch)  # [B, 3]

        # Normalize c_m to [0, 1] from [-1, 1]
        c_m_normalized = (c_m + 1.0) * 0.5

        # Adaptive fusion
        p_fused, alpha_m, lambda_c, U_m = self.adaptive_fusion(
            [probs_url, probs_html, probs_visual],
            r_m,
            c_m_normalized,
        )

        return {
            "probs": p_fused,
            "alpha_m": alpha_m,
            "lambda_c": lambda_c,
            "U_m": U_m,
            "logits_url": url_logits,
            "logits_html": html_logits,
            "logits_visual": visual_logits,
        }

    def _compute_consistency_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute consistency scores for a batch using C-Module."""
        batch_size = len(batch["id"])
        device = next(self.parameters()).device

        # Extract batch fields
        sample_ids = self._batch_to_list(batch.get("id"))
        image_paths = self._batch_to_list(batch.get("image_path"))
        html_paths = self._batch_to_list(batch.get("html_path"))
        url_texts = self._batch_to_list(batch.get("url_text"))

        # Initialize scores
        c_url_list = []
        c_html_list = []
        c_visual_list = []

        # Process each sample in batch
        for idx in range(batch_size):
            # Build sample dict for C-Module with all available fields
            sample = {
                "sample_id": sample_ids[idx] if idx < len(sample_ids) else None,
                "id": sample_ids[idx] if idx < len(sample_ids) else None,
                "url_text": url_texts[idx] if idx < len(url_texts) else "",
                "html_path": html_paths[idx] if idx < len(html_paths) else None,
                "image_path": image_paths[idx] if idx < len(image_paths) else None,
            }

            # Call C-Module
            result = self.c_module.score_consistency(sample)

            # Extract per-modality scores
            c_url = float(result.get("c_url", 0.0))
            c_html = float(result.get("c_html", 0.0))
            c_visual = float(result.get("c_visual", 0.0))

            c_url_list.append(c_url)
            c_html_list.append(c_html)
            c_visual_list.append(c_visual)

        # Stack into tensor [B, 3]
        c_m = torch.tensor(
            [
                [c_url_list[i], c_html_list[i], c_visual_list[i]]
                for i in range(batch_size)
            ],
            dtype=torch.float32,
            device=device,
        )

        # NaN fallback: replace NaN/Inf with 0.0 (no consistency signal)
        # This allows fusion to proceed with only reliability scores
        c_m = torch.nan_to_num(c_m, nan=0.0, posinf=0.0, neginf=0.0)

        return c_m

    def _compute_reliability(self, logits: torch.Tensor, modality: str) -> torch.Tensor:
        """Compute reliability score using U-Module (uncertainty quantification)."""
        # Simple entropy-based uncertainty for now
        # In full implementation, would use MC Dropout
        probs = torch.sigmoid(logits)

        # Clamp to avoid log(0) which causes NaN
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)

        # Binary entropy
        entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)

        # Normalize entropy to [0, 1] (binary entropy max is ln(2) ≈ 0.693)
        # reliability = 1 - normalized_entropy
        reliability = 1.0 - (entropy / 0.693)

        # NaN fallback: use default medium reliability if computation failed
        reliability = torch.nan_to_num(reliability, nan=0.5, posinf=0.5, neginf=0.5)

        return reliability

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step using adaptive fusion."""
        outputs = self(batch)

        # Get fused probabilities [B, 2]
        p_fused = outputs["probs"]

        # Convert to logits for BCEWithLogitsLoss
        # Since p_fused is already probabilities, we need to convert back
        # Using log-odds: logit = log(p / (1 - p))
        # For binary case, we use class 1 probability
        p_pos = p_fused[:, 1]  # Probability of positive class

        # NaN detection before logit conversion
        if torch.isnan(p_pos).any():
            nan_indices = torch.isnan(p_pos).nonzero(as_tuple=True)[0]
            sample_ids = batch.get("id", ["<unknown>"] * len(p_pos))
            url_texts = batch.get("url_text", ["<unknown>"] * len(p_pos))
            html_paths = batch.get("html_path", ["<unknown>"] * len(p_pos))
            log.error(
                f"[NaN Detection] batch_idx={batch_idx}, "
                f"affected_samples={len(nan_indices)}/{len(p_pos)}"
            )
            for idx in nan_indices[:3]:  # Log first 3 affected samples
                i = idx.item()
                log.error(
                    f"  Sample {i}: id={sample_ids[i]}, "
                    f"url={url_texts[i][:60] if i < len(url_texts) else 'N/A'}, "
                    f"html={html_paths[i] if i < len(html_paths) else 'N/A'}"
                )

        logits_fused = torch.log(p_pos / (1 - p_pos + 1e-8) + 1e-8)

        labels = batch["label"].float()

        # Classification loss
        cls_loss = self.loss_fn(logits_fused, labels)

        # L2 regularization on lambda gate parameters ONLY
        reg_loss = 0.0
        if self.lambda_regularization > 0:
            lambda_params = list(self.adaptive_fusion.lambda_gate.parameters())
            l2_norm = sum(p.pow(2).sum() for p in lambda_params)
            reg_loss = self.lambda_regularization * l2_norm

        # Total loss
        total_loss = cls_loss + reg_loss

        # Metrics
        preds = (p_pos > 0.5).long()
        self.train_acc(preds, batch["label"])
        self.train_auroc(p_pos, batch["label"])
        self.train_f1(preds, batch["label"])

        # Store lambda_c for monitoring
        self.train_lambda_c_buffer.append(outputs["lambda_c"].detach())

        # Logging
        self.log(
            "train/cls_loss", cls_loss, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/reg_loss", reg_loss, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=False
        )

        return total_loss

    def on_train_epoch_end(self) -> None:
        """Monitor lambda_c statistics at end of training epoch."""
        if self.train_lambda_c_buffer:
            lambda_c_all = torch.cat(self.train_lambda_c_buffer, dim=0)  # [N, 3]
            lambda_c_mean = lambda_c_all.mean().item()
            lambda_c_std = lambda_c_all.std().item()

            self.log("train/lambda_c_mean", lambda_c_mean)
            self.log("train/lambda_c_std", lambda_c_std)

            # Sanity checks
            if lambda_c_std < 0.05:
                warnings.warn(
                    f"⚠️ Lambda_c collapsed! std={lambda_c_std:.4f} < 0.05. "
                    "Consider adding regularization or adjusting learning rate."
                )
            if lambda_c_mean < 0.2 or lambda_c_mean > 0.8:
                warnings.warn(
                    f"⚠️ Lambda_c mean={lambda_c_mean:.4f} out of range [0.2, 0.8]. "
                    "May indicate training instability."
                )

            # Clear buffer
            self.train_lambda_c_buffer.clear()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step."""
        outputs = self(batch)
        p_fused = outputs["probs"]
        p_pos = p_fused[:, 1]

        preds = (p_pos > 0.5).long()
        self.val_acc(preds, batch["label"])
        self.val_auroc(p_pos, batch["label"])
        self.val_f1(preds, batch["label"])

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/auroc", self.val_auroc, prog_bar=True)
        self.log("val/f1", self.val_f1, prog_bar=False)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Test step - accumulate results for scenario-specific analysis."""
        outputs = self(batch)
        p_fused = outputs["probs"]
        p_pos = p_fused[:, 1]

        preds = (p_pos > 0.5).long()
        self.test_acc(preds, batch["label"])
        self.test_auroc(p_pos, batch["label"])
        self.test_f1(preds, batch["label"])

        # Accumulate for scenario-specific analysis
        self.test_lambda_buffer.append(outputs["lambda_c"].detach().cpu())
        self.test_alpha_buffer.append(outputs["alpha_m"].detach().cpu())
        self.test_pred_buffer.append(p_pos.detach().cpu())
        self.test_label_buffer.append(batch["label"].detach().cpu())

        # Extract scenario labels from meta
        if "meta" in batch and "scenario" in batch["meta"]:
            scenarios = batch["meta"]["scenario"]
            self.test_scenario_buffer.extend(scenarios)
        else:
            # Fallback to 'unknown' if meta not available
            self.test_scenario_buffer.extend(["unknown"] * len(batch["label"]))

        # Extract sample IDs
        if "id" in batch:
            ids = batch["id"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            self.test_sample_id_buffer.extend([str(x) for x in ids])
        else:
            self.test_sample_id_buffer.extend(
                [f"sample_{batch_idx}_{i}" for i in range(len(batch["label"]))]
            )

    def on_test_epoch_end(self) -> None:
        """Generate test metrics and output files."""
        # Compute overall metrics
        self.log("test/acc", self.test_acc)
        self.log("test/auroc", self.test_auroc)
        self.log("test/f1", self.test_f1)

        # Generate output files if we have test results
        if self.test_lambda_buffer:
            self._generate_output_files()

        # Clear buffers
        self.test_lambda_buffer.clear()
        self.test_alpha_buffer.clear()
        self.test_scenario_buffer.clear()
        self.test_pred_buffer.clear()
        self.test_label_buffer.clear()
        self.test_sample_id_buffer.clear()

    def _generate_output_files(self) -> None:
        """Generate s4_lambda_stats.json and s4_per_sample.csv."""
        # Concatenate all buffers
        lambda_c_all = torch.cat(self.test_lambda_buffer, dim=0).numpy()  # [N, 3]
        alpha_m_all = torch.cat(self.test_alpha_buffer, dim=0).numpy()  # [N, 3]
        preds_all = torch.cat(self.test_pred_buffer, dim=0).numpy()  # [N]
        labels_all = torch.cat(self.test_label_buffer, dim=0).numpy()  # [N]
        scenarios_all = self.test_scenario_buffer  # List[str]
        sample_ids_all = self.test_sample_id_buffer  # List[str]

        # Get logger directory
        if hasattr(self.logger, "log_dir") and self.logger.log_dir:
            log_dir = Path(self.logger.log_dir)
        elif hasattr(self.logger, "save_dir") and self.logger.save_dir:
            log_dir = Path(self.logger.save_dir)
        else:
            log_dir = Path(".")

        log_dir.mkdir(parents=True, exist_ok=True)

        # ===== Generate s4_lambda_stats.json =====
        stats_by_scenario = {}
        unique_scenarios = set(scenarios_all)

        for scenario in unique_scenarios:
            # Get indices for this scenario
            indices = [i for i, s in enumerate(scenarios_all) if s == scenario]
            if not indices:
                continue

            lambda_c_scenario = lambda_c_all[indices]  # [N_scenario, 3]
            alpha_m_scenario = alpha_m_all[indices]  # [N_scenario, 3]

            stats_by_scenario[scenario] = {
                "lambda_c": {
                    "mean": float(lambda_c_scenario.mean()),
                    "std": float(lambda_c_scenario.std()),
                    "min": float(lambda_c_scenario.min()),
                    "max": float(lambda_c_scenario.max()),
                },
                "alpha_m": {
                    "url": {
                        "mean": float(alpha_m_scenario[:, 0].mean()),
                        "std": float(alpha_m_scenario[:, 0].std()),
                    },
                    "html": {
                        "mean": float(alpha_m_scenario[:, 1].mean()),
                        "std": float(alpha_m_scenario[:, 1].std()),
                    },
                    "visual": {
                        "mean": float(alpha_m_scenario[:, 2].mean()),
                        "std": float(alpha_m_scenario[:, 2].std()),
                    },
                },
                "count": len(indices),
            }

        stats_path = log_dir / "s4_lambda_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats_by_scenario, f, indent=2)

        log.info(f"[S4] Saved lambda statistics to {stats_path}")

        # ===== Generate s4_per_sample.csv =====
        df = pd.DataFrame(
            {
                "sample_id": sample_ids_all,
                "scenario": scenarios_all,
                "lambda_c_url": lambda_c_all[:, 0],
                "lambda_c_html": lambda_c_all[:, 1],
                "lambda_c_visual": lambda_c_all[:, 2],
                "alpha_url": alpha_m_all[:, 0],
                "alpha_html": alpha_m_all[:, 1],
                "alpha_visual": alpha_m_all[:, 2],
                "pred": preds_all,
                "label": labels_all,
            }
        )

        csv_path = log_dir / "s4_per_sample.csv"
        df.to_csv(csv_path, index=False)

        log.info(f"[S4] Saved per-sample data to {csv_path}")

    def configure_optimizers(self) -> Any:
        """Configure optimizer with optional separate learning rates."""
        # Check if we have separate learning rates for encoders and fusion
        encoder_lr = self.hparams.learning_rate
        fusion_lr = self.hparams.learning_rate

        # Try to get separate LRs from config
        if hasattr(self.cfg, "optimizer"):
            encoder_lr = getattr(self.cfg.optimizer, "encoder_lr", encoder_lr)
            fusion_lr = getattr(self.cfg.optimizer, "fusion_lr", fusion_lr)

        # Group parameters
        encoder_params = []
        encoder_params.extend(list(self.url_encoder.parameters()))
        encoder_params.extend(list(self.html_encoder.parameters()))
        encoder_params.extend(list(self.visual_encoder.parameters()))
        encoder_params.extend(list(self.url_head.parameters()))
        encoder_params.extend(list(self.html_head.parameters()))
        encoder_params.extend(list(self.visual_head.parameters()))

        fusion_params = list(self.adaptive_fusion.parameters())

        # Create optimizer with parameter groups
        optimizer = torch.optim.Adam(
            [
                {"params": encoder_params, "lr": encoder_lr},
                {"params": fusion_params, "lr": fusion_lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        # Optional: Add scheduler
        if (
            hasattr(self.cfg, "scheduler")
            and getattr(self.cfg.scheduler, "type", None) == "cosine"
        ):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=encoder_lr * 0.1,
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

        return optimizer

    @staticmethod
    def _decode_url_tokens(url_tensor: torch.Tensor) -> List[str]:
        """
        Decode tokenized URLs back to strings for C-Module brand extraction.

        Copied from S0LateAverageSystem to ensure inline text fallback.
        """
        if not isinstance(url_tensor, torch.Tensor):
            return []
        if url_tensor.dim() == 1:
            url_tensor = url_tensor.unsqueeze(0)
        rows = url_tensor.detach().cpu().tolist()
        urls: List[str] = []
        for row in rows:
            chars: List[str] = []
            for value in row:
                code = int(value)
                if code <= 0:
                    break
                code = min(max(code, 32), 255)
                try:
                    chars.append(chr(code))
                except ValueError:
                    continue
            urls.append("".join(chars))
        return urls

    @staticmethod
    def _batch_to_list(field: Any) -> List[Any]:
        """Convert batch field to list format."""
        if field is None:
            return []
        if isinstance(field, (list, tuple)):
            return list(field)
        if isinstance(field, torch.Tensor):
            return field.detach().cpu().tolist()
        return [field]

    def _gather_metadata_sources(self) -> List[str]:
        """
        Gather metadata CSV sources for C-Module.

        Copied from S0LateAverageSystem to ensure C-Module can access
        url_text, html_text, and other raw data for brand extraction.
        """
        datamodule_cfg = getattr(self.cfg, "datamodule", None)
        if datamodule_cfg is None:
            return []

        seen: set[str] = set()
        sources: List[str] = []

        for attr in ("train_csv", "val_csv", "test_csv", "test_ood_csv"):
            raw = getattr(datamodule_cfg, attr, None)
            if not raw:
                continue

            for candidate in self._expand_csv_candidates(str(raw)):
                if candidate in seen:
                    continue
                seen.add(candidate)
                sources.append(candidate)

        return sources

    @staticmethod
    def _expand_csv_candidates(path_str: str) -> List[str]:
        """
        Expand CSV path to include cached variants.

        Returns both original and *_cached.csv versions.
        """
        path = Path(path_str)
        candidates = [str(path)]

        cached = path.with_name(f"{path.stem}_cached{path.suffix}")
        if cached != path:
            candidates.append(str(cached))

        return candidates


if __name__ == "__main__":
    print("[OK] S4RCAFSystem module loaded successfully.")
