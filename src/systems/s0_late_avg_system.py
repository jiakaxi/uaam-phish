"""
Late-fusion (uniform average) multimodal baseline for S0 experiments.
"""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.dropout import _DropoutNd
from torchmetrics import Accuracy, AUROC, F1Score
from sklearn.metrics import roc_auc_score

from src.models.url_encoder import URLEncoder
from src.models.html_encoder import HTMLEncoder
from src.models.visual_encoder import VisualEncoder
from src.modules.c_module import CModule
from src.modules.u_module import (
    UModule,
    mc_dropout_predict,
    temperature_scaling,
)
from src.utils.protocol_artifacts import ArtifactsWriter
from src.utils.visualizer import ResultVisualizer
from src.utils.metrics import ece as compute_ece_metric, brier_score
from src.utils.logging import get_logger


log = get_logger(__name__)


class S0LateAverageSystem(pl.LightningModule):
    """
    Late-fusion baseline: average the probabilities of three modality-specific heads.
    """

    def __init__(
        self,
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
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        pos_weight: Optional[float] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        # Enable TF32 for faster training on Ampere+ GPUs
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # Encoders
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

        self.url_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))
        self.html_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))
        self.visual_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))

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

        self.modalities = ("url", "html", "visual")
        self.metrics_cfg = getattr(cfg, "metrics", None)
        self.modules_cfg = getattr(cfg, "modules", None)
        self.umodule_cfg = getattr(cfg, "umodule", None)
        self.umodule_enabled = self._should_enable_umodule()
        self.u_module: Optional[UModule] = None
        if self.umodule_enabled:
            self.u_module = UModule(
                mc_iters=getattr(self.umodule_cfg, "mc_iters", 10),
                dropout_p=getattr(self.umodule_cfg, "dropout", dropout),
                init_temperature=getattr(self.umodule_cfg, "temperature_init", 1.0),
                lambda_u=getattr(self.umodule_cfg, "lambda_u", 1.0),
                learnable=getattr(self.umodule_cfg, "learnable", False),
            )
        self.c_module: Optional[CModule] = None
        self.c_module_cfg = (
            getattr(self.modules_cfg, "c_module", None) if self.modules_cfg else None
        )
        self.cmodule_enabled = bool(
            getattr(self.modules_cfg, "use_cmodule", False)
            if self.modules_cfg
            else False
        )
        self.fusion_mode: str = (
            getattr(self.modules_cfg, "fusion_mode", "lateavg")
            if self.modules_cfg
            else "lateavg"
        )
        self.lambda_c: float = float(
            getattr(self.modules_cfg, "lambda_c", 0.5) if self.modules_cfg else 0.5
        )
        self.c_module_threshold = self._resolve_consistency_thresh()
        if self.cmodule_enabled:
            metadata_sources = self._gather_metadata_sources()
            self.c_module = CModule(
                model_name=getattr(
                    self.c_module_cfg,
                    "model_name",
                    "sentence-transformers/all-MiniLM-L6-v2",
                ),
                thresh=self.c_module_threshold,
                brand_lexicon_path=getattr(
                    self.c_module_cfg, "brand_lexicon_path", None
                ),
                use_ocr=bool(getattr(self.c_module_cfg, "use_ocr", False)),
                metadata_sources=metadata_sources,
            )

        self._dropout_layers: list[_DropoutNd] = []
        self._cache_dropout_layers()

        tracked_stages = ("val", "test")
        self._stage_labels: Dict[str, list[torch.Tensor]] = {
            stage: [] for stage in tracked_stages
        }
        self._modal_outputs: Dict[str, Dict[str, Dict[str, list[torch.Tensor]]]] = {
            stage: {
                mod: {
                    "logits": [],
                    "probs": [],
                    "probs_ts": [],
                    "reliability": [],
                    "var": [],
                }
                for mod in self.modalities
            }
            for stage in tracked_stages
        }
        self._fused_probs: Dict[str, list[torch.Tensor]] = {
            stage: [] for stage in tracked_stages
        }
        self._fused_post_probs: Dict[str, list[torch.Tensor]] = {
            stage: [] for stage in tracked_stages
        }
        self._calibration_summary: Dict[str, Dict[str, float | str | int]] = {}
        self._reliability_column_map = {
            "url": "r_url",
            "html": "r_html",
            "visual": "r_img",
        }
        self._consistency_scores: Dict[str, List[float]] = {
            stage: [] for stage in ("val", "test")
        }
        self._fusion_probs: Dict[str, List[torch.Tensor]] = {
            stage: [] for stage in tracked_stages
        }
        self._alpha_history: Dict[str, Dict[str, List[torch.Tensor]]] = {
            stage: {mod: [] for mod in self.modalities} for stage in tracked_stages
        }
        self._u_history: Dict[str, Dict[str, List[torch.Tensor]]] = {
            stage: {mod: [] for mod in self.modalities} for stage in tracked_stages
        }

        self.artifacts_dir: Optional[Path] = None
        self.results_dir: Optional[Path] = None
        self.split_metadata: Dict[str, Any] = {}
        self.artifacts_writer: Optional[ArtifactsWriter] = None
        self._preds: Dict[str, list] = {"val": [], "test": []}

        # Collect outputs for epoch-end metric computation (performance optimization)
        self.train_step_outputs: list = []
        self.val_step_outputs: list = []

    # ------------------------------------------------------------------ #
    def set_artifact_dir(self, artifact_dir: Path | str) -> None:
        self.artifacts_dir = Path(artifact_dir)
        log.info(">> Artifact directory set to %s", self.artifacts_dir)

    def set_results_dir(self, results_dir: Path | str) -> None:
        self.results_dir = Path(results_dir)
        log.info(">> Results directory set to %s", self.results_dir)

    def set_split_metadata(self, split_metadata: Dict[str, Any]) -> None:
        self.split_metadata = split_metadata or {}
        if self.artifacts_writer:
            self.artifacts_writer.update_split_metadata(self.split_metadata)

    # ------------------------------------------------------------------ #
    def _should_enable_umodule(self) -> bool:
        if self.modules_cfg and hasattr(self.modules_cfg, "use_umodule"):
            return bool(getattr(self.modules_cfg, "use_umodule"))
        if self.umodule_cfg and hasattr(self.umodule_cfg, "enabled"):
            return bool(getattr(self.umodule_cfg, "enabled"))
        return False

    def _resolve_consistency_thresh(self) -> float:
        if self.metrics_cfg and hasattr(self.metrics_cfg, "consistency_thresh"):
            return float(getattr(self.metrics_cfg, "consistency_thresh"))
        if self.c_module_cfg and hasattr(self.c_module_cfg, "thresh"):
            return float(getattr(self.c_module_cfg, "thresh"))
        return 0.6

    def _gather_metadata_sources(self) -> List[str]:
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
        path = Path(path_str)
        candidates = [str(path)]
        cached = path.with_name(f"{path.stem}_cached{path.suffix}")
        if cached != path:
            candidates.append(str(cached))
        return candidates

    # ------------------------------------------------------------------ #
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self._pos_weight is None:
            return self.loss_fn(logits, labels)
        pos_weight_tensor = torch.tensor(
            [self._pos_weight], dtype=logits.dtype, device=logits.device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        return loss_fn(logits, labels)

    def _encode_modalities(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        z_url = self.url_encoder(batch["url"])
        z_html = self.html_encoder(
            input_ids=batch["html"]["input_ids"],
            attention_mask=batch["html"]["attention_mask"],
        )
        z_visual = self.visual_encoder(batch["visual"])
        return {"url": z_url, "html": z_html, "visual": z_visual}

    def _compute_logits(
        self,
        batch: Dict[str, Any],
        enable_mc_dropout: bool = False,
        dropout_override: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        with self._mc_dropout_context(enable_mc_dropout, dropout_override):
            embeddings = self._encode_modalities(batch)
            return {
                "url": self.url_head(embeddings["url"]),
                "html": self.html_head(embeddings["html"]),
                "visual": self.visual_head(embeddings["visual"]),
            }

    # ------------------------------------------------------------------ #
    def _run_c_module(
        self,
        batch: Dict[str, Any],
        stage: str,
        device: torch.device,
        batch_size: int,
    ) -> Optional[Dict[str, Any]]:
        if not (self.cmodule_enabled and self.c_module and stage in ("val", "test")):
            return None
        # Gate C-Module by brand presence: only use C for samples with brand_present == 1
        brand_present = batch.get("brand_present")
        if isinstance(brand_present, torch.Tensor):
            bp_mask = (brand_present.to(device) == 1)
            if not bool(bp_mask.any()):
                # No samples with brand evidence; skip C-Module
                return None
        else:
            bp_mask = None
        url_tensor = batch.get("url")
        if not isinstance(url_tensor, torch.Tensor):
            return None
        sample_ids = self._batch_to_list(batch.get("id"))
        urls = self._decode_url_tokens(url_tensor)
        # Extract image paths from batch for C-Module
        image_paths = self._batch_to_list(batch.get("image_path"))

        # DEBUG: Log image path extraction
        if stage == "test":
            non_none_paths = [p for p in image_paths if p is not None]
            log.info(
                f">> IMAGE PATH DEBUG: Extracted {len(non_none_paths)}/{len(image_paths)} non-None paths"
            )
            if non_none_paths:
                log.info(
                    f"   Sample path: {non_none_paths[0][:80] if non_none_paths[0] else 'None'}..."
                )

        if len(sample_ids) < batch_size:
            sample_ids.extend([None] * (batch_size - len(sample_ids)))
        if len(image_paths) < batch_size:
            image_paths.extend([None] * (batch_size - len(image_paths)))

        results: List[Dict[str, Any]] = []
        c_scores: List[float] = []
        c_url_scores: List[float] = []
        c_html_scores: List[float] = []
        c_visual_scores: List[float] = []
        brand_url: List[str] = []
        brand_html: List[str] = []
        brand_vis: List[str] = []

        for idx in range(batch_size):
            payload = {
                "sample_id": sample_ids[idx],
                "id": sample_ids[idx],
                "url_text": urls[idx] if idx < len(urls) else "",
                "image_path": (
                    image_paths[idx] if idx < len(image_paths) else None
                ),  # Added for visual OCR
            }
            # Skip C-Module when brand_present == 0 for this sample
            if bp_mask is not None and not bool(bp_mask[idx].item()):
                result = {}
            else:
                result = self.c_module.score_consistency(payload)
            results.append(result)
            value = result.get("c_mean", math.nan)
            try:
                score = float(value)
            except (TypeError, ValueError):
                score = math.nan
            c_scores.append(score)

            # Extract per-modality consistency scores
            try:
                c_url_scores.append(float(result.get("c_url", math.nan)))
            except (TypeError, ValueError):
                c_url_scores.append(math.nan)
            try:
                c_html_scores.append(float(result.get("c_html", math.nan)))
            except (TypeError, ValueError):
                c_html_scores.append(math.nan)
            try:
                c_visual_scores.append(float(result.get("c_visual", math.nan)))
            except (TypeError, ValueError):
                c_visual_scores.append(math.nan)

            brands = result.get("meta", {}).get("brands", {})
            brand_url.append(brands.get("url") or "")
            brand_html.append(brands.get("html") or "")
            brand_vis.append(brands.get("visual") or "")

        valid_scores = [score for score in c_scores if not math.isnan(score)]
        self._consistency_scores.setdefault(stage, []).extend(valid_scores)
        tensor = torch.tensor(c_scores, dtype=torch.float32, device=device)
        c_url_tensor = torch.tensor(c_url_scores, dtype=torch.float32, device=device)
        c_html_tensor = torch.tensor(c_html_scores, dtype=torch.float32, device=device)
        c_visual_tensor = torch.tensor(
            c_visual_scores, dtype=torch.float32, device=device
        )

        # Enhanced debug for C-Module
        if stage == "test":
            log.info(">> C-MODULE DEBUG:")
            log.info(
                f"   - brand_url: {len([b for b in brand_url if b])/len(brand_url):.1%} non-empty"
            )
            log.info(
                f"   - brand_html: {len([b for b in brand_html if b])/len(brand_html):.1%} non-empty"
            )
            log.info(
                f"   - brand_vis: {len([b for b in brand_vis if b])/len(brand_vis):.1%} non-empty"
            )
            log.info(
                f"   - c_url: min={c_url_tensor.min():.3f}, max={c_url_tensor.max():.3f}, mean={c_url_tensor.mean():.3f}"
            )
            log.info(
                f"   - c_html: min={c_html_tensor.min():.3f}, max={c_html_tensor.max():.3f}, mean={c_html_tensor.mean():.3f}"
            )
            log.info(
                f"   - c_visual: min={c_visual_tensor.min():.3f}, max={c_visual_tensor.max():.3f}, mean={c_visual_tensor.mean():.3f}"
            )
            log.info(
                f"   - c_visual has NaN: {torch.isnan(c_visual_tensor).any().item()}"
            )

        return {
            "c_mean": tensor,
            "c_url": c_url_tensor,
            "c_html": c_html_tensor,
            "c_visual": c_visual_tensor,
            "brand_url": brand_url,
            "brand_html": brand_html,
            "brand_vis": brand_vis,
        }

    @staticmethod
    def _decode_url_tokens(url_tensor: torch.Tensor) -> List[str]:
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
    def _batch_to_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _log_consistency_metrics(self, stage: str) -> None:
        if stage not in self._consistency_scores:
            return
        if not (self.cmodule_enabled and self.c_module):
            self._consistency_scores[stage].clear()
            return
        if not self._consistency_scores[stage]:
            return
        scores = torch.tensor(self._consistency_scores[stage], dtype=torch.float32)
        acs = float(scores.mean().item())
        mismatch = float((scores < self.c_module_threshold).float().mean().item())
        self.log(
            f"{stage}/consistency/acs",
            acs,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}/consistency/mr@{self.c_module_threshold:.2f}",
            mismatch,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self._consistency_scores[stage].clear()

    def _is_fixed_fusion_active(self, stage: str) -> bool:
        return (
            self.fusion_mode == "fixed"
            and stage in ("val", "test")
            and self.umodule_enabled
            and self.cmodule_enabled
        )

    def _record_fusion_stats(
        self,
        stage: str,
        alpha_dict: Dict[str, torch.Tensor],
        score_dict: Dict[str, torch.Tensor],
        fused_prob: torch.Tensor,
    ) -> None:
        if stage not in self._alpha_history:
            return
        for mod in self.modalities:
            alpha_tensor = alpha_dict.get(mod)
            score_tensor = score_dict.get(mod)
            if alpha_tensor is not None:
                self._alpha_history[stage][mod].append(alpha_tensor.detach().cpu())
            if score_tensor is not None:
                self._u_history[stage][mod].append(score_tensor.detach().cpu())
        self._fusion_probs.setdefault(stage, []).append(fused_prob.detach().cpu())

    def _log_fusion_metrics(
        self, stage: str, alpha_dict: Dict[str, torch.Tensor]
    ) -> None:
        if not self._is_fixed_fusion_active(stage):
            return
        for mod, tensor in alpha_dict.items():
            if tensor.numel() == 0:
                continue
            self.log(
                f"{stage}/fusion/alpha_{mod}",
                tensor.mean(),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def _build_fusion_record(self, fusion_block: Dict[str, Any]) -> Dict[str, Any]:
        record: Dict[str, torch.Tensor | List[str]] = {}
        for mod in self.modalities:
            record[f"U_{mod}"] = fusion_block["scores"][mod].detach().cpu().view(-1)
            record[f"alpha_{mod}"] = fusion_block["alpha"][mod].detach().cpu().view(-1)
        return record

    def _apply_fixed_fusion(
        self,
        stage: str,
        probs_dict: Dict[str, torch.Tensor],
        reliability_block: Dict[str, Dict[str, torch.Tensor]],
        cmodule_block: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self._is_fixed_fusion_active(stage):
            return None

        # Relaxed check: allow partial availability (need at least 2 modalities)
        if not reliability_block and not cmodule_block:
            if stage == "test":
                log.debug(
                    "Fixed fusion: both reliability_block and cmodule_block are empty"
                )
            return None

        # Collect available modalities with their r and c values
        available_modalities = []
        r_tensors: List[torch.Tensor] = []
        c_primes: List[torch.Tensor] = []
        missing_modalities = []
        fallback_reasons = []

        for mod in self.modalities:
            info = reliability_block.get(mod) if reliability_block else None
            c_tensor = cmodule_block.get(f"c_{mod}") if cmodule_block else None

            # Check if both r and c are available for this modality
            has_r = info is not None and "reliability" in info
            has_c = c_tensor is not None

            if not has_r or not has_c:
                missing_modalities.append(mod)
                if not has_r:
                    fallback_reasons.append(f"{mod}_no_reliability")
                if not has_c:
                    fallback_reasons.append(f"{mod}_no_consistency")
                continue

            r_tensor = info["reliability"]

            # Normalize c from [-1,1] to [0,1]
            if c_tensor.dim() == 1:
                c_tensor = c_tensor.unsqueeze(1)
            c_tensor = c_tensor.to(r_tensor.device)
            c_prime = torch.clamp(((c_tensor + 1.0) * 0.5), 0.0, 1.0)
            if r_tensor.shape != c_prime.shape:
                c_prime = c_prime.expand_as(r_tensor)

            # Check for NaN/Inf
            r_nan_count = torch.isnan(r_tensor).sum().item()
            c_nan_count = torch.isnan(c_prime).sum().item()

            if r_nan_count > 0 or c_nan_count > 0:
                if stage == "test":
                    log.debug(f"{mod}: r_nan={r_nan_count}, c_nan={c_nan_count}")
                missing_modalities.append(mod)
                fallback_reasons.append(f"{mod}_has_nan")
                continue

            available_modalities.append(mod)
            r_tensors.append(r_tensor)
            c_primes.append(c_prime)

        # Need at least 1 modality (allow single modality for now, but log warning)
        if len(available_modalities) == 0:
            if stage == "test":
                log.error(
                    f"❌ Fixed fusion FAILED: NO modalities available! Reasons: {fallback_reasons}"
                )
            return None
        elif len(available_modalities) == 1:
            if stage == "test":
                log.warning(
                    f"⚠ Fixed fusion with SINGLE modality: {available_modalities[0]}"
                )
                log.warning(
                    f"   Missing: {missing_modalities}, reasons: {fallback_reasons}"
                )

        if stage == "test":
            log.debug(
                f"Fixed fusion: using {len(available_modalities)}/3 modalities: {available_modalities}"
            )
            if missing_modalities:
                log.debug(
                    f"  Missing: {missing_modalities}, reasons: {fallback_reasons}"
                )

        # Compute U values for available modalities
        u_values: List[torch.Tensor] = []
        valid_mask: Optional[torch.Tensor] = None

        for r_tensor, c_prime in zip(r_tensors, c_primes):
            u_values.append(r_tensor + self.lambda_c * c_prime)
            mask = torch.isfinite(r_tensor) & torch.isfinite(c_prime)
            valid_mask = mask if valid_mask is None else (valid_mask & mask)

        if valid_mask is None:
            valid_mask = torch.ones_like(r_tensors[0], dtype=torch.bool)

        valid_count = valid_mask.sum().item()
        if stage == "test":
            log.debug(
                f"Fixed fusion: valid samples = {valid_count}/{valid_mask.numel()}"
            )

        # Compute softmax over available modalities
        u_stack = torch.stack(u_values, dim=0)
        u_stack = torch.nan_to_num(u_stack, nan=0.0, posinf=0.0, neginf=0.0)
        alpha_stack = torch.softmax(u_stack, dim=0)

        # For invalid samples, use uniform weights
        uniform_weight = 1.0 / len(available_modalities)
        uniform = torch.full_like(r_tensors[0], uniform_weight)
        for idx in range(len(available_modalities)):
            alpha_stack[idx] = torch.where(valid_mask, alpha_stack[idx], uniform)

        # Build fused probability and alpha dict
        fused_prob = torch.zeros_like(next(iter(probs_dict.values())))
        alpha_dict: Dict[str, torch.Tensor] = {}
        score_dict: Dict[str, torch.Tensor] = {}

        # Add available modalities with computed alphas
        for idx, mod in enumerate(available_modalities):
            alpha = alpha_stack[idx]
            alpha_dict[mod] = alpha
            score_dict[mod] = u_values[idx]
            fused_prob = fused_prob + alpha * probs_dict[mod]

        # Add missing modalities with zero alpha (for tracking)
        for mod in missing_modalities:
            alpha_dict[mod] = torch.zeros_like(fused_prob)
            score_dict[mod] = torch.zeros_like(fused_prob)

        return {
            "prob": fused_prob,
            "alpha": alpha_dict,
            "scores": score_dict,
            "fallback_info": {
                "available_modalities": available_modalities,
                "missing_modalities": missing_modalities,
                "fallback_reasons": fallback_reasons,
            },
        }

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits_dict = self._compute_logits(batch)
        probs = torch.stack(
            [torch.sigmoid(logits_dict[key]) for key in ("url", "html", "visual")],
            dim=0,
        )
        avg_prob = probs.mean(dim=0)
        avg_logit = torch.logit(avg_prob.clamp(min=1e-6, max=1 - 1e-6))
        return avg_logit

    def _shared_step(
        self, batch: Dict[str, Any], stage: str
    ) -> Dict[str, torch.Tensor]:
        logits_dict = self._compute_logits(batch)
        device = next(iter(logits_dict.values())).device
        labels = batch["label"].float().unsqueeze(1).to(device)
        labels_int = labels.view(-1).long()

        losses = [self._compute_loss(logits, labels) for logits in logits_dict.values()]
        loss = torch.stack(losses).mean()

        probs_dict = {mod: torch.sigmoid(logits) for mod, logits in logits_dict.items()}
        probs_stack = torch.stack([probs_dict[mod] for mod in self.modalities], dim=0)
        probs = probs_stack.mean(dim=0)
        preds = (probs > 0.5).long().view(-1)
        avg_logit = torch.logit(probs.clamp(min=1e-6, max=1 - 1e-6))

        var_probs = self._um_mc_dropout_predict(batch, stage)

        self._maybe_cache_umodule_inputs(
            stage, labels_int, logits_dict, probs_dict, probs, var_probs
        )
        reliability_block: Dict[str, Dict[str, torch.Tensor]] = {}
        if stage in ("val", "test"):
            reliability_block = self._um_collect_reliability(
                stage, logits_dict, probs_dict, var_probs
            )

        extras: Dict[str, torch.Tensor] = {}
        if stage == "test" and self.umodule_enabled:
            extras = self._um_compute_test_reliability(
                reliability_block, labels_int.shape[0], labels.device
            )

        cmodule_block = self._run_c_module(
            batch,
            stage=stage,
            device=labels.device,
            batch_size=labels_int.shape[0],
        )

        fusion_block = self._apply_fixed_fusion(
            stage, probs_dict, reliability_block, cmodule_block
        )
        if fusion_block:
            probs = fusion_block["prob"]
            avg_logit = torch.logit(probs.clamp(min=1e-6, max=1 - 1e-6))
            self._record_fusion_stats(
                stage, fusion_block["alpha"], fusion_block["scores"], probs
            )
            self._log_fusion_metrics(stage, fusion_block["alpha"])
            preds = (probs > 0.5).long().view(-1)

        # For test stage, compute metrics immediately (needed for final evaluation)
        if stage == "test":
            metrics_map = {
                "test": (self.test_acc, self.test_auroc, self.test_f1),
            }
            acc_metric, auroc_metric, f1_metric = metrics_map[stage]
            acc_metric(preds, labels_int)
            auroc_metric(probs.view(-1), labels_int)
            f1_metric(preds, labels_int)

            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/acc": acc_metric,
                    f"{stage}/auroc": auroc_metric,
                    f"{stage}/f1": f1_metric,
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        # Log loss immediately (needed for training)
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=stage != "train",
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=False,
        )

        cache = self._preds.get(stage)
        if cache is not None:
            record = {
                "id": batch.get("id"),
                "y_true": labels_int.detach().cpu(),
                "logit": avg_logit.detach().cpu(),
                "prob": probs.detach().cpu(),
            }
            if extras:
                record["extras"] = extras
            if cmodule_block:
                record["cmodule"] = {
                    "c_mean": cmodule_block["c_mean"].detach().cpu(),
                    "c_url": cmodule_block["c_url"].detach().cpu(),
                    "c_html": cmodule_block["c_html"].detach().cpu(),
                    "c_visual": cmodule_block["c_visual"].detach().cpu(),
                    "brand_url": cmodule_block["brand_url"],
                    "brand_html": cmodule_block["brand_html"],
                    "brand_vis": cmodule_block["brand_vis"],
                }
            if fusion_block:
                record["fusion"] = self._build_fusion_record(fusion_block)
                # Log fallback info if available (but don't add to CSV)
                if stage == "test" and "fallback_info" in fusion_block:
                    fallback_info = fusion_block["fallback_info"]
                    if fallback_info["missing_modalities"]:
                        log.debug(
                            f"Batch fallback: missing={fallback_info['missing_modalities']}, "
                            f"reasons={fallback_info['fallback_reasons']}"
                        )
            elif self._is_fixed_fusion_active(stage):
                # Fixed fusion was supposed to activate but didn't - log why
                if stage == "test":
                    log.warning(
                        f"Fixed fusion returned None: "
                        f"has_reliability={bool(reliability_block)}, "
                        f"has_cmodule={bool(cmodule_block)}"
                    )
            cache.append(record)

        return {
            "loss": loss,
            "probs": probs.detach().cpu(),
            "labels": labels_int.detach().cpu(),
            "preds": preds.detach().cpu(),
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        output = self._shared_step(batch, "train")
        # Collect outputs for epoch-end metric computation
        self.train_step_outputs.append(output)
        return output["loss"]

    def on_train_epoch_end(self) -> None:
        """Compute training metrics at epoch end for performance optimization."""
        if not self.train_step_outputs:
            return

        # Concatenate all outputs
        all_probs = torch.cat(
            [out["probs"].view(-1) for out in self.train_step_outputs]
        )
        all_labels = torch.cat([out["labels"] for out in self.train_step_outputs])
        all_preds = torch.cat([out["preds"] for out in self.train_step_outputs])

        # Compute metrics
        self.train_acc(all_preds, all_labels)
        self.train_auroc(all_probs, all_labels)
        self.train_f1(all_preds, all_labels)

        # Log metrics
        self.log_dict(
            {
                "train/acc": self.train_acc,
                "train/auroc": self.train_auroc,
                "train/f1": self.train_f1,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        # Clear outputs
        self.train_step_outputs.clear()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        output = self._shared_step(batch, "val")
        # Collect outputs for epoch-end metric computation
        self.val_step_outputs.append(output)
        return output

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        output = self._shared_step(batch, "test")
        return {
            "loss": output["loss"],
            "y_true": output["labels"].detach().cpu(),
            "y_prob": output["probs"].detach().cpu(),
        }

    def on_test_start(self) -> None:
        super().on_test_start()

        # Enhanced Debug: Check dropout layers status with detailed breakdown
        if self.umodule_enabled:
            if not self._dropout_layers:
                self._cache_dropout_layers()

            if self._dropout_layers:
                log.info(
                    f">> Test start: {len(self._dropout_layers)} dropout layers detected"
                )

                # Categorize dropout layers by modality
                dropout_by_modality = {"url": 0, "html": 0, "visual": 0, "other": 0}
                for name, module in self.named_modules():
                    if isinstance(module, _DropoutNd):
                        if "url" in name.lower():
                            dropout_by_modality["url"] += 1
                        elif "html" in name.lower():
                            dropout_by_modality["html"] += 1
                        elif "visual" in name.lower():
                            dropout_by_modality["visual"] += 1
                        else:
                            dropout_by_modality["other"] += 1

                log.info(f"   Dropout layers by modality: {dropout_by_modality}")

                if dropout_by_modality["visual"] == 0:
                    log.warning(
                        "   ⚠️  WARNING: No dropout layers found in visual branch!"
                    )
                    log.warning(
                        "   This will cause MC Dropout to fail for visual modality"
                    )

        if self.umodule_enabled and self.u_module and not self.u_module.tau_cache:
            self._load_calibration_from_disk()

        # Debug: Fixed fusion configuration
        if self._is_fixed_fusion_active("test"):
            log.info(
                f">> Fixed fusion ACTIVE for test: lambda_c={self.lambda_c}, "
                f"umodule_enabled={self.umodule_enabled}, cmodule_enabled={self.cmodule_enabled}"
            )

    # ------------------------------------------------------------------ #
    def _get_artifacts_writer(self) -> ArtifactsWriter:
        if self.artifacts_writer is None:
            if self.artifacts_dir is None:
                log.warning("Artifact directory not set; defaulting to ./artifacts")
                self.artifacts_dir = Path("./artifacts")
                self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            self.artifacts_writer = ArtifactsWriter(
                module=self,
                output_dir=self.artifacts_dir,
                split_metadata=self.split_metadata,
            )
        return self.artifacts_writer

    def on_validation_epoch_end(self) -> None:
        """Compute validation metrics at epoch end and save artifacts."""
        if self.val_step_outputs:
            # Concatenate all outputs
            all_probs = torch.cat(
                [out["probs"].view(-1) for out in self.val_step_outputs]
            )
            all_labels = torch.cat([out["labels"] for out in self.val_step_outputs])
            all_preds = torch.cat([out["preds"] for out in self.val_step_outputs])

            # Compute metrics
            self.val_acc(all_preds, all_labels)
            self.val_auroc(all_probs, all_labels)
            self.val_f1(all_preds, all_labels)

            # Log metrics
            self.log_dict(
                {
                    "val/acc": self.val_acc,
                    "val/auroc": self.val_auroc,
                    "val/f1": self.val_f1,
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

            # Clear outputs
            self.val_step_outputs.clear()

        # Save artifacts
        if self._preds["val"]:
            writer = self._get_artifacts_writer()
            writer.save_stage_artifacts(self._preds["val"], stage="val")
            self._preds["val"].clear()

        if self.umodule_enabled and self.u_module:
            self._um_fit_temperature_on_val()
        self._log_consistency_metrics("val")

    def on_test_epoch_end(self) -> None:
        if self._preds["test"]:
            writer = self._get_artifacts_writer()
            writer.save_stage_artifacts(self._preds["test"], stage="test")
            self._preds["test"].clear()
        self._um_log_reliability_metrics()
        if self.umodule_enabled and self.u_module:
            if self._fused_post_probs["test"]:
                self._um_plot_reliability()
            else:
                self._clear_stage_cache("test")
        self._log_consistency_metrics("test")

    # ------------------------------------------------------------------ #
    # MC Dropout helpers                                                 #
    # ------------------------------------------------------------------ #
    def _cache_dropout_layers(self) -> None:
        self._dropout_layers = [
            module for module in self.modules() if isinstance(module, _DropoutNd)
        ]
        log.info(f">> Cached {len(self._dropout_layers)} dropout layers for MC Dropout")

    @contextmanager
    def _mc_dropout_context(
        self, enable: bool, dropout_override: Optional[float] = None
    ):
        if not enable:
            yield
            return
        if not self._dropout_layers:
            self._cache_dropout_layers()
        states = []
        for layer in self._dropout_layers:
            states.append((layer, layer.training, getattr(layer, "p", None)))
        try:
            for layer, _, prev_p in states:
                layer.train()
                if dropout_override is not None and prev_p is not None:
                    layer.p = dropout_override
            yield
        finally:
            for layer, prev_training, prev_p in states:
                layer.train(prev_training)
                if dropout_override is not None and prev_p is not None:
                    layer.p = prev_p

    def _um_mc_dropout_predict(
        self, batch: Dict[str, Any], stage: str
    ) -> Dict[str, torch.Tensor]:
        if not (self.umodule_enabled and self.u_module and stage in ("val", "test")):
            return {}

        def _batched_logits_fn(
            data: Dict[str, Any],
            enable_mc_dropout: bool = False,
            dropout_p: Optional[float] = None,
        ) -> Dict[str, torch.Tensor]:
            return self._compute_logits(
                data,
                enable_mc_dropout=enable_mc_dropout,
                dropout_override=dropout_p,
            )

        # CRITICAL DEBUG: Check what logits are produced
        if stage == "test":
            test_logits = _batched_logits_fn(
                batch, enable_mc_dropout=False, dropout_p=None
            )
            log.info(">> MC DROPOUT PRE-CHECK:")
            log.info(f"   Test logits keys: {list(test_logits.keys())}")
            for mod, logit_tensor in test_logits.items():
                log.info(
                    f"   - {mod}: shape={logit_tensor.shape}, has_nan={torch.isnan(logit_tensor).any().item()}"
                )

        with torch.no_grad():
            _, _, var_probs = mc_dropout_predict(
                logits_fn=_batched_logits_fn,
                inputs=batch,
                mc_iters=self.u_module.mc_iters,
                dropout_p=self.u_module.dropout_p,
            )

        # Enhanced debug logging
        if stage == "test":
            log.info(">> MC DROPOUT RESULTS:")
            log.info(
                f"   var_probs keys: {list(var_probs.keys()) if var_probs else 'EMPTY'}"
            )
            if var_probs:
                for mod in ["url", "html", "visual"]:
                    if mod in var_probs:
                        var_tensor = var_probs[mod]
                        log.info(
                            f"   ✓ {mod}: shape={var_tensor.shape}, "
                            f"var_range=[{var_tensor.min():.6f}, {var_tensor.max():.6f}], "
                            f"mean_var={var_tensor.mean():.6f}"
                        )
                    else:
                        log.warning(f"   ✗ {mod}: MISSING from var_probs!")

        return var_probs

    def _um_collect_reliability(
        self,
        stage: str,
        logits_dict: Dict[str, torch.Tensor],
        probs_dict: Dict[str, torch.Tensor],
        var_probs: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        if not (
            self.umodule_enabled
            and self.u_module
            and stage in ("val", "test")
            and var_probs
        ):
            if stage == "test":
                log.debug(
                    f"Reliability collection skipped: umodule_enabled={self.umodule_enabled}, "
                    f"u_module={self.u_module is not None}, stage={stage}, var_probs={bool(var_probs)}"
                )
            return {}

        block: Dict[str, Dict[str, torch.Tensor]] = {}
        for mod in self.modalities:
            var_tensor = var_probs.get(mod)
            if var_tensor is None:
                if stage == "test":
                    log.warning(
                        f"⚠ {mod.upper()} modality: var_tensor is None (MC Dropout failed)"
                    )
                    # WORKAROUND: Use default low variance for visual to enable fusion
                    if mod == "visual" and mod in probs_dict:
                        log.warning(
                            "   Using default variance for visual modality (workaround)"
                        )
                        var_tensor = torch.full_like(
                            probs_dict[mod], 0.01
                        )  # Low variance = high reliability
                    else:
                        continue
                else:
                    continue

            tau = self.u_module.tau_cache.get(mod)
            if tau is None:
                tau = getattr(self.u_module, "init_temperature", 1.0)

            # Enhanced debug for visual modality
            if stage == "test" and mod == "visual":
                log.info(">> VISUAL MODALITY DEBUG:")
                log.info(f"   - var_tensor shape: {var_tensor.shape}")
                log.info(
                    f"   - var_tensor stats: min={var_tensor.min():.6f}, max={var_tensor.max():.6f}, mean={var_tensor.mean():.6f}"
                )
                log.info(f"   - tau value: {tau}")
                log.info(f"   - has NaN: {torch.isnan(var_tensor).any().item()}")

            probs_ts, reliability = self.u_module.estimate_reliability(
                logits=logits_dict[mod].detach(),
                probs=probs_dict[mod].detach(),
                var_prob=var_tensor.to(logits_dict[mod].device),
                tau=tau,
            )
            probs_ts = probs_ts.detach()
            reliability = reliability.detach()

            # Enhanced debug for visual reliability
            if stage == "test" and mod == "visual":
                log.info(f"   - reliability shape: {reliability.shape}")
                log.info(
                    f"   - reliability stats: min={reliability.min():.6f}, max={reliability.max():.6f}, mean={reliability.mean():.6f}"
                )
                log.info(
                    f"   - reliability has NaN: {torch.isnan(reliability).any().item()}"
                )

            block[mod] = {"probs_ts": probs_ts, "reliability": reliability, "tau": tau}
            if stage in self._modal_outputs:
                self._modal_outputs[stage][mod]["probs_ts"].append(
                    probs_ts.detach().cpu()
                )
                self._modal_outputs[stage][mod]["reliability"].append(
                    reliability.detach().cpu()
                )

        if stage == "test":
            log.info(f">> Reliability block collected for: {list(block.keys())}")
            if "visual" not in block:
                log.warning("⚠ VISUAL modality MISSING from reliability block!")

        return block

    def _maybe_cache_umodule_inputs(
        self,
        stage: str,
        labels: torch.Tensor,
        logits_dict: Dict[str, torch.Tensor],
        probs_dict: Dict[str, torch.Tensor],
        fused_probs: torch.Tensor,
        var_probs: Dict[str, torch.Tensor],
    ) -> None:
        if not self.umodule_enabled:
            return
        if stage not in self._stage_labels:
            return
        self._stage_labels[stage].append(labels.detach().cpu())
        self._fused_probs[stage].append(fused_probs.detach().cpu())
        for mod in self.modalities:
            self._modal_outputs[stage][mod]["logits"].append(
                logits_dict[mod].detach().cpu()
            )
            self._modal_outputs[stage][mod]["probs"].append(
                probs_dict[mod].detach().cpu()
            )
            if var_probs:
                var_tensor = var_probs.get(mod)
                if var_tensor is not None:
                    self._modal_outputs[stage][mod]["var"].append(
                        var_tensor.detach().cpu()
                    )

    def _um_compute_test_reliability(
        self,
        reliability_block: Dict[str, Dict[str, torch.Tensor]],
        batch_size: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        extras: Dict[str, torch.Tensor] = {}
        fused_post_collection: list[torch.Tensor] = []
        for mod in self.modalities:
            info = reliability_block.get(mod) if reliability_block else None
            if not info:
                continue
            column_name = self._reliability_column_map.get(mod, f"r_{mod}")
            extras[column_name] = info["reliability"].detach().view(-1).cpu()
            fused_post_collection.append(info["probs_ts"].detach())

        if fused_post_collection:
            fused_post = torch.stack(fused_post_collection, dim=0).mean(dim=0)
            self._fused_post_probs["test"].append(fused_post.detach().cpu())

        for mod, column in self._reliability_column_map.items():
            if column not in extras:
                extras[column] = torch.full(
                    (batch_size,),
                    float("nan"),
                    device=device,
                    dtype=torch.float32,
                )
        return extras

    def _um_fit_temperature_on_val(self) -> None:
        if not self.umodule_enabled or not self.u_module:
            return
        label_chunks = self._stage_labels.get("val", [])
        if not label_chunks:
            return
        labels = torch.cat(label_chunks).view(-1).float()
        if labels.numel() == 0:
            return

        ece_bins = self._get_ece_bins()
        ts_per_mod: Dict[str, torch.Tensor] = {}

        for mod in self.modalities:
            logits_list = self._modal_outputs["val"][mod]["logits"]
            if not logits_list:
                continue
            logits = torch.cat(logits_list, dim=0)
            probs = torch.cat(self._modal_outputs["val"][mod]["probs"], dim=0)
            meta = self.u_module.fit_temperature_on_val(mod, logits, labels)

            scaled_logits = temperature_scaling(logits, meta["tau"])
            probs_ts = torch.sigmoid(scaled_logits)
            ts_per_mod[mod] = probs_ts.detach()
            self._modal_outputs["val"][mod]["probs_ts"].append(probs_ts.detach().cpu())

            ece_pre, stats_pre = compute_ece_metric(probs, labels, n_bins=ece_bins)
            ece_post, stats_post = compute_ece_metric(probs_ts, labels, n_bins=ece_bins)
            brier_post = brier_score(probs_ts, labels)
            nll_post = F.binary_cross_entropy_with_logits(
                scaled_logits,
                labels.view(-1, 1).to(scaled_logits.device),
            ).item()

            meta.update(
                {
                    "ece_pre": float(ece_pre),
                    "ece_post": float(ece_post),
                    "ece_bins_pre": int(stats_pre["bins_used"]),
                    "ece_bins_post": int(stats_post["bins_used"]),
                    "brier_post": float(brier_post),
                    "nll_post": float(nll_post),
                }
            )
            self._calibration_summary[mod] = meta

            self.log(
                f"val/umodule/ece_pre_{mod}",
                float(ece_pre),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"val/umodule/ece_post_{mod}",
                float(ece_post),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"val/umodule/brier_post_{mod}",
                float(brier_post),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        if ts_per_mod:
            fused_post = torch.stack(list(ts_per_mod.values()), dim=0).mean(dim=0)
            self._fused_post_probs["val"] = [fused_post.detach().cpu()]

        self._write_calibration_artifacts()
        self._clear_stage_cache("val")

    def _write_calibration_artifacts(self) -> None:
        if not self._calibration_summary:
            return
        writer = self._get_artifacts_writer()
        output_dir = writer.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        calib_path = output_dir / "calibration.json"
        payload = {"modalities": self._calibration_summary}
        for mod, meta in self._calibration_summary.items():
            payload[f"tau_{mod}"] = meta.get("tau")
            payload[f"{mod}_tau_source"] = meta.get("tau_source")
            payload[f"{mod}_n_val"] = meta.get("n_val")
            payload[f"{mod}_nll"] = meta.get("nll")
        with open(calib_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        log.info(">> Saved calibration params: %s", calib_path)

    def _load_calibration_from_disk(self) -> None:
        writer = self._get_artifacts_writer()
        calib_path = writer.output_dir / "calibration.json"
        if not calib_path.exists():
            log.warning(">> Calibration file missing: %s", calib_path)
            return
        try:
            with open(calib_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            log.warning(">> Failed to load calibration: %s", exc)
            return
        modalities = data.get("modalities", {})
        for mod in self.modalities:
            info = modalities.get(mod)
            if not info and f"tau_{mod}" in data:
                info = {
                    "tau": data.get(f"tau_{mod}"),
                    "tau_source": data.get(f"{mod}_tau_source"),
                    "n_val": data.get(f"{mod}_n_val"),
                    "nll": data.get(f"{mod}_nll"),
                }
            if info and info.get("tau") is not None and self.u_module:
                self.u_module.tau_cache[mod] = float(info["tau"])
                self._calibration_summary[mod] = info
        log.info(">> Loaded calibration params from %s", calib_path)

    def _clear_stage_cache(self, stage: str) -> None:
        if stage not in self._stage_labels:
            return
        self._stage_labels[stage].clear()
        self._fused_probs[stage].clear()
        self._fused_post_probs[stage].clear()
        for mod in self.modalities:
            for key in self._modal_outputs[stage][mod]:
                self._modal_outputs[stage][mod][key].clear()
        if stage in self._consistency_scores:
            self._consistency_scores[stage].clear()
        self._reset_fusion_tracking(stage)

    def _reset_fusion_tracking(self, stage: str) -> None:
        if stage in self._fusion_probs:
            self._fusion_probs[stage].clear()
        if stage in self._alpha_history:
            for mod in self.modalities:
                self._alpha_history[stage][mod].clear()
                self._u_history[stage][mod].clear()

    def _get_ece_bins(self) -> int:
        if self.metrics_cfg and hasattr(self.metrics_cfg, "ece_bins"):
            return int(getattr(self.metrics_cfg, "ece_bins"))
        return 15

    def _um_plot_reliability(self) -> None:
        if not (
            self.umodule_enabled
            and self._fused_probs["test"]
            and self._fused_post_probs["test"]
            and self._stage_labels["test"]
        ):
            return
        y_true = torch.cat(self._stage_labels["test"]).view(-1).cpu().numpy()
        pre_probs = torch.cat(self._fused_probs["test"], dim=0).view(-1).cpu().numpy()
        post_probs = (
            torch.cat(self._fused_post_probs["test"], dim=0).view(-1).cpu().numpy()
        )
        if pre_probs.size == 0 or post_probs.size == 0:
            return

        ece_bins = self._get_ece_bins()
        ece_pre, stats_pre = compute_ece_metric(pre_probs, y_true, n_bins=ece_bins)
        ece_post, stats_post = compute_ece_metric(post_probs, y_true, n_bins=ece_bins)

        writer = self._get_artifacts_writer()
        pre_path = writer.output_dir / "reliability_pre_test.png"
        post_path = writer.output_dir / "reliability_post_test.png"
        ResultVisualizer.save_calibration_curve(
            y_true=y_true,
            y_prob=pre_probs,
            path=pre_path,
            n_bins=int(stats_pre["bins_used"]),
            ece_value=float(ece_pre),
            warn_small_sample=bool(stats_pre.get("ece_reason")),
            title=f"Reliability (pre) - ECE={ece_pre:.4f} (bins={stats_pre['bins_used']})",
        )
        ResultVisualizer.save_calibration_curve(
            y_true=y_true,
            y_prob=post_probs,
            path=post_path,
            n_bins=int(stats_post["bins_used"]),
            ece_value=float(ece_post),
            warn_small_sample=bool(stats_post.get("ece_reason")),
            title=f"Reliability (post) - ECE={ece_post:.4f} (bins={stats_post['bins_used']})",
        )

        self.log(
            "test/umodule/ece_pre_fused",
            float(ece_pre),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "test/umodule/ece_post_fused",
            float(ece_post),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self._clear_stage_cache("test")

    def _um_log_reliability_metrics(self) -> None:
        if not (
            self.results_dir and self._stage_labels["test"] and self.umodule_enabled
        ):
            return

        labels = torch.cat(self._stage_labels["test"]).view(-1).float().cpu()
        labels_np = labels.numpy()
        ece_bins = self._get_ece_bins()

        def _concat(chunks: List[torch.Tensor]) -> Optional[torch.Tensor]:
            if not chunks:
                return None
            return torch.cat(chunks).view(-1).cpu()

        summary: Dict[str, Dict[str, float]] = {"modalities": {}, "fused": {}}
        for mod in self.modalities:
            pre_probs = _concat(self._modal_outputs["test"][mod]["probs"])
            post_probs = _concat(self._modal_outputs["test"][mod]["probs_ts"])
            entry: Dict[str, float] = {}
            if pre_probs is not None:
                ece_pre, stats_pre = compute_ece_metric(
                    pre_probs.numpy(), labels_np, n_bins=ece_bins
                )
                entry["ece_pre"] = float(ece_pre)
                entry["ece_bins_pre"] = int(stats_pre["bins_used"])
            if post_probs is not None:
                ece_post, stats_post = compute_ece_metric(
                    post_probs.numpy(), labels_np, n_bins=ece_bins
                )
                entry["ece_post"] = float(ece_post)
                entry["ece_bins_post"] = int(stats_post["bins_used"])
                entry["brier_post"] = float(brier_score(post_probs.numpy(), labels_np))
            summary["modalities"][mod] = entry

        fused_pre = _concat(self._fused_probs["test"])
        fused_post = _concat(self._fused_post_probs["test"])
        fused_entry: Dict[str, float] = {}
        if fused_pre is not None:
            ece_pre, stats_pre = compute_ece_metric(
                fused_pre.numpy(), labels_np, n_bins=ece_bins
            )
            fused_entry["ece_pre"] = float(ece_pre)
            fused_entry["ece_bins_pre"] = int(stats_pre["bins_used"])
        if fused_post is not None:
            ece_post, stats_post = compute_ece_metric(
                fused_post.numpy(), labels_np, n_bins=ece_bins
            )
            fused_entry["ece_post"] = float(ece_post)
            fused_entry["ece_bins_post"] = int(stats_post["bins_used"])
            fused_entry["brier_post"] = float(
                brier_score(fused_post.numpy(), labels_np)
            )
        summary["fused"] = fused_entry

        s3_summary = self._build_s3_summary(labels_np)
        if s3_summary:
            summary["s3"] = s3_summary

        eval_path = self.results_dir / "eval_summary.json"
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        log.info(">> Saved eval summary to %s", eval_path)

    def _build_s3_summary(self, labels_np: np.ndarray) -> Dict[str, Any]:
        if not (self._is_fixed_fusion_active("test") and self._fusion_probs["test"]):
            return {}
        fusion_probs = torch.cat(self._fusion_probs["test"], dim=0).view(-1).cpu()
        if fusion_probs.numel() != labels_np.size:
            return {}
        probs_np = fusion_probs.numpy()
        try:
            auroc_val = float(roc_auc_score(labels_np, probs_np))
        except ValueError:
            auroc_val = float("nan")
        ece_value, stats = compute_ece_metric(
            probs_np, labels_np, n_bins=self._get_ece_bins()
        )
        brier_val = float(brier_score(probs_np, labels_np))
        summary = {
            "fusion_mode": self.fusion_mode,
            "lambda_c": float(self.lambda_c),
            "auroc": auroc_val,
            "ece": float(ece_value),
            "ece_bins_used": int(stats["bins_used"]),
            "brier": brier_val,
            "alpha_stats": self._summarize_alpha_history("test"),
        }
        synergy = self._compute_synergy_metric(auroc_val)
        if synergy:
            summary["synergy_metrics"] = synergy
        return summary

    def _summarize_alpha_history(self, stage: str) -> Dict[str, Dict[str, float]]:
        stats = {"mean_alpha": {}, "var_alpha": {}}
        history = self._alpha_history.get(stage)
        if not history:
            return stats
        for mod, tensors in history.items():
            if not tensors:
                continue
            combined = torch.cat(tensors, dim=0).view(-1)
            if combined.numel() == 0:
                continue
            stats["mean_alpha"][mod] = float(combined.mean().item())
            stats["var_alpha"][mod] = float(combined.var(unbiased=False).item())
        return stats

    def _compute_synergy_metric(self, s3_metric: float) -> Dict[str, Any]:
        baselines = self._load_synergy_baselines()
        if not baselines:
            return {}
        metric_name = "auroc"
        candidates = [
            (name, data.get(metric_name))
            for name, data in baselines.items()
            if isinstance(data, dict) and data.get(metric_name) is not None
        ]
        if not candidates:
            return {}
        best_label, best_value = max(candidates, key=lambda item: item[1])
        delta = s3_metric - float(best_value)
        return {
            "metric": metric_name,
            "s3": s3_metric,
            "best_baseline": {best_label: float(best_value)},
            "delta_vs_best": float(delta),
        }

    def _load_synergy_baselines(self) -> Dict[str, Any]:
        if not self.results_dir:
            return {}
        candidates = [
            self.results_dir / "synergy_baselines.json",
            self.results_dir.parent / "synergy_baselines.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                continue
            if isinstance(data, dict):
                return data.get("baselines") or data
        return {}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.umodule_enabled and self.u_module:
            checkpoint["u_module_tau_cache"] = dict(self.u_module.tau_cache)
            checkpoint["u_module_meta"] = self._calibration_summary

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if not (self.umodule_enabled and self.u_module):
            return
        tau_cache = checkpoint.get("u_module_tau_cache")
        if tau_cache:
            self.u_module.tau_cache.update(tau_cache)
        meta = checkpoint.get("u_module_meta")
        if meta:
            self._calibration_summary.update(meta)

    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        bert_params = [
            p for p in self.html_encoder.bert.parameters() if p.requires_grad
        ]
        non_bert_params = []
        non_bert_params += [p for p in self.url_encoder.parameters() if p.requires_grad]
        non_bert_params += [
            p for p in self.html_encoder.projection.parameters() if p.requires_grad
        ]
        non_bert_params += [
            p for p in self.visual_encoder.parameters() if p.requires_grad
        ]
        non_bert_params += [
            p
            for p in list(self.url_head.parameters())
            + list(self.html_head.parameters())
            + list(self.visual_head.parameters())
            if p.requires_grad
        ]

        param_groups = []
        if bert_params:
            param_groups.append({"params": bert_params, "lr": 2e-5})
        if non_bert_params:
            param_groups.append(
                {"params": non_bert_params, "lr": self.hparams.learning_rate}
            )

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.hparams.weight_decay
        )

        # Use ReduceLROnPlateau scheduler if configured, otherwise use CosineAnnealingLR
        if (
            self.cfg
            and hasattr(self.cfg, "eval")
            and hasattr(self.cfg.eval, "reduce_lr_on_plateau")
        ):
            rlr_config = self.cfg.eval.reduce_lr_on_plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=rlr_config.get("mode", "min"),
                factor=rlr_config.get("factor", 0.5),
                patience=rlr_config.get("patience", 2),
                min_lr=rlr_config.get("min_lr", 1e-6),
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": rlr_config.get("monitor", "val/loss"),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            # Fallback to CosineAnnealingLR if not configured
            t_max = 20
            if self.cfg and hasattr(self.cfg, "train"):
                t_max = getattr(self.cfg.train, "epochs", t_max)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=t_max,
                eta_min=1e-6,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
