"""
Uncertainty Module utilities for MC Dropout & temperature scaling.
"""

from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from typing import Callable, Deque, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from src.utils.logging import get_logger


log = get_logger(__name__)

DropoutCallable = Callable[..., Dict[str, torch.Tensor] | torch.Tensor]
MCOutputs = Tuple[
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
    Dict[str, torch.Tensor],
]


@contextmanager
def _local_dropout_mode(module: nn.Module) -> Iterable[nn.Dropout]:
    """Temporarily enable dropout layers without touching the global eval/train mode."""

    dropout_layers: List[nn.Dropout] = [
        layer for layer in module.modules() if isinstance(layer, nn.Dropout)
    ]
    prev_states = {layer: layer.training for layer in dropout_layers}
    try:
        for layer in dropout_layers:
            layer.train()
        yield dropout_layers
    finally:
        for layer in dropout_layers:
            layer.train(prev_states[layer])


def _normalize_logits(
    output: Dict[str, torch.Tensor] | torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Force logits into dict form for downstream aggregation."""
    if isinstance(output, dict):
        return output
    if isinstance(output, torch.Tensor):
        return {"default": output}
    raise TypeError(f"Unsupported logits output type: {type(output)}")


def mc_dropout_predict(
    logits_fn: DropoutCallable,
    inputs: Dict[str, torch.Tensor],
    mc_iters: int = 10,
    dropout_p: float = 0.1,
) -> MCOutputs:
    """
    Run MC-dropout by repeatedly calling `logits_fn` under local dropout noise.

    Prefers calling `logits_fn(..., enable_mc_dropout=True, dropout_p=dropout_p)`.
    When that signature is not supported, all `nn.Dropout` layers within the bound
    module are toggled to `.train()` during sampling and restored afterwards.

    Returns:
        Tuple of (mean_logits, mean_probs, var_probs) dictionaries keyed by modality.
    """

    if mc_iters <= 0:
        raise ValueError("mc_iters must be > 0")

    bound_module = getattr(logits_fn, "__self__", None)
    logits_samples: Dict[str, List[torch.Tensor]] = defaultdict(list)

    try:
        first_logits = logits_fn(inputs, enable_mc_dropout=True, dropout_p=dropout_p)
        supports_flag = True
        logits_samples_dict = [_normalize_logits(first_logits)]
    except TypeError:
        supports_flag = False
        logits_samples_dict = []

    if supports_flag:
        for tensor_dict in logits_samples_dict:
            for key, tensor in tensor_dict.items():
                logits_samples[key].append(tensor.detach())
        for _ in range(len(logits_samples_dict), mc_iters):
            logits_dict = _normalize_logits(
                logits_fn(inputs, enable_mc_dropout=True, dropout_p=dropout_p)
            )
            for key, tensor in logits_dict.items():
                logits_samples[key].append(tensor.detach())
    else:
        ctx = (
            _local_dropout_mode(bound_module)
            if isinstance(bound_module, nn.Module)
            else nullcontext()
        )
        with ctx:
            for _ in range(mc_iters):
                logits_dict = _normalize_logits(logits_fn(inputs))
                for key, tensor in logits_dict.items():
                    logits_samples[key].append(tensor.detach())

    mean_logits: Dict[str, torch.Tensor] = {}
    mean_probs: Dict[str, torch.Tensor] = {}
    var_probs: Dict[str, torch.Tensor] = {}

    for key, tensors in logits_samples.items():
        stacked = torch.stack(tensors, dim=0)
        mean_logits[key] = stacked.mean(dim=0)
        prob_samples = torch.sigmoid(stacked)
        mean_probs[key] = prob_samples.mean(dim=0)
        var_probs[key] = prob_samples.var(dim=0, unbiased=False)

    return mean_logits, mean_probs, var_probs


def temperature_scaling(
    logits: torch.Tensor, temperature: float | torch.Tensor
) -> torch.Tensor:
    """Apply scalar temperature to logits."""

    if torch.is_tensor(temperature):
        tau = temperature.to(dtype=logits.dtype, device=logits.device)
    else:
        tau = torch.tensor(float(temperature), dtype=logits.dtype, device=logits.device)
    tau = torch.clamp(tau, min=1e-4)
    return logits / tau


class UModule:
    """
    Utility class that fits per-modality temperature scalars and derives reliability.
    """

    def __init__(
        self,
        mc_iters: int = 10,
        dropout_p: float = 0.1,
        init_temperature: float = 1.0,
        lambda_u: float = 1.0,
        learnable: bool = False,
        min_val_samples: int = 150,
    ) -> None:
        self.mc_iters = mc_iters
        self.dropout_p = dropout_p
        self.init_temperature = init_temperature
        self.lambda_u = lambda_u
        self.learnable = learnable
        self.min_val_samples = min_val_samples

        self._tau_cache: Dict[str, float] = {}
        self._fit_meta: Dict[str, Dict[str, float | str | int]] = {}
        self._history: Dict[str, Deque[Tuple[torch.Tensor, torch.Tensor]]] = (
            defaultdict(lambda: deque(maxlen=3))
        )

    @property
    def tau_cache(self) -> Dict[str, float]:
        return self._tau_cache

    @property
    def fit_meta(self) -> Dict[str, Dict[str, float | str | int]]:
        return self._fit_meta

    def _prepare_history(
        self,
        modality: str,
        logits_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        logits_cpu = logits_val.detach().view(-1, 1).cpu()
        labels_cpu = y_val.detach().view(-1, 1).cpu()

        history = self._history[modality]
        history.append((logits_cpu, labels_cpu))

        if logits_cpu.shape[0] < self.min_val_samples and len(history) > 1:
            merged_logits = torch.cat([item[0] for item in history], dim=0)
            merged_labels = torch.cat([item[1] for item in history], dim=0)
            return merged_logits, merged_labels, "val_merged"
        return logits_cpu, labels_cpu, "val"

    def _optimize_temperature(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, float]:
        device = logits.device
        dtype = logits.dtype
        log_tau = torch.tensor(
            float(torch.log(torch.tensor(self.init_temperature))),
            device=device,
            dtype=dtype,
        ).requires_grad_()

        # Try with strong_wolfe first, fall back to None if numerical issues occur
        try:
            optimizer = torch.optim.LBFGS(
                [log_tau], max_iter=50, line_search_fn="strong_wolfe"
            )
            loss_fn = nn.BCEWithLogitsLoss()

            labels = labels.view(-1, 1).to(device=device, dtype=dtype)
            logits = logits.view(-1, 1).to(device=device, dtype=dtype)

            def closure():
                optimizer.zero_grad()
                tau = torch.exp(log_tau)
                scaled = logits / torch.clamp(tau, min=1e-4)
                loss = loss_fn(scaled, labels)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
            tau_value = torch.exp(log_tau.detach()).item()
            return tau_value, float(loss.item())
        except (ZeroDivisionError, RuntimeError):
            # Fall back to LBFGS without line search on numerical issues
            log_tau = torch.tensor(
                float(torch.log(torch.tensor(self.init_temperature))),
                device=device,
                dtype=dtype,
            ).requires_grad_()
            optimizer = torch.optim.LBFGS([log_tau], max_iter=50, line_search_fn=None)
            loss_fn = nn.BCEWithLogitsLoss()

            labels = labels.view(-1, 1).to(device=device, dtype=dtype)
            logits = logits.view(-1, 1).to(device=device, dtype=dtype)

        def closure():
            optimizer.zero_grad()
            tau = torch.exp(log_tau)
            scaled = logits / torch.clamp(tau, min=1e-4)
            loss = loss_fn(scaled, labels)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        tau_value = torch.exp(log_tau.detach()).item()
        return tau_value, float(loss.item())

    def fit_temperature_on_val(
        self,
        modality: str,
        logits_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> Dict[str, float | str | int]:
        """
        Fit temperature on validation logits for a modality.
        Returns metadata with tau, data source, sample count, and fitted NLL.
        """
        merged_logits, merged_labels, source = self._prepare_history(
            modality, logits_val, y_val
        )
        tau, nll = self._optimize_temperature(merged_logits, merged_labels)
        self._tau_cache[modality] = tau
        meta = {
            "tau": tau,
            "tau_source": source,
            "n_val": int(merged_logits.shape[0]),
            "nll": nll,
        }
        self._fit_meta[modality] = meta
        return meta

    def estimate_reliability(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        var_prob: torch.Tensor,
        tau: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temperature scaling + uncertainty penalty to estimate modality reliability.

        Returns:
            Tuple of (temperature-scaled probability, reliability score r_m).
        """
        scaled_logits = temperature_scaling(logits, tau)
        probs_ts = torch.sigmoid(scaled_logits)
        reliability = torch.sigmoid(probs_ts - self.lambda_u * var_prob)
        return probs_ts, reliability
