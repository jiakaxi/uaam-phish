# Uncertainty Module - 实现细节

> **实现版本:** 1.0
> **状态:** 规划中
> **规格文档:** [uncertainty.md](../specs/uncertainty.md)
> **最后更新:** 2025-10-22

---

## 📂 文件结构

```
src/modules/
└── uncertainty/
    ├── __init__.py
    ├── base.py              # 基类定义
    ├── mc_dropout.py        # MC Dropout 实现
    ├── ensemble.py          # Deep Ensembles 实现
    ├── bayesian.py          # 贝叶斯NN实现
    ├── calibration.py       # 校准模块
    └── metrics.py           # 评估指标
```

---

## 🔨 实现步骤

### Phase 1: 基础框架 (Week 1)

#### 1.1 基类实现

**文件:** `src/modules/uncertainty/base.py`

```python
from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
import torch.nn as nn


class UncertaintyEstimator(ABC, nn.Module):
    """
    不确定性估计器抽象基类
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.calibrator = None

    @abstractmethod
    def estimate(
        self,
        x: torch.Tensor,
        model: nn.Module,
        return_samples: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        估计不确定性

        Args:
            x: 输入特征 [B, D]
            model: 预测模型
            return_samples: 是否返回所有采样

        Returns:
            {
                'mean': 平均预测 [B],
                'epistemic': 认知不确定性 [B],
                'aleatoric': 偶然不确定性 [B],
                'samples': 采样结果 [S, B] (optional)
            }
        """
        pass

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        method: str = "temperature_scaling"
    ):
        """
        校准模型

        Args:
            logits: 模型输出 [N]
            labels: 真实标签 [N]
            method: 校准方法
        """
        from .calibration import get_calibrator
        self.calibrator = get_calibrator(method)
        self.calibrator.fit(logits, labels)

    def forward(
        self,
        x: torch.Tensor,
        model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（包含校准）
        """
        results = self.estimate(x, model)

        # 应用校准
        if self.calibrator is not None:
            results['mean'] = self.calibrator(results['mean'])

        return results
```

#### 1.2 MC Dropout 实现

**文件:** `src/modules/uncertainty/mc_dropout.py`

```python
import torch
import torch.nn as nn
from typing import Dict
from .base import UncertaintyEstimator


class MCDropoutEstimator(UncertaintyEstimator):
    """
    Monte Carlo Dropout 不确定性估计

    使用方法：
    1. 训练时正常使用 dropout
    2. 推理时保持 dropout 激活
    3. 多次前向传播获得分布
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.num_samples = config.get("num_samples", 100)
        self.dropout_rate = config.get("dropout_rate", 0.1)

    def estimate(
        self,
        x: torch.Tensor,
        model: nn.Module,
        return_samples: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        MC Dropout 估计
        """
        # 确保模型在训练模式（激活 dropout）
        was_training = model.training
        model.train()

        # 禁用梯度计算
        with torch.no_grad():
            samples = []
            for _ in range(self.num_samples):
                logits = model(x)
                probs = torch.sigmoid(logits)
                samples.append(probs)

            samples = torch.stack(samples)  # [S, B]

        # 恢复模型状态
        model.train(was_training)

        # 计算统计量
        mean_pred = samples.mean(dim=0)  # [B]
        epistemic = samples.var(dim=0)   # [B]

        # 估计偶然不确定性（二分类的伯努利方差）
        aleatoric = mean_pred * (1 - mean_pred)

        # 总不确定性
        total = epistemic + aleatoric

        results = {
            'mean': mean_pred,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total,
        }

        if return_samples:
            results['samples'] = samples

        return results

    @staticmethod
    def enable_dropout(model: nn.Module):
        """
        启用模型中的所有 Dropout 层
        """
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
```

#### 1.3 Deep Ensembles 实现

**文件:** `src/modules/uncertainty/ensemble.py`

```python
import torch
import torch.nn as nn
from typing import Dict, List
from .base import UncertaintyEstimator


class EnsembleEstimator(UncertaintyEstimator):
    """
    Deep Ensembles 不确定性估计

    使用方法：
    1. 训练多个独立初始化的模型
    2. 推理时聚合所有模型的预测
    3. 使用预测方差作为不确定性
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.num_models = config.get("num_models", 5)
        self.aggregation = config.get("aggregation", "mean")
        self.models = nn.ModuleList()

    def add_model(self, model: nn.Module):
        """添加一个模型到集成中"""
        self.models.append(model)

    def estimate(
        self,
        x: torch.Tensor,
        model: nn.Module = None,  # 不使用，保持接口一致
        return_samples: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Ensemble 估计
        """
        if len(self.models) == 0:
            raise ValueError("没有可用的模型，请先使用 add_model() 添加")

        # 收集所有模型的预测
        with torch.no_grad():
            predictions = []
            for m in self.models:
                m.eval()
                logits = m(x)
                probs = torch.sigmoid(logits)
                predictions.append(probs)

            predictions = torch.stack(predictions)  # [M, B]

        # 聚合预测
        if self.aggregation == "mean":
            mean_pred = predictions.mean(dim=0)
        elif self.aggregation == "median":
            mean_pred = predictions.median(dim=0)[0]
        elif self.aggregation == "weighted":
            # TODO: 实现加权聚合
            weights = self._compute_weights(predictions)
            mean_pred = (predictions * weights.unsqueeze(1)).sum(dim=0)
        else:
            raise ValueError(f"未知的聚合方法: {self.aggregation}")

        # 认知不确定性（模型间的差异）
        epistemic = predictions.var(dim=0)

        # 偶然不确定性（平均预测的内在不确定性）
        aleatoric = mean_pred * (1 - mean_pred)

        # 总不确定性
        total = epistemic + aleatoric

        results = {
            'mean': mean_pred,
            'epistemic': epistemic,
            'aleatoric': aleatoric,
            'total': total,
        }

        if return_samples:
            results['samples'] = predictions

        return results

    def _compute_weights(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        计算模型权重（基于历史性能）

        Args:
            predictions: [M, B] 所有模型的预测

        Returns:
            weights: [M] 归一化权重
        """
        # 简单实现：使用均匀权重
        # TODO: 基于验证集性能计算权重
        num_models = predictions.shape[0]
        return torch.ones(num_models) / num_models
```

---

### Phase 2: 校准模块 (Week 2)

**文件:** `src/modules/uncertainty/calibration.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class TemperatureScaling(nn.Module):
    """
    温度缩放校准

    简单但有效的校准方法：
    calibrated_prob = sigmoid(logit / temperature)
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        应用温度缩放

        Args:
            logits: 原始 logits [N]

        Returns:
            校准后的概率 [N]
        """
        return torch.sigmoid(logits / self.temperature)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        在验证集上优化温度参数

        Args:
            logits: 验证集 logits [N]
            labels: 验证集标签 [N]
            lr: 学习率
            max_iter: 最大迭代次数
        """
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nn.BCEWithLogitsLoss()(
                logits / self.temperature,
                labels.float()
            )
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        print(f"最优温度: {self.temperature.item():.4f}")


class IsotonicRegression:
    """
    保序回归校准

    更灵活但需要更多数据
    """

    def __init__(self):
        self.calibrator = None

    def fit(self, probs: torch.Tensor, labels: torch.Tensor):
        """
        拟合保序回归模型
        """
        from sklearn.isotonic import IsotonicRegression as IR

        probs_np = probs.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        self.calibrator = IR(out_of_bounds='clip')
        self.calibrator.fit(probs_np, labels_np)

    def __call__(self, probs: torch.Tensor) -> torch.Tensor:
        """
        应用校准
        """
        if self.calibrator is None:
            raise ValueError("请先调用 fit() 方法")

        probs_np = probs.detach().cpu().numpy()
        calibrated = self.calibrator.transform(probs_np)

        return torch.from_numpy(calibrated).to(probs.device)


def get_calibrator(method: str = "temperature_scaling"):
    """
    工厂函数：获取校准器
    """
    if method == "temperature_scaling":
        return TemperatureScaling()
    elif method == "isotonic":
        return IsotonicRegression()
    else:
        raise ValueError(f"未知的校准方法: {method}")
```

---

### Phase 3: 评估指标 (Week 2)

**文件:** `src/modules/uncertainty/metrics.py`

```python
import torch
import numpy as np
from typing import Tuple


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10
) -> float:
    """
    期望校准误差 (ECE)

    Args:
        probs: 预测概率 [N]
        labels: 真实标签 [N]
        n_bins: 分箱数量

    Returns:
        ECE 值
    """
    probs = probs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def brier_score(
    probs: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Brier 分数（越小越好）

    BS = mean((prob - label)^2)
    """
    return ((probs - labels.float()) ** 2).mean().item()


def uncertainty_auroc(
    uncertainties: torch.Tensor,
    errors: torch.Tensor
) -> float:
    """
    用不确定性预测错误的 AUROC

    Args:
        uncertainties: 不确定性分数 [N]
        errors: 是否预测错误 [N] (0 或 1)

    Returns:
        AUROC 值
    """
    from sklearn.metrics import roc_auc_score

    unc_np = uncertainties.detach().cpu().numpy()
    err_np = errors.detach().cpu().numpy()

    return roc_auc_score(err_np, unc_np)


def compute_all_metrics(
    probs: torch.Tensor,
    labels: torch.Tensor,
    uncertainties: torch.Tensor
) -> dict:
    """
    计算所有不确定性指标
    """
    preds = (probs > 0.5).float()
    errors = (preds != labels).float()

    metrics = {
        'ece': expected_calibration_error(probs, labels),
        'brier': brier_score(probs, labels),
        'unc_auroc': uncertainty_auroc(uncertainties, errors),
    }

    return metrics
```

---

## 🧪 使用示例

### 示例 1: MC Dropout

```python
from src.modules.uncertainty import MCDropoutEstimator

# 配置
config = {
    "num_samples": 100,
    "dropout_rate": 0.1
}

# 创建估计器
unc_estimator = MCDropoutEstimator(config)

# 使用
results = unc_estimator(features, model)

print(f"预测: {results['mean']}")
print(f"认知不确定性: {results['epistemic']}")
print(f"总不确定性: {results['total']}")

# 校准
unc_estimator.calibrate(val_logits, val_labels)
```

### 示例 2: Deep Ensembles

```python
from src.modules.uncertainty import EnsembleEstimator

# 创建ensemble
config = {"num_models": 5, "aggregation": "mean"}
ensemble = EnsembleEstimator(config)

# 添加训练好的模型
for model_path in model_paths:
    model = load_model(model_path)
    ensemble.add_model(model)

# 预测
results = ensemble(features)
```

---

## ✅ 测试清单

- [ ] 单元测试：MC Dropout
- [ ] 单元测试：Deep Ensembles
- [ ] 单元测试：温度缩放
- [ ] 集成测试：完整流程
- [ ] 性能测试：推理速度
- [ ] 校准质量测试：ECE < 0.05

---

**实现者:** UAAM-Phish Team
**代码审查:** Pending
