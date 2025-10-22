# Consistency Module - 实现细节

> **实现版本:** 1.0
> **状态:** 规划中
> **规格文档:** [consistency.md](../specs/consistency.md)
> **最后更新:** 2025-10-22

---

## 📂 文件结构

```
src/modules/consistency/
├── __init__.py
├── checker.py           # 一致性检查器
├── metrics.py           # 一致性指标
└── rules.py             # 规则引擎
```

---

## 🔨 实现示例

### ConsistencyChecker 实现

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tensor


class ConsistencyChecker(nn.Module):
    """
    跨模态一致性检查器
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.threshold = config.get("consistency_threshold", 0.7)
        self.conflict_threshold = config.get("conflict_threshold", 0.3)

    def forward(
        self,
        predictions: Dict[str, Tensor],
        confidences: Dict[str, Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        检查一致性

        Args:
            predictions: {modality: pred_tensor} 各模态预测
            confidences: {modality: conf_tensor} 各模态置信度

        Returns:
            一致性检查结果
        """
        # 收集所有预测
        pred_list = list(predictions.values())
        pred_stack = torch.stack(pred_list)  # [M, B]

        # 计算一致性分数
        # 方法1: 简单投票一致性
        mode_pred, _ = torch.mode(pred_stack, dim=0)
        agreement = (pred_stack == mode_pred.unsqueeze(0)).float()
        consistency_score = agreement.mean(dim=0)  # [B]

        # 方法2: 如果有置信度，使用加权一致性
        if confidences is not None:
            conf_list = list(confidences.values())
            conf_stack = torch.stack(conf_list)  # [M, B]

            # 加权投票
            weighted_votes = (pred_stack * conf_stack).sum(dim=0)
            total_conf = conf_stack.sum(dim=0)
            weighted_pred = (weighted_votes / (total_conf + 1e-9) > 0.5).float()

            # 加权一致性
            weighted_agreement = (pred_stack == weighted_pred.unsqueeze(0)).float()
            weighted_consistency = (weighted_agreement * conf_stack).sum(dim=0) / (total_conf + 1e-9)

            consistency_score = 0.5 * consistency_score + 0.5 * weighted_consistency

        # 检测冲突
        is_consistent = consistency_score >= self.threshold
        has_conflict = consistency_score < self.conflict_threshold

        # 可靠性评分
        reliability = self._compute_reliability(
            consistency_score,
            confidences
        )

        return {
            'consistency_score': consistency_score,
            'is_consistent': is_consistent,
            'has_conflict': has_conflict,
            'reliability': reliability,
        }

    def _compute_reliability(
        self,
        consistency: Tensor,
        confidences: Dict[str, Tensor] = None
    ) -> Tensor:
        """
        计算可靠性分数

        reliability = consistency * avg_confidence
        """
        if confidences is None:
            return consistency

        conf_list = list(confidences.values())
        avg_conf = torch.stack(conf_list).mean(dim=0)

        return consistency * avg_conf
```

### 一致性指标

```python
def krippendorff_alpha(predictions: Tensor) -> Tensor:
    """
    Krippendorff's Alpha 一致性系数

    Args:
        predictions: [M, B] M个模态的B个预测

    Returns:
        alpha: [B] 每个样本的一致性系数
    """
    # 简化实现
    M, B = predictions.shape

    # 计算观察到的不一致
    disagreement = 0
    for i in range(M):
        for j in range(i+1, M):
            disagreement += (predictions[i] != predictions[j]).float()

    observed_disagreement = disagreement / (M * (M-1) / 2)

    # Alpha = 1 - observed / expected
    # 简化：假设期望不一致为 0.5
    expected_disagreement = 0.5

    alpha = 1 - observed_disagreement / expected_disagreement
    return alpha
```

---

## 🧪 使用示例

```python
# 创建检查器
checker = ConsistencyChecker({
    "consistency_threshold": 0.7,
    "conflict_threshold": 0.3
})

# 检查一致性
predictions = {
    "url": url_preds,
    "html": html_preds,
    "image": img_preds
}

confidences = {
    "url": url_conf,
    "html": html_conf,
    "image": img_conf
}

results = checker(predictions, confidences)

# 处理不一致的样本
inconsistent_mask = ~results['is_consistent']
flagged_samples = data[inconsistent_mask]
```

---

**实现者:** UAAM-Phish Team
