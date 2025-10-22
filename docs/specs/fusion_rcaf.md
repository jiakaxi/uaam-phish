# Fusion (RCAF) Module - 技术规格

> **模块名称:** 多模态融合模块 (RCAF)
> **版本:** 1.0
> **状态:** 规划中
> **最后更新:** 2025-10-22

---

## 📋 概述

**RCAF (Reliability-Constrained Attention Fusion)** 是一个基于可靠性约束的注意力融合方法，用于：
- 融合 URL、HTML、图像多模态特征
- 根据可靠性动态调整模态权重
- 处理模态缺失和噪声
- 提供可解释的融合决策

---

## 🎯 核心思想

### 融合公式

```
融合特征 = Σ(α_i * f_i)

其中:
α_i = attention_weight_i * reliability_i
f_i = modality_embedding_i
```

### 关键组件

1. **注意力机制**：学习模态重要性
2. **可靠性约束**：基于不确定性和一致性调整权重
3. **门控机制**：处理模态缺失

---

## 📊 架构设计

### 输入

```python
{
    "url_embedding": Tensor[B, D],      # URL特征
    "html_embedding": Tensor[B, D],     # HTML特征
    "img_embedding": Tensor[B, D],      # 图像特征
    "url_uncertainty": Tensor[B],       # URL不确定性
    "html_uncertainty": Tensor[B],      # HTML不确定性
    "img_uncertainty": Tensor[B],       # 图像不确定性
    "consistency_score": Tensor[B],     # 一致性分数
    "available_modalities": List[str],  # 可用模态
}
```

### 输出

```python
{
    "fused_embedding": Tensor[B, D],    # 融合特征
    "attention_weights": Dict[str, Tensor[B]],  # 注意力权重
    "reliability_scores": Dict[str, Tensor[B]], # 可靠性分数
    "prediction": Tensor[B],            # 最终预测
    "confidence": Tensor[B],            # 预测置信度
}
```

---

## 🔧 核心模块

### 1. 注意力模块

```python
class MultimodalAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, embeddings: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """
        计算注意力权重

        Returns:
            fused: 融合特征
            weights: 注意力权重
        """
        pass
```

### 2. 可靠性计算

```python
def compute_reliability(
    uncertainty: Tensor,
    consistency: Tensor,
    alpha: float = 0.5
) -> Tensor:
    """
    计算模态可靠性

    reliability = α * (1 - uncertainty) + (1 - α) * consistency
    """
    return alpha * (1 - uncertainty) + (1 - alpha) * consistency
```

### 3. RCAF 融合器

```python
class RCAFFusion(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.attention = MultimodalAttention(config['embedding_dim'])
        self.gate = GatingNetwork(config)
        self.classifier = nn.Linear(config['embedding_dim'], 1)

    def forward(
        self,
        embeddings: Dict[str, Tensor],
        uncertainties: Dict[str, Tensor],
        consistency: Tensor
    ) -> Dict[str, Tensor]:
        """
        RCAF 融合
        """
        # 计算可靠性
        reliabilities = {
            mod: compute_reliability(unc, consistency)
            for mod, unc in uncertainties.items()
        }

        # 注意力融合
        fused, attn_weights = self.attention(list(embeddings.values()))

        # 可靠性加权
        reliability_weights = torch.stack(list(reliabilities.values()))
        reliability_weights = F.softmax(reliability_weights, dim=0)

        # 组合权重
        final_weights = attn_weights * reliability_weights
        final_weights = final_weights / final_weights.sum(dim=0)

        # 加权融合
        weighted_embeddings = [
            w.unsqueeze(1) * emb
            for w, emb in zip(final_weights, embeddings.values())
        ]
        fused = torch.stack(weighted_embeddings).sum(dim=0)

        # 预测
        logits = self.classifier(fused)
        probs = torch.sigmoid(logits)

        return {
            'fused_embedding': fused,
            'attention_weights': dict(zip(embeddings.keys(), attn_weights)),
            'reliability_scores': reliabilities,
            'prediction': (probs > 0.5).float(),
            'confidence': probs,
        }
```

---

## 📈 训练策略

### 损失函数

```python
total_loss = λ1 * classification_loss
           + λ2 * attention_regularization
           + λ3 * diversity_loss
```

### 正则化

- **注意力平滑**: 防止过度依赖单一模态
- **多样性损失**: 鼓励模态互补

---

**规格文档:** [fusion_rcaf.md](../specs/fusion_rcaf.md)
**实现文档:** [fusion_rcaf_impl.md](../impl/fusion_rcaf_impl.md)
