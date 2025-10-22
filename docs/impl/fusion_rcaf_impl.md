# Fusion (RCAF) Module - 实现细节

> **实现版本:** 1.0
> **状态:** 规划中
> **规格文档:** [fusion_rcaf.md](../specs/fusion_rcaf.md)
> **最后更新:** 2025-10-22

---

## 📂 文件结构

```
src/modules/fusion/
├── __init__.py
├── rcaf.py              # RCAF主模块
├── attention.py         # 注意力机制
├── gating.py            # 门控网络
└── losses.py            # 融合损失函数
```

---

## 🔨 完整实现

### RCAF Fusion 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class RCAFFusion(nn.Module):
    """
    Reliability-Constrained Attention Fusion

    融合多模态特征，考虑：
    1. 注意力机制学习模态重要性
    2. 不确定性约束模态权重
    3. 一致性约束增强鲁棒性
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.embedding_dim = config['embedding_dim']
        self.num_modalities = config.get('num_modalities', 3)
        self.reliability_weight = config.get('reliability_weight', 0.5)

        # 注意力模块
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config.get('num_heads', 4),
            dropout=config.get('dropout', 0.1)
        )

        # 可靠性编码器
        self.reliability_encoder = nn.Sequential(
            nn.Linear(2, 32),  # [uncertainty, consistency]
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 门控网络（处理缺失模态）
        self.gate = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, self.num_modalities),
            nn.Sigmoid()
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.embedding_dim // 2, 1)
        )

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
        uncertainties: Optional[Dict[str, torch.Tensor]] = None,
        consistency: Optional[torch.Tensor] = None,
        mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            embeddings: {modality: [B, D]} 各模态嵌入
            uncertainties: {modality: [B]} 各模态不确定性
            consistency: [B] 一致性分数
            mask: {modality: [B]} 模态可用性掩码

        Returns:
            融合结果
        """
        batch_size = list(embeddings.values())[0].shape[0]
        device = list(embeddings.values())[0].device

        # 1. 堆叠所有模态嵌入
        modality_names = list(embeddings.keys())
        emb_list = [embeddings[mod] for mod in modality_names]
        emb_stack = torch.stack(emb_list, dim=1)  # [B, M, D]

        # 2. 计算注意力权重
        # 使用自注意力
        attn_output, attn_weights = self.attention(
            query=emb_stack.transpose(0, 1),  # [M, B, D]
            key=emb_stack.transpose(0, 1),
            value=emb_stack.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)  # [B, M, D]

        # 3. 计算可靠性权重
        if uncertainties is not None and consistency is not None:
            reliability_weights = []
            for mod in modality_names:
                unc = uncertainties.get(mod, torch.zeros(batch_size).to(device))
                cons = consistency

                # 编码可靠性
                reliability_input = torch.stack([unc, cons], dim=1)  # [B, 2]
                reliability = self.reliability_encoder(reliability_input).squeeze(1)  # [B]
                reliability_weights.append(reliability)

            reliability_weights = torch.stack(reliability_weights, dim=1)  # [B, M]
        else:
            reliability_weights = torch.ones(batch_size, len(modality_names)).to(device)

        # 4. 处理模态掩码（缺失模态）
        if mask is not None:
            mask_tensor = torch.stack([mask.get(mod, torch.ones(batch_size).to(device))
                                      for mod in modality_names], dim=1)  # [B, M]
            reliability_weights = reliability_weights * mask_tensor

        # 5. 归一化权重
        reliability_weights = F.softmax(reliability_weights, dim=1)  # [B, M]

        # 6. 加权融合
        weighted_embeddings = attn_output * reliability_weights.unsqueeze(2)  # [B, M, D]
        fused_embedding = weighted_embeddings.sum(dim=1)  # [B, D]

        # 7. 分类
        logits = self.classifier(fused_embedding).squeeze(1)  # [B]
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # 8. 构建输出
        return {
            'fused_embedding': fused_embedding,
            'prediction': preds,
            'probability': probs,
            'logits': logits,
            'attention_weights': {
                mod: reliability_weights[:, i]
                for i, mod in enumerate(modality_names)
            },
            'reliability_scores': {
                mod: reliability_weights[:, i]
                for i, mod in enumerate(modality_names)
            }
        }


class FusionLoss(nn.Module):
    """
    融合模型的损失函数
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.lambda_cls = config.get('lambda_cls', 1.0)
        self.lambda_reg = config.get('lambda_reg', 0.1)
        self.lambda_div = config.get('lambda_div', 0.1)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_weights: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        """
        # 分类损失
        cls_loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # 注意力正则化（防止过度集中）
        weights_stack = torch.stack(list(attention_weights.values()), dim=1)  # [B, M]
        entropy = -(weights_stack * torch.log(weights_stack + 1e-9)).sum(dim=1).mean()
        reg_loss = -entropy  # 最大化熵 = 最小化负熵

        # 多样性损失（鼓励不同模态关注不同方面）
        # TODO: 实现多样性损失
        div_loss = torch.tensor(0.0).to(logits.device)

        # 总损失
        total_loss = (self.lambda_cls * cls_loss +
                     self.lambda_reg * reg_loss +
                     self.lambda_div * div_loss)

        return {
            'total': total_loss,
            'classification': cls_loss,
            'regularization': reg_loss,
            'diversity': div_loss
        }
```

---

## 🧪 使用示例

```python
# 配置
config = {
    'embedding_dim': 768,
    'num_modalities': 3,
    'num_heads': 4,
    'dropout': 0.1,
    'reliability_weight': 0.5
}

# 创建模型
fusion_model = RCAFFusion(config)

# 前向传播
embeddings = {
    'url': url_embeddings,    # [B, 768]
    'html': html_embeddings,  # [B, 768]
    'image': img_embeddings   # [B, 768]
}

uncertainties = {
    'url': url_uncertainty,   # [B]
    'html': html_uncertainty, # [B]
    'image': img_uncertainty  # [B]
}

consistency = consistency_scores  # [B]

results = fusion_model(embeddings, uncertainties, consistency)

# 训练
loss_fn = FusionLoss(config)
losses = loss_fn(results['logits'], labels, results['attention_weights'])
losses['total'].backward()
```

---

**实现者:** UAAM-Phish Team
