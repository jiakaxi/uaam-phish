# Consistency Module - 技术规格

> **模块名称:** 一致性检查模块
> **版本:** 1.0
> **状态:** 规划中
> **最后更新:** 2025-10-22

---

## 📋 概述

一致性检查模块负责验证多模态预测之间的一致性，用于：
- 检测模态间的矛盾
- 提供可靠性评分
- 支持异常检测
- 增强系统鲁棒性

---

## 🎯 功能目标

### 核心功能
- **跨模态一致性检查**：检查 URL/HTML/图像预测是否一致
- **矛盾检测**：识别模态间的显著差异
- **可靠性评分**：基于一致性计算整体可靠性
- **规则验证**：验证业务规则和领域知识

### 输入输出

#### 输入
```python
{
    "url_pred": Tensor[B],       # URL模态预测
    "html_pred": Tensor[B],      # HTML模态预测
    "img_pred": Tensor[B],       # 图像模态预测
    "url_conf": Tensor[B],       # URL置信度
    "html_conf": Tensor[B],      # HTML置信度
    "img_conf": Tensor[B],       # 图像置信度
}
```

#### 输出
```python
{
    "consistency_score": Tensor[B],  # 一致性分数 [0,1]
    "is_consistent": Tensor[B],      # 是否一致
    "conflicts": List[str],          # 冲突描述
    "reliability": Tensor[B],        # 可靠性分数
}
```

---

## 📊 一致性度量方法

### 方法 1: 预测一致性

```python
# Krippendorff's Alpha
consistency = krippendorff_alpha(predictions)

# 简单一致性率
consistency = (url_pred == html_pred == img_pred).float()
```

### 方法 2: 置信度加权一致性

```python
# 加权一致性
weights = F.softmax(torch.stack([url_conf, html_conf, img_conf]), dim=0)
weighted_pred = (weights * predictions).sum(dim=0)
```

### 方法 3: 语义一致性

检查提取的特征是否在语义空间中对齐：

```python
# 特征相似度
url_feat = url_encoder(url)
html_feat = html_encoder(html)
similarity = cosine_similarity(url_feat, html_feat)
consistency = (similarity > threshold).float()
```

---

## 🔧 接口设计

```python
class ConsistencyChecker(nn.Module):
    def __init__(self, config: Dict):
        self.threshold = config.get("threshold", 0.7)
        self.method = config.get("method", "voting")

    def check_consistency(
        self,
        predictions: Dict[str, Tensor],
        confidences: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        检查一致性
        """
        pass

    def detect_conflicts(
        self,
        predictions: Dict[str, Tensor]
    ) -> List[str]:
        """
        检测冲突
        """
        pass
```

---

**规格文档:** [consistency.md](../specs/consistency.md)
**实现文档:** [consistency_impl.md](../impl/consistency_impl.md)
