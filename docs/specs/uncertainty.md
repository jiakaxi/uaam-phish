# Uncertainty Module - 技术规格

> **模块名称:** 不确定性估计模块
> **版本:** 1.0
> **状态:** 规划中
> **最后更新:** 2025-10-22

---

## 📋 概述

不确定性估计模块负责量化模型预测的不确定性，用于：
- 识别模型不自信的预测
- 提供可靠性评分
- 支持主动学习和人工审核
- 增强多模态融合的鲁棒性

---

## 🎯 功能目标

### 1. 核心功能
- **预测不确定性估计**：量化单个预测的不确定性
- **多方法支持**：Monte Carlo Dropout, Deep Ensembles, 贝叶斯神经网络
- **不确定性分解**：区分认知不确定性和偶然不确定性
- **校准**：输出校准后的置信度分数

### 2. 输入输出

#### 输入
```python
{
    "embeddings": Tensor[B, D],      # 特征嵌入
    "model": nn.Module,              # 训练好的模型
    "num_samples": int,              # MC 采样次数（默认100）
    "method": str,                   # "mc_dropout" | "ensemble" | "bayesian"
}
```

#### 输出
```python
{
    "predictions": Tensor[B],        # 预测标签
    "probabilities": Tensor[B],      # 预测概率
    "epistemic_unc": Tensor[B],      # 认知不确定性
    "aleatoric_unc": Tensor[B],      # 偶然不确定性
    "total_unc": Tensor[B],          # 总不确定性
    "confidence": Tensor[B],         # 校准后的置信度
}
```

---

## 📊 方法详细说明

### 方法 1: Monte Carlo Dropout (MC Dropout)

**原理：**
- 训练时使用 Dropout
- 推理时保持 Dropout 激活
- 多次前向传播获得预测分布

**优势：**
- ✅ 实现简单
- ✅ 计算高效
- ✅ 适用于现有模型

**参数：**
```python
mc_dropout_config = {
    "dropout_rate": 0.1,      # Dropout 比例
    "num_samples": 100,       # MC 采样次数
    "use_batch_norm": False,  # 是否在推理时更新 BN
}
```

**不确定性计算：**
```python
# 认知不确定性（模型不确定性）
epistemic = var(predictions)

# 偶然不确定性（数据不确定性）
aleatoric = mean(predicted_variance)

# 总不确定性
total = epistemic + aleatoric
```

---

### 方法 2: Deep Ensembles

**原理：**
- 训练多个独立模型
- 聚合预测结果
- 计算预测方差

**优势：**
- ✅ 性能最佳
- ✅ 不确定性估计准确
- ✅ 无需特殊训练

**参数：**
```python
ensemble_config = {
    "num_models": 5,          # 模型数量
    "aggregation": "mean",    # "mean" | "weighted" | "voting"
    "diversity_loss": True,   # 是否使用多样性损失
}
```

---

### 方法 3: Bayesian Neural Networks

**原理：**
- 使用贝叶斯权重
- 变分推断
- 后验分布采样

**优势：**
- ✅ 理论基础扎实
- ✅ 不确定性估计准确

**参数：**
```python
bayesian_config = {
    "prior_std": 0.1,         # 先验标准差
    "posterior_samples": 50,  # 后验采样次数
    "kl_weight": 0.01,        # KL 散度权重
}
```

---

## 🔧 接口设计

### 主类：UncertaintyEstimator

```python
class UncertaintyEstimator(nn.Module):
    """
    不确定性估计器基类

    Args:
        method: 估计方法 ("mc_dropout" | "ensemble" | "bayesian")
        config: 方法特定配置
    """

    def __init__(self, method: str, config: Dict):
        super().__init__()
        self.method = method
        self.config = config
        self._setup_estimator()

    def forward(self, x: Tensor, model: nn.Module) -> Dict[str, Tensor]:
        """
        前向传播计算不确定性

        Args:
            x: 输入张量 [B, D]
            model: 预测模型

        Returns:
            包含预测和不确定性的字典
        """
        pass

    def calibrate(self, probs: Tensor, labels: Tensor) -> nn.Module:
        """
        校准置信度

        Args:
            probs: 预测概率 [N]
            labels: 真实标签 [N]

        Returns:
            校准模型
        """
        pass
```

### MC Dropout 实现

```python
class MCDropoutEstimator(UncertaintyEstimator):
    def forward(self, x: Tensor, model: nn.Module) -> Dict[str, Tensor]:
        # 启用 dropout
        model.train()

        # MC 采样
        predictions = []
        for _ in range(self.config["num_samples"]):
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [S, B, 1]

        # 计算统计量
        mean_pred = predictions.mean(dim=0)
        epistemic = predictions.var(dim=0)

        return {
            "predictions": (mean_pred > 0.5).float(),
            "probabilities": torch.sigmoid(mean_pred),
            "epistemic_unc": epistemic,
            "total_unc": epistemic,
        }
```

### Ensemble 实现

```python
class EnsembleEstimator(UncertaintyEstimator):
    def __init__(self, config: Dict):
        super().__init__("ensemble", config)
        self.models = nn.ModuleList([
            self._create_model()
            for _ in range(config["num_models"])
        ])

    def forward(self, x: Tensor, model: nn.Module = None) -> Dict[str, Tensor]:
        # 集成预测
        predictions = []
        for m in self.models:
            m.eval()
            with torch.no_grad():
                pred = m(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [M, B, 1]

        # 聚合
        mean_pred = predictions.mean(dim=0)
        epistemic = predictions.var(dim=0)

        return {
            "predictions": (mean_pred > 0.5).float(),
            "probabilities": torch.sigmoid(mean_pred),
            "epistemic_unc": epistemic,
            "total_unc": epistemic,
        }
```

---

## 📈 评估指标

### 1. 不确定性质量指标

- **ECE (Expected Calibration Error)**: 期望校准误差
- **NLL (Negative Log-Likelihood)**: 负对数似然
- **Brier Score**: 预测准确性

### 2. 可靠性指标

- **AUROC-uncertainty**: 用不确定性预测错误的能力
- **Coverage**: 高置信度预测的覆盖率
- **Risk-coverage curve**: 风险-覆盖率曲线

---

## 🎛️ 配置参数

### 全局配置

```yaml
uncertainty:
  method: mc_dropout  # mc_dropout | ensemble | bayesian
  calibration: true   # 是否校准

  # MC Dropout 配置
  mc_dropout:
    dropout_rate: 0.1
    num_samples: 100

  # Ensemble 配置
  ensemble:
    num_models: 5
    aggregation: mean

  # 校准配置
  calibration:
    method: temperature_scaling  # temperature_scaling | isotonic
    val_size: 0.2
```

---

## 🔗 与其他模块的集成

### 1. 与编码器集成

```python
# URL 编码器 + 不确定性
url_embedding = url_encoder(batch)
uncertainty_output = uncertainty_estimator(url_embedding, classifier)
```

### 2. 与融合模块集成

```python
# 提供可靠性权重
fusion_weights = 1.0 / (1.0 + uncertainty_output["total_unc"])
```

### 3. 与一致性检查集成

```python
# 高不确定性 + 低一致性 = 需要人工审核
if uncertainty > threshold and consistency < threshold:
    flag_for_review()
```

---

## 📚 参考文献

1. **MC Dropout**: Gal & Ghahramani (2016) - "Dropout as a Bayesian Approximation"
2. **Deep Ensembles**: Lakshminarayanan et al. (2017) - "Simple and Scalable Predictive Uncertainty Estimation"
3. **Calibration**: Guo et al. (2017) - "On Calibration of Modern Neural Networks"

---

## ✅ 验收标准

- [ ] 实现至少两种不确定性估计方法
- [ ] ECE < 0.05 （校准后）
- [ ] AUROC-uncertainty > 0.8
- [ ] 推理速度 < 100ms/sample（MC Dropout）
- [ ] 完整的单元测试覆盖
- [ ] 详细的使用文档

---

**作者:** UAAM-Phish Team
**审核:** Pending
**实现文档:** [uncertainty_impl.md](../impl/uncertainty_impl.md)
