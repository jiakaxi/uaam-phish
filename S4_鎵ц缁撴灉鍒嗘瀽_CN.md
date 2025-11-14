

---

## 🔴 新发现的问题 (首次训练测试)

### 问题: 所有样本的模态都无效

**症状**:
\`\`text
[WARNING] Some samples have no valid modalities! Using uniform weights.
\`\`

这个警告在训练中大量出现，说明自适应融合退化为均匀权重。`n
**影响**: 🔴 **致命** - S4 失去了核心价值（自适应调整权重）`n
**可能原因**:
1. \_compute_consistency_batch\ 返回�?NaN (C-Module 处理批次数据问题)
2. \_compute_reliability\ 返回 NaN (熵计算问�?
3. image_path 字段在批次中无效

**需要立即修�?*:
- [ ] �?\_compute_reliability\ �?\_compute_consistency_batch\ 中添�?NaN 检查`n- [ ] 为无效值提供合理的 fallback (例如: r_m 默认 0.5, c_m 默认 0.0)
- [ ] 确保至少一个模态有有效�?r_m �?c_m

**临时 workaround**:
\`\`python
# �?_compute_reliability 中`ndef _compute_reliability(self, logits, modality):
    probs = torch.sigmoid(logits)
    entropy = -probs * torch.log(probs + 1e-8) - (1 - probs) * torch.log(1 - probs + 1e-8)
    reliability = 1.0 - entropy
    # 添加 NaN 检查`n    reliability = torch.nan_to_num(reliability, nan=0.5)  # 默认中等可靠性`n    return reliability

# �?_compute_consistency_batch 中`ndef _compute_consistency_batch(self, batch):
    # ... 现有代码 ...
    c_m = torch.tensor(...)
    # 添加 NaN 检查`n    c_m = torch.nan_to_num(c_m, nan=0.0)  # 默认无一致性信号`n    return c_m
\`\`

**优先�?*: 🔴 **P0** - 必须立即修复才能继续实验
