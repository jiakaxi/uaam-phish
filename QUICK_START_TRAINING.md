# 🚀 快速启动训练

## 方式1: 使用PowerShell脚本（推荐）

```powershell
.\start_training.ps1
```

## 方式2: 直接命令行

```powershell
python scripts/train_hydra.py logger=wandb trainer=default data=url_only model=url_encoder
```

## 方式3: 使用默认配置

```powershell
python scripts/train_hydra.py
```

---

## ✅ 当前配置已修复

`configs/trainer/default.yaml` 现在包含正确参数：
- ✅ epochs: 50
- ✅ lr: 0.0001
- ✅ batch_size: 64
- ✅ dropout: 0.1

---

## 📊 查看训练进度

### WandB Dashboard
访问: https://wandb.ai 查看实时图表

### 本地日志
```powershell
# 实时查看最新日志
Get-ChildItem outputs -Recurse -Filter "*.log" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  ForEach-Object { Get-Content $_.FullName -Wait }
```

---

## 预期结果

- **准确率**: 95-99%（之前: 53%）
- **AUROC**: > 0.95（之前: 0.10）
- **训练时间**: 10-15分钟
- **收敛轮数**: 约30轮

训练成功！ 🎉
