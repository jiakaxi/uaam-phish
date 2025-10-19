# UAAM-Phish — URL-only MVP (Lightning)

这是一个最小化的第一个里程碑：基于BERT的URL分类器，使用PyTorch Lightning进行端到端训练。

## 1) 安装
```bash
python -m venv .venv && source .venv/bin/activate  # 或使用 conda
pip install -r requirements.txt
```

## 2) 准备数据
在 `data/processed/` 目录下创建三个CSV文件，包含以下列：

```
url_text,label
http://example.com/login?session=...,0
http://paypal.com.secure-update.example.cn/verify,1
...
```

或者，将单个源CSV文件放在 `data/raw/urls.csv` 并运行：
```bash
python scripts/preprocess.py --src data/raw/urls.csv --outdir data/processed
```

## 3) 配置
- 主配置：`configs/default.yaml`
- 硬件/数据配置文件：
  - 本地小数据集：`configs/profiles/local.yaml`
  - 服务器大数据集：`configs/profiles/server.yaml`

您可以通过设置环境变量 `DATA_ROOT` 来切换数据根目录而无需编辑配置：
```bash
export DATA_ROOT=/path/to/processed
```

## 4) 训练和测试
### 本地（小子集，单GPU）
```bash
export DATA_ROOT=./data/processed
python scripts/train.py --profile local
```

### 服务器（完整数据集，GPU / 多GPU）
```bash
export DATA_ROOT=/data/uaam_phish/processed
python scripts/train.py --profile server
```
（对于多GPU，在 `configs/profiles/server.yaml` 中设置 `devices` 和 `strategy: ddp`。）

指标（loss, F1, AUROC, FPR）将按epoch记录。

## 5) 下一步
- 在服务器上增加 `train.epochs` 和批量大小。
- 调整配置文件中的 `sample_fraction` 以便在本地更快迭代。
- MVP稳定后，集成HTML图和截图编码器，然后是UAAM。
