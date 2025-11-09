# Multimodal Data Schema (S0 Baseline)

> 统一数据格式可以确保 URL / HTML / 图像三模态 DataModule 在所有实验（IID、Brand-OOD、Corruption）中行为一致。本文件约定 **必需列**、**可选列**、**Corruption 扩展列** 以及验证建议。

## 1. 必需列（MultimodalDataModule）

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `id` | string / int | 唯一 | 样本 ID，贯穿 splits / artifacts |
| `label` | int | {0,1} | 0=benign，1=phishing |
| `url_text` | string | 非空 | URL 文本（URL encoder 输入） |
| `html_path` | string | 可相对/绝对 | HTML 原文路径，允许为空字符串 |
| `img_path` | string | 可相对/绝对 | 截图路径，允许为空字符串 |
| `split` | string | train/val/test | 仅当 `split_protocol=presplit` 且使用 `master_csv` 时必备 |

> S0 Hydra 配置推荐直接提供 `train_csv/val_csv/test_csv`，因此 `split` 列并非强制；但 Master Data 仍需提供用于回溯。

## 2. 可选列（Brand-OOD + 日志）

| 列名 | 类型 | 说明 |
|------|------|------|
| `brand` | string | 品牌（需归一化：strip + lower） |
| `timestamp` | ISO string | 时间戳（Brand-OOD 三重门禁） |
| `domain` | string | 原始域名 |
| `etld_plus_one` | string | eTLD+1（若缺失，分割脚本会调用 `tldextract` 自动生成） |
| `source` | string | 数据来源（dataset 根路径/采集渠道） |
| `url_text_corrupt` | string | URL corruption 结果（可选） |
| `html_path_corrupt` | string | HTML corruption 路径（可选） |

## 3. Corruption 扩展列（必需）

用于 `workspace/data/corrupt/**` 产物，以及 DataLoader 优先读取逻辑：

| 列名 | 类型 | 说明 |
|------|------|------|
| `img_path_corrupt` | string | **优先读取**的 corruption 截图；若为相对路径，则默认相对于 `workspace/data/corrupt` |
| `img_sha256_corrupt` | string | corruption 截图 SHA256 校验 |

> 其余模态（URL / HTML）腐化列按照可选字段处理。若缺失，则 DataLoader 自动回退到原始 `img_path` / `url_text` / `html_path`。

## 4. 目录约定

```
workspace/
  data/
    splits/
      iid/{train,val,test}.csv
      brandood/{train,val,test}.csv
    corrupt/
      url/**.csv
      html/**.csv
      img/shot/<sample_id>.jpg     # img_path_corrupt 相对该目录
```

* 训练/评估脚本默认读取 `workspace/data/**`；原始 master 数据保持在 `data/processed/master_v2.csv`，只读。
* `MultimodalDataModule` 新增 `train_csv/val_csv/test_csv` 参数。配置示例：
  ```yaml
  datamodule:
    _target_: src.data.multimodal_datamodule.MultimodalDataModule
    train_csv: workspace/data/splits/iid/train.csv
    val_csv: workspace/data/splits/iid/val.csv
    test_csv: workspace/data/splits/iid/test.csv
    image_dir: data/processed/screenshots
    corrupt_root: workspace/data/corrupt
    batch_size: 64
    num_workers: 4
    persistent_workers: false
  ```

## 5. 示例（最小 + 扩展）

```csv
id,label,url_text,html_path,img_path,brand,timestamp,etld_plus_one,source,img_path_corrupt,img_sha256_corrupt
iid_train_0001,0,http://example.com/login,data/raw/html/0001.html,data/raw/img/0001.png,example,2024-05-01T10:00:00Z,example.com,benign_dataset,,
iid_train_0002,1,http://paypal-secure.cn/update,data/raw/html/0002.html,data/raw/img/0002.png,paypal,2024-05-01T10:05:00Z,paypal-secure.cn,phish_dataset,,
iid_test_0100,1,http://apple.id-confirm.ru/index,data/raw/html/0100.html,data/raw/img/0100.png,apple,2024-05-01T11:00:00Z,apple.id-confirm.ru,phish_dataset,img/shot/iid_test_0100.jpg,6a47...
```

* IID/Brand-OOD Split CSV 必须包含 **表 1 + 表 2** 列。
* Corruption CSV 在此基础上新增表 3 列，并可追加 `url_text_corrupt/html_path_corrupt`。

## 6. 校验建议

1. **列名检查**：
   ```bash
   python tools/split_iid.py --check-only --in data/processed/master_v2.csv
   ```
2. **Schema 验证**（保留原有 make 目标）：
   ```bash
   make validate-data
   ```
3. **Corruption 校验**：
   * 确认 `img_path_corrupt` 文件存在且 SHA256 匹配。
   * DataLoader 会在日志中输出第一次 fallback（若 `img_path_corrupt` 缺失）。

## 7. 常见问题

| 问题 | 解决方案 |
|------|----------|
| 缺少 `etld_plus_one` | 运行 `tools/split_iid.py` / `split_brandood.py` 会自动生成 |
| `img_path_corrupt` 为空 | DataModule 自动回退到 `img_path`，但质量门禁会报告缺失 |
| Brand-OOD 门禁失败 | 确保 `brand`、`timestamp`、`etld_plus_one`、`source` 均在 CSV 中 |
| ReduceLROnPlateau 没有监控数据 | 确认 `val/loss` 在 Lightning 日志中存在 |

---

如需扩展 schema（例如新增语言、可疑标签等），请在本文件补充列定义并同步更新 `tools/*` 分割脚本与 `src/data/multimodal_datamodule.py`。
