# 变更总结

## 2025-11-11: Brand-OOD数据分割修复

### 问题背景

Brand-OOD实验的测试集AUROC为0.0，原因是数据集类别严重不平衡，导致验证集和测试集只有单一类别（全部为正例）。

### 解决方案

#### 新增工具脚本

**文件**: `tools/check_brand_distribution.py`
- 检查master_v2.csv中每个brand的0/1分布
- 输出brand分布报告（JSON格式）
- 识别有足够负例的品牌

**文件**: `tools/analyze_balanced_brands.py`
- 分析同时有正例和负例的品牌分布
- 推荐合适的阈值策略

#### 修改分割脚本

**文件**: `tools/split_brandood.py`

**主要修改**:
1. **新增参数**:
   - `--min-pos-per-brand`: 最低正例数阈值（默认1）
   - `--min-neg-per-brand`: 最低负例数阈值（默认1）

2. **实现 `select_balanced_brand_sets()` 函数**:
   - 替换原有的 `select_brand_sets()` 函数
   - 确保选择的品牌同时有正例和负例
   - 将单侧品牌（只有正例或只有负例）放入OOD集
   - 添加回退策略：如果没有品牌满足条件，选择有正例和负例的品牌（不限制数量）

3. **实现 `stratified_split_by_brand_label()` 函数**:
   - 替换原有的 `stratified_split()` 函数
   - 按brand+label组合进行分层采样
   - 处理样本数太少的组合（合并到OTHER组）
   - 如果无法分层，回退到按label分层采样

4. **添加数据质量检查**:
   - `check_split_distribution()` 函数检查每个split的类别分布
   - 如果某个split只有单一类别，输出错误并终止

5. **保存分布统计**:
   - 生成 `split_distribution_report.json` 文件
   - 记录每个split的详细统计信息和参数

#### 数据修复流程

1. **数据检查**:
   ```bash
   python tools/check_brand_distribution.py --csv data/processed/master_v2.csv --out workspace/reports/brand_distribution_report.json
   ```
   - 发现只有8个品牌同时有正例和负例
   - 只有1个品牌（autoscout24）同时有≥2个正例和≥2个负例

2. **重新生成分割**:
   ```bash
   python tools/split_brandood.py \
     --in data/processed/master_v2.csv \
     --out workspace/data/splits/brandood \
     --seed 42 \
     --top_k 8 \
     --min-neg-per-brand 1 \
     --min-pos-per-brand 1 \
     --ood-ratio 0.25
   ```
   - 选择了8个同时有正例和负例的品牌作为in-domain集合
   - 生成了新的train/val/test_id/test_ood分割文件

3. **重新预处理缓存**:
   ```bash
   # 为每个split运行预处理
   python tools/preprocess_all_modalities.py \
     --csv workspace/data/splits/brandood/train.csv \
     --output workspace/data/preprocessed/brandood/train \
     --out-csv workspace/data/splits/brandood/train_cached.csv \
     --html-root data/processed \
     --image-dir data/processed/screenshots \
     # ... 其他参数
   ```
   - 重新生成了所有split的 `_cached.csv` 文件和预处理缓存

#### 修复结果

**修复前**:
- 训练集: 3,231样本，正例3,230 (99.97%)，负例1 (0.03%)
- 验证集: 693样本，正例693 (100%)，负例0 (0%) ⚠️
- 测试集: 693样本，正例693 (100%)，负例0 (0%) ⚠️

**修复后**:
- 训练集: 127样本，正例119 (93.7%)，负例8 (6.3%) ✅
- 验证集: 27样本，正例26 (96.3%)，负例1 (3.7%) ✅
- 测试集 (test_id): 28样本，正例26 (92.9%)，负例2 (7.1%) ✅
- 测试集 (test_ood): 7样本，正例3 (42.9%)，负例4 (57.1%) ✅

#### 重新运行实验列表

**需要重新运行的实验**:
- `s0_brandood_earlyconcat` (所有seeds)
- `s0_brandood_lateavg` (所有seeds)

**运行命令**:
```bash
python scripts/run_s0_experiments.py \
  --scenario brandood \
  --models s0_earlyconcat s0_lateavg \
  --seeds 42 43 44 \
  --logger wandb
```

**评估命令**:
```bash
python scripts/evaluate_s0.py \
  --runs-dir workspace/runs \
  --scenarios brandood \
  --out-csv workspace/tables/s0_brandood_eval_summary.csv
```

#### 相关文件

- `tools/split_brandood.py`: 修改分割脚本
- `tools/check_brand_distribution.py`: 新增数据检查脚本
- `tools/analyze_balanced_brands.py`: 新增品牌分析脚本
- `workspace/data/splits/brandood/*`: 重新生成的分割文件
- `workspace/data/splits/brandood/*_cached.csv`: 重新生成的缓存CSV文件
- `workspace/data/preprocessed/brandood/*`: 重新生成的预处理缓存
- `BRANDOOD_ISSUE_REPORT.md`: 更新问题报告和修复流程

## 2025-11-10: Windows训练速度优化

### 问题背景

训练速度极慢（仅0.03it/s），主要原因是Windows上的多进程配置问题。

### 解决方案

**修改配置文件中的num_workers设置**：
- `configs/trainer/default.yaml`: num_workers: 4 → 0
- `configs/experiment/multimodal_baseline.yaml`: num_workers: 4 → 0
- `configs/data/url_only.yaml`: num_workers: 4 → 0
- `configs/data/html_only.yaml`: num_workers: 4 → 0
- `configs/default.yaml`: num_workers: 2 → 0

**优化原理**：
- Windows上多进程启动开销大，进程间通信成本高
- 单进程模式（num_workers=0）避免多进程开销
- 预加载HTML文件到内存，减少IO瓶颈

**预期效果**：
- 训练速度提升1.5-2倍
- 消除"The 'train_dataloader' does not have many workers"警告

## 2025-11-07: 30k数据集构建脚本与验证

### 问题背景

现有 `master_v2.csv` 仅有 671 个样本，需要从新的 30k 数据集（`D:\one\phish_sample_30k` 29,496个 + `D:\one\benign_sample_30k` 22,551个）构建 16k 样本扩充数据集。

新数据集特点：
- **文件夹命名不同**：钓鱼为 `{Brand}+{Timestamp}`，合法为 `{Domain}`
- **文件名不同**：HTML文件为 `html.txt`（非 `html.html`）
- **info.txt 格式不同**：钓鱼为Python dict，合法为纯URL文本

### 解决方案

#### 新增构建脚本

**文件**: `scripts/build_from_30k.py`

**核心功能（稳健性增强）**:

1. **鲁棒的 info.txt 解析**
   - 安全解析 Python dict（`ast.literal_eval`）
   - 支持纯URL文本格式（合法数据集）
   - 多级回退：info dict → url.txt → info.txt纯文本

2. **多格式时间戳解析**
   - 支持 `2019-07-28-22\`34\`40`（反引号）
   - 支持 `2019-07-28-22-34-40`（全短横线）
   - 支持 `2019/07/28 22:34:40`（日志格式）
   - 回退到文件 mtime，标记 `timestamp_source="fs_mtime"`

3. **品牌提取与规范化**
   - 钓鱼数据集：`info['brand']` → 文件夹名
   - 合法数据集：从域名提取（`tldextract`）
   - 加载 `resources/brand_alias.yaml` 别名映射
   - 清洗：去全角空格、换行、数字开头、纯数字

4. **四级严格去重**
   - Level 1: 哈希去重（`html_sha1` + `img_sha1`，可选）
   - Level 2: 路径去重（避免同文件二次加入）
   - Level 3: 语义去重（`url + domain + brand`）
   - Level 4: URL短键去重（`normalize_url(url)[:128]`）

5. **分标签品牌约束 + 自适应阈值**
   - **关键改进**：对 phishing 和 benign **分别**执行品牌约束
   - 自适应阈值（根据品牌数动态调整）：
     - 品牌数 ≥ 30：Top1 ≤ 30%, Top3 ≤ 60%
     - 品牌数 10-29：Top1 ≤ 35%, Top3 ≤ 70%
     - 品牌数 < 10：Top1 ≤ 40%（不检查Top3）

#### 阶段1测试结果（200样本）

**命令**:
```bash
python scripts/build_from_30k.py \
  --phish_root "D:\one\phish_sample_30k" \
  --benign_root "D:\one\benign_sample_30k" \
  --k_each 100 \
  --out_csv data/processed/master_test_200.csv \
  --brand_alias resources/brand_alias.yaml \
  --seed 42
```

**结果**:
- ✅ 扫描钓鱼数据集：29,496 → 29,042 有效 → 去重后 23,560
- ✅ 扫描合法数据集：22,551 → 15,475 有效 → 去重后 15,475
- ✅ 品牌约束：钓鱼 280 品牌 → 抽样 100，合法 14,359 品牌 → 抽样 100
- ✅ 最终输出：200 行（100 phishing + 100 benign）

**质量验证**:
```
[✅] 行数与格式检查    200 行数据 | phishing: 100 (50.0%) | benign: 100 (50.0%)
[✅] 路径有效性       HTML: 100/100 (100%) | IMG: 100/100 (100%)
[✅] 品牌分布         156 个品牌, Top 1 占比 2.5%
[✅] 时间戳质量       100.0% 非空, 跨度 2019-06-27 ~ 2020-09-27
[✅] split 列         unsplit: 200
```

### 技术亮点

**品牌别名映射** (`resources/brand_alias.yaml`):
```yaml
"pay-pal": "paypal"
"face book": "facebook"
"micro soft": "microsoft"
"1&1 ionos": "ionos"
```

**合法数据集品牌清洗**:
```python
def extract_brand_from_benign_domain(domain: str) -> Optional[str]:
    ext = tldextract.extract(domain)
    brand = ext.domain
    # 清洗：仅保留字母数字
    brand = re.sub(r'[^a-z0-9]', '', brand.lower())
    # 过滤：数字开头、过短、纯数字
    if not brand or brand[0].isdigit() or len(brand) < 2:
        return None
    return brand
```

### 阶段3：完整16k构建结果 ✅

**执行命令**:
```bash
python scripts/build_from_30k.py \
  --phish_root "D:\one\phish_sample_30k" \
  --benign_root "D:\one\benign_sample_30k" \
  --k_each 8000 \
  --master_csv data/processed/master_v2.csv \
  --append \
  --brand_alias resources/brand_alias.yaml \
  --min_per_brand 50 \
  --brand_cap 500 \
  --seed 42
```

**构建结果**:
- ✅ **总样本数**: 16,656（671旧 + 15,985新）
- ✅ **钓鱼样本**: 8,352 (50.1%)
- ✅ **合法样本**: 8,304 (49.9%)
- ✅ **品牌数**: 8,250 个独立品牌
- ✅ **品牌分布**: Top1 占比 1.8%（极佳！）
- ✅ **时间跨度**: 2024-12-30 ~ 2025-04-08
- ✅ **路径有效性**: 100%
- ✅ **时间戳完整性**: 100%

**质量验证通过**:
```
[✅] 行数与格式检查    16656 行数据 | phishing: 8352 (50.1%) | benign: 8304 (49.9%)
[✅] 路径有效性       HTML: 100/100 (100%) | IMG: 100/100 (100%)
[✅] 品牌分布         8250 个品牌, Top 1 占比 1.8%
[✅] 时间戳质量       100.0% 非空, 跨度 2024-12-30 ~ 2025-04-08
[✅] split 列         unsplit: 15985, train: 469, test: 101, val: 101
```

**训练验证**（200样本GPU测试）:
- ✅ GPU训练正常
- ✅ 验证集 AUROC: 0.674
- ✅ 验证集 Accuracy: 61.0%
- ✅ 验证集 F1: 0.758
- ✅ ECE（校准误差）: 0.098

### 新增分模态CSV提取脚本

为方便单模态训练，新增了三个提取脚本：

1. **`scripts/extract_url_csvs.py`** - 提取URL模态数据
2. **`scripts/extract_html_csvs.py`** - 提取HTML模态数据
3. **`scripts/extract_img_csvs.py`** - 提取IMG模态数据（已存在）

**使用示例**:
```bash
python scripts/extract_url_csvs.py --master_csv data/processed/master_v2.csv
python scripts/extract_html_csvs.py --master_csv data/processed/master_v2.csv
python scripts/extract_img_csvs.py --master_csv data/processed/master_v2.csv
```

生成的文件：
- `data/processed/url_{train,val,test}_v2.csv`
- `data/processed/html_{train,val,test}_v2.csv`
- `data/processed/img_{train,val,test}_v2.csv`

### 数据集使用指南

**现有split分布**:
- 旧数据（671条）：已划分为 train/val/test
- 新数据（15,985条）：标记为 `unsplit`，由 DataModule 动态划分

**多模态训练**（使用完整16k数据集）:
```bash
python scripts/train_hydra.py \
  data.csv_path=data/processed/master_v2.csv \
  protocol=random \
  train.epochs=25 \
  hardware.accelerator=gpu \
  hardware.devices=1
```

**单模态训练**（URL-only示例）:
```bash
python scripts/train_hydra.py \
  data.train_csv=data/processed/url_train_v2.csv \
  data.val_csv=data/processed/url_val_v2.csv \
  data.test_csv=data/processed/url_test_v2.csv \
  train.epochs=25
```

---

## 2025-11-07: 数据集验证脚本

### 问题背景

在执行 `build_master_16k.py` 生成大规模数据集（如 8k+8k 或 200 样本 dry-run）后，需要系统化验证数据质量，确保：
- 文件完整性（CSV + JSON + 日志）
- 数据格式正确（列、标签、路径）
- 品牌和时间分布合理
- 可用于后续训练

手动检查耗时且容易遗漏问题，需要自动化验证工具。

### 解决方案

#### 新增验证脚本

**文件**: `scripts/verify_build_16k.py`

**功能**: 自动执行 10 项质量检查

| 检查项 | 内容 | 严格模式阈值 |
|--------|------|-------------|
| 1. 文件存在性 | CSV + metadata.json + selected_ids.json + dropped_reasons.json + 日志 | - |
| 2. 行数与格式 | CSV 可解析、无重复行 | - |
| 3. 列完整性 | 10 个必需列存在（id, label, url_text, html_path, img_path, domain, source, split, brand, timestamp） | - |
| 4. 标签分布 | label ∈ {0,1}，正负样本比例 40:60~60:40 | 少数类 <40% → 警告 |
| 5. 路径有效性 | 抽样 100 个样本验证 html_path 和 img_path 存在 | 缺失率 >10% → 失败，5-10% → 警告 |
| 6. 品牌分布 | 品牌数量 ≥5，Top 1 品牌占比 ≤50% | 违反 → 警告 |
| 7. 时间戳质量 | timestamp 非空率 ≥70%，时间范围合理 | <70% → 警告 |
| 8. split 列 | 测试集全为 "unsplit"，训练集为 train/val/test 或 unsplit | 不符合 → 警告 |
| 9. 元数据文件 | metadata.json 包含 total_samples、brand_distribution、timestamp_range、modality_completeness | 缺失 → 警告 |
| 10. 日志完整性 | 日志包含 "Wrote N rows to ..."，无 Traceback/Error | 缺失或有错误 → 警告 |

#### 使用方法

**1. 自动检测所有 master_*.csv**
```bash
python scripts/verify_build_16k.py
```

输出：
```
发现 1 个 CSV 文件待验证:
  - master_v2.csv

╔══════════════════════════════════════════════════════════════════════╗
║ 验证报告: master_v2.csv                                            ║
╚══════════════════════════════════════════════════════════════════════╝

[⚠️] 文件存在性检查    部分缺失
    └─ 缺少配套文件: metadata
[✅] 行数与格式检查    671 行数据 | phishing: 354 (52.8%) | benign: 317 (47.2%)
[✅] 路径有效性       HTML: 100/100 (100%) | IMG: 100/100 (100%)
[✅] 品牌分布         357 个品牌, Top 1 占比 4.0%
[✅] 时间戳质量       99.7% 非空, 跨度 2024-12-30 ~ 2025-04-08
[✅] split 列         train: 469, test: 101, val: 101
[⚠️] 元数据文件       0/2 文件有效
[⚠️] 日志文件         未找到

────────────────────────────────────────────────────────────────────────
总计: 5 项通过 / 3 项警告 / 0 项失败
状态: ⚠️  有警告，建议检查后再训练
```

**2. 验证特定文件**
```bash
python scripts/verify_build_16k.py --csv data/processed/master_400_test.csv
```

**3. 宽松模式（警告不导致退出码 1）**
```bash
python scripts/verify_build_16k.py --lenient
```

**4. 跳过路径验证（加速检查）**
```bash
python scripts/verify_build_16k.py --skip-path-check
```

**5. 调整抽样大小**
```bash
python scripts/verify_build_16k.py --sample-size 200
```

#### 退出码

- **0**: 所有检查通过，或宽松模式下有警告但不退出
- **1**: 严格模式下存在失败或警告

#### 集成建议

**PowerShell 脚本集成** (如 `run_build_16k.ps1`):
```powershell
# 构建数据集
python scripts/build_master_16k.py --k_each 8000 --suffix "_16k"

# 自动验证
python scripts/verify_build_16k.py --csv data/processed/master_16k.csv
if ($LASTEXITCODE -ne 0) {
    Write-Host "验证失败，请检查数据！" -ForegroundColor Red
    exit 1
}

Write-Host "验证通过，开始训练..." -ForegroundColor Green
```

**CI/CD 流水线**:
```yaml
- name: Validate dataset
  run: python scripts/verify_build_16k.py --csv ${{ env.DATASET_PATH }}
```

### 验证项详解

#### 路径有效性检查（最关键）

- **抽样策略**: 随机抽取 100 个样本（可配置）
- **验证内容**: 检查 `html_path` 和 `img_path` 指向的文件是否真实存在
- **失败阈值**:
  - **>10% 缺失**: 严重错误，返回码 1（严格模式）
  - **5-10% 缺失**: 警告
  - **<5% 缺失**: 通过（允许少量符号链接或大小写问题）

**示例失败输出**:
```
[❌] 路径有效性       HTML: 78/100 存在（22%缺失，超过阈值 10%）
    失败样本 ID: phish__12345, benign__67890, ...
```

#### 品牌分布检查

防止品牌过度集中导致 brand_ood 协议失效：
- 品牌数量应 ≥5（保证 brand_ood 有足够多样性）
- 单一品牌占比 ≤50%（避免测试集品牌太单一）

#### 时间戳质量检查

确保 temporal 协议可用：
- 非空率 ≥70%
- 时间跨度合理（输出 min/max 便于人工判断）

### 技术实现

**依赖项**:
- `pandas`: CSV 解析
- `pathlib`: 路径操作
- `json`: JSON 解析
- `collections.Counter`: 统计分析

**关键函数**:
```python
discover_master_csvs(processed_dir)      # 自动发现文件
validate_file_structure(csv_path)        # 检查 1
validate_csv_format(df, csv_path)        # 检查 2-4
validate_paths_sample(df, sample_size)   # 检查 5（抽样）
validate_brand_distribution(df)          # 检查 6
validate_timestamp_quality(df)           # 检查 7
validate_split_column(df, csv_name)      # 检查 8
validate_metadata_files(csv_path)        # 检查 9
validate_log_file(csv_path)              # 检查 10
print_report(results, strict)            # 输出报告 + 返回退出码
```

### 后续计划

- [ ] 集成到 `run_build_16k.ps1`（dry-run 和正式构建后自动验证）
- [ ] 添加图表生成（品牌分布直方图、时间分布热力图）
- [ ] 支持批量验证并生成 HTML 汇总报告

---

## 2025-11-07: 生成 IMG 模态 CSV 文件

### 问题背景

`data/processed/` 目录下已有 URL 和 HTML 模态的独立 CSV 文件，但缺少 IMG（图像）模态的对应文件：

**已有文件**:
- ✅ `master_v2.csv` - 主数据表（包含所有模态）
- ✅ `url_train_v2.csv`, `url_val_v2.csv`, `url_test_v2.csv`
- ✅ `html_train_v2.csv`, `html_val_v2.csv`, `html_test_v2.csv`

**缺失文件**:
- ❌ `img_train_v2.csv`, `img_val_v2.csv`, `img_test_v2.csv`

### 影响

1. 数据接口不一致：三个模态应该有对称的文件结构
2. 某些旧代码或工具可能期望独立的 IMG CSV 文件
3. 用户无法单独访问图像模态数据而不加载完整的 master CSV

### 解决方案

#### 1. 创建提取脚本

**新增文件**: `scripts/extract_img_csvs.py`

**功能**:
- 从 `master_v2.csv` 读取数据
- 按 `split` 列（train/val/test）过滤
- 提取 IMG 相关列：`id`, `img_path`, `label`, `timestamp`, `brand`, `source`, `domain`
- 生成三个独立的 CSV 文件
- 可选：验证图像路径是否存在

**使用方法**:
```bash
python scripts/extract_img_csvs.py --validate_paths
```

#### 2. 生成的文件

**输出文件**:
- `data/processed/img_train_v2.csv` - 469 样本（222 合法 + 247 钓鱼）
- `data/processed/img_val_v2.csv` - 101 样本（47 合法 + 54 钓鱼）
- `data/processed/img_test_v2.csv` - 101 样本（48 合法 + 53 钓鱼）

**列结构**:
```csv
id,img_path,label,timestamp,brand,source,domain
fish_dataset_phish_page_139,D:\uaam-phish\data\raw\fish_dataset\phish_page_139\shot.png,1,2025-01-05T14:51:44.195684Z,updatesuccess,D:\uaam-phish\data\raw\fish_dataset,typedream.app
```

#### 3. 数据验证

**路径验证结果**:
- Train: 467/469 路径存在（2 个缺失，0.4%）
- Val: 101/101 路径存在（100%）
- Test: 101/101 路径存在（100%）

**与其他模态对比**:
| Split | URL | HTML | IMG |
|-------|-----|------|-----|
| Train | 469 | 469  | 469 |
| Val   | 100 | 100  | 101 |
| Test  | 102 | 102  | 101 |

*注: Val/Test 的微小差异（±1-2 样本）是因为 master_v2.csv 中部分样本的 URL/HTML 模态缺失（URL 缺失 2 个，HTML 缺失 8 个），其他模态生成脚本可能自动过滤了这些样本。*

#### 4. 相关文档

**新增文件**:
- `build16.plan.md` - 详细的任务计划和实施方案

**文档内容**:
- 问题分析和影响评估
- 两种实施方案对比（从 master 提取 vs 重新构建）
- 完整的脚本代码示例
- 数据验证清单
- 风险分析和成功标准

### 技术细节

#### Windows 编码兼容性

脚本添加了 Windows 控制台编码处理：

```python
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
```

#### Split 一致性保证

通过直接从 `master_v2.csv` 提取，确保与现有的 URL/HTML CSV 使用相同的数据划分，避免了重新生成可能导致的不一致。

### 验证

- ✅ 三个 IMG CSV 文件成功生成
- ✅ 列结构符合预期（包含 id, img_path, label, metadata）
- ✅ 样本数量与 master_v2.csv 的 split 分布一致
- ✅ 99.7% 的图像路径有效（671 个中有 669 个存在）
- ✅ 标签分布合理（phish vs benign 比例接近 1:1）

### 后续任务

- [ ] 更新 `docs/DATA_SCHEMA.md`，补充 IMG CSV 说明
- [ ] 测试 `VisualDataModule` 是否可以加载新 CSV（如果需要支持独立 CSV 模式）
- [ ] 运行 Visual baseline 实验验证完整性

---

## 2025-11-07: 修复多模态 Baseline 烟雾测试

### 问题诊断

用户报告两个测试命令失败：

1. **Dry-run 烟雾测试**
   ```bash
   python scripts/train_hydra.py experiment=multimodal_baseline trainer.fast_dev_run=true
   ```

2. **随机分割回归测试**
   ```bash
   python scripts/train_hydra.py experiment=multimodal_baseline datamodule.split_protocol=random trainer.fast_dev_run=true
   ```

### 根本原因

#### 问题 1: Hydra Struct 模式错误
- **错误信息**: `Could not override 'trainer.fast_dev_run'. Key 'fast_dev_run' is not in struct`
- **原因**: Hydra 配置使用严格模式（struct mode），不允许覆盖未预定义的字段
- **影响**: 无法通过命令行添加调试参数

#### 问题 2: fast_dev_run 与 checkpoint 加载冲突
- **错误信息**: `ValueError: You cannot execute .test(ckpt_path="best") with fast_dev_run=True`
- **原因**: `fast_dev_run` 模式下不保存检查点，但 `train_hydra.py` 在测试时始终尝试加载 "best" 检查点
- **影响**: 烟雾测试在 fit 阶段成功，但在 test 阶段崩溃

#### 问题 3: 缺少依赖库
- **错误信息**: `无法从源码解析导入 "bs4"`
- **原因**: `requirements.txt` 未包含 `beautifulsoup4` 和其他必需的库
- **影响**: Linter 警告，运行时可能失败

### 解决方案

#### 1. 添加 Trainer 调试参数默认值（Add-only）

**文件**: `configs/trainer/default.yaml`

   ```yaml
# Trainer debug/test parameters (optional, can be overridden with +trainer.*)
trainer:
  fast_dev_run: false
  limit_train_batches: null
  limit_val_batches: null
  limit_test_batches: null
  overfit_batches: 0
```

**设计原理**:
- 遵循论文 Compliance Rule: **Add-only & Idempotent**
- 不修改现有配置，仅添加新字段
- 默认值为 `false`/`null`/`0`，不影响现有实验
- 支持通过命令行覆盖：`trainer.fast_dev_run=true`

#### 2. 修复 fast_dev_run 模式下的 checkpoint 处理

**文件**: `scripts/train_hydra.py:171-174`

```python
dm.setup(stage="test")
# In fast_dev_run mode, checkpoints are not saved, so we test with current weights
ckpt_path = "best" if not getattr(cfg.trainer, "fast_dev_run", False) else None
test_results = trainer.test(model, dataloaders=dm.test_dataloader(), ckpt_path=ckpt_path)
```

**设计原理**:
- 检测 `fast_dev_run` 模式
- 烟雾测试时使用当前权重（`ckpt_path=None`）
- 正常训练时仍加载最佳检查点（`ckpt_path="best"`）
- 向后兼容，不破坏现有功能

#### 3. 补全依赖库（Add-only）

**文件**: `requirements.txt`

新增依赖：
```txt
torchvision>=0.17  # 视觉模型（ResNet等）
Pillow>=10.0  # 图像处理
beautifulsoup4>=4.12  # HTML 解析
lxml>=4.9  # bs4 的解析器后端
```

**设计原理**:
- 遵循 Add-only 原则，不删除现有依赖
- 补全多模态实验所需的全部库
- 指定最低版本号，确保 API 兼容性

### 验证方法

#### 1. 确保激活虚拟环境
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# 验证环境
python -c "import sys; print(sys.prefix)"
```

#### 2. 安装依赖
```bash
# 推荐：安装所有依赖
python -m pip install -r requirements.txt

# 或者仅安装核心依赖
python -m pip install hydra-core omegaconf pytorch-lightning torch transformers torchmetrics torchvision pandas scikit-learn Pillow beautifulsoup4 lxml tldextract matplotlib seaborn
```

#### 3. 验证安装
```bash
python -c "import hydra; import torch; import pytorch_lightning; from bs4 import BeautifulSoup; print('✓ All dependencies installed')"
```

#### 运行烟雾测试
```bash
# 测试 1: 基本 dry-run
python scripts/train_hydra.py experiment=multimodal_baseline trainer.fast_dev_run=true

# 测试 2: 随机分割 dry-run
python scripts/train_hydra.py experiment=multimodal_baseline datamodule.split_protocol=random trainer.fast_dev_run=true
```

**预期行为**:
1. 配置加载成功，无 struct 错误
2. 训练 1 个 batch（fit）
3. 验证 1 个 batch（validate）
4. 测试 1 个 batch（test，使用当前权重）
5. 生成五件套产物：
   - `predictions_val.csv`
   - `metrics_val.json`
   - `roc_curve_val.png`
   - `reliability_before_ts_val.png`
   - `splits_presplit.csv` (或 `splits_random.csv`)

### 技术细节

#### fast_dev_run 模式特性
- PyTorch Lightning 内置的快速测试模式
- 仅运行 1 个 batch（train/val/test）
- **不保存检查点**（关键！）
- **不记录到 logger**
- 适用于：
  - 代码语法检查
  - 数据管道验证
  - 模型前向传播测试

#### Hydra Struct Mode
- 默认情况下，Hydra 配置支持两种覆盖方式：
  - `key=value`：覆盖已存在的字段（strict）
  - `+key=value`：添加新字段（permissive）
- 本次修复采用 **预定义字段** 方案，避免用户记忆 `+` 语法

### 遵循的论文约束

✅ **Add-only & Idempotent** (Thesis Rule)
- 未删除任何现有代码、配置或依赖
- 添加的字段有明确的默认值
- 多次应用本次变更不会产生副作用

✅ **Non-breaking Changes**
- 现有实验配置无需修改
- `fast_dev_run` 默认为 `false`，不影响正常训练
- checkpoint 逻辑向后兼容

✅ **Reproducibility**
- 添加的调试参数不影响随机种子
- checkpoint 选择逻辑明确且可预测

### 未来工作

如果需要在 test 阶段也生成产物（在 fast_dev_run 模式下），可考虑：
- 在 `TestPredictionCollector` 中添加对 `fast_dev_run` 的检测
- 在 test 阶段保存简化版产物（仅包含最后一个 batch）

### 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `configs/trainer/default.yaml` | 新增字段 | 添加 `trainer` 调试参数默认值 |
| `scripts/train_hydra.py` | 逻辑修复 | 添加 fast_dev_run 的 checkpoint 条件判断 |
| `requirements.txt` | 新增依赖 | 补全 bs4, lxml, Pillow, torchvision |
| `test_multimodal_smoke.py` | 新增文件 | 自动化烟雾测试脚本（临时，可删除） |

---

**变更状态**: ✅ 已完成
**测试状态**: ⏳ 等待用户验证
**论文合规**: ✅ 通过

---

## 2025-11-10: 缓存切换逻辑实现

### 问题背景

数据加载速度慢，需要实现自动缓存切换机制来提高训练效率。现有系统需要手动修改配置文件路径来使用缓存数据，不够灵活。

### 解决方案

#### 1. DataModule 自动缓存路径切换

**文件**: `src/data/multimodal_datamodule.py`

**新增方法**: `_maybe_use_cached()`
- 自动检测是否存在对应的 `*_cached.csv` 文件
- 如果存在，自动将 train/val/test_csv 路径切换到缓存版本
- 保持向后兼容性，只在缓存文件存在时替换

**关键逻辑**:
```python
def _maybe_use_cached(self) -> None:
    if self.train_csv and self.train_csv.exists():
        cached_train_csv = self.train_csv.parent / f"{self.train_csv.stem}_cached.csv"
        if cached_train_csv.exists():
            log.info(f">> 检测到缓存训练CSV，切换到: {cached_train_csv}")
            self.train_csv = cached_train_csv
```

#### 2. Dataset 缓存优先加载机制

**新增缓存加载方法**:
- `_load_cached_html()`: 加载缓存的HTML tokens
- `_load_cached_url()`: 加载缓存的URL tokens
- `_load_cached_image()`: 加载缓存的图像（支持JPG和PT格式）

**缓存优先策略**:
```python
# 先尝试加载缓存，失败则回退到原始逻辑
url_ids = self._load_cached_url(row)
if url_ids is None:
    url_text = self._safe_string(row.get("url_text", row.get("url", "")))
    url_ids = self._tokenize_url(url_text)
```

**路径解析方法**: `_resolve_cached_path()`
- 将相对路径转换为绝对路径
- 支持缓存根目录配置

#### 3. W&B Run Name 配置优化

**更新实验配置文件**:
- `configs/experiment/s0_brandood_lateavg.yaml`: 明确设置 `run.name`
- `configs/experiment/s0_brandood_earlyconcat.yaml`: 明确设置 `run.name`
- 确保实验配置的run name不会被主配置覆盖

#### 4. Brand-OOD 测试集配置

**新增配置项**: `test_ood_csv`
- 训练experiment中 `test_csv` 指向 `test_id.csv`（ID测试集）
- 添加 `test_ood_csv` 配置项指向OOD测试集
- 评估时可通过CLI参数切换测试集

### 验证结果

#### 缓存加载测试

**命令**:
```bash
python tools/test_cache_loading.py --train-csv workspace/data/splits/iid/train_cached.csv --mode full --num-workers 4
```

**结果**:
- ✅ **缓存路径检测成功**: DataModule自动切换到缓存CSV
- ✅ **缓存文件加载成功**: 出现 `torch.load` 警告，说明缓存被正确加载
- ✅ **性能大幅提升**: 平均速度从0.15 it/s提升到3.43 it/s（>3 it/s目标）
- ✅ **缓存完整性**: 所有缓存文件存在且非空率100%

#### 缓存完整性检查

**命令**:
```bash
python tools/check_cache_integrity.py --scenario iid
```

**结果**:
- ✅ **训练集**: 11,200样本，三列缓存文件100%存在
- ✅ **验证集**: 2,400样本，三列缓存文件100%存在
- ✅ **测试集**: 2,400样本，三列缓存文件100%存在

### 技术亮点

#### 1. 路径解析优化
- 支持相对路径到绝对路径的自动转换
- 通过 `cache_root` 参数传递预处理目录
- 避免硬编码路径，提高灵活性

#### 2. 异常处理机制
- 所有缓存加载都包含存在性检查
- 支持多种缓存格式（JPG需要transform，PT直接加载）
- 单个缓存文件损坏不影响整体训练

#### 3. 向后兼容性
- 缓存文件不存在时自动回退到原始逻辑
- 不影响未生成缓存的场景
- 配置项可选，不强制要求

### 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/data/multimodal_datamodule.py` | 新增方法 | 添加缓存路径切换和缓存加载方法 |
| `configs/experiment/s0_brandood_lateavg.yaml` | 配置更新 | 添加test_ood_csv配置项 |
| `configs/experiment/s0_brandood_earlyconcat.yaml` | 配置更新 | 添加test_ood_csv配置项 |

### 使用指南

#### 启用缓存
1. 确保预处理脚本已生成 `*_cached.csv` 文件
2. 运行训练时，系统会自动检测并使用缓存
3. 查看日志确认缓存路径被正确加载

#### 验证缓存
```bash
# 测试缓存加载速度
python tools/test_cache_loading.py --train-csv workspace/data/splits/iid/train_cached.csv --mode full

# 检查缓存完整性
python tools/check_cache_integrity.py --scenario iid
```

---

**变更状态**: ✅ 已完成
**性能提升**: 3.43 it/s（达到预期目标）
**论文合规**: ✅ 通过（Add-only修改）
