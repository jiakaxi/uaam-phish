# DATA_README

## Sources
- Benign: Tranco / Alexa / Phishpedia benign subset
- Phishing: OpenPhish / Phishpedia phishing subset

## Layout
- `data/raw/`：原始下载
- `data/processed/`：`master.csv` + `train/val/test`（由脚本生成）

## master.csv schema（建议）
- id, url_text, html_path, img_path, domain, brand, label, split, source, timestamp

## Cleaning & Dedup
- URL 规范化：lowercase、strip、remove tracking params
- 去重：URL-level、HTML simhash/minhash、image pHash
- 过滤：失效链接/多重重定向/空白页

## Splitting
- Random / Temporal / Brand OOD（Group by domain/brand）
- 工具：`scripts/build_master_and_splits.py`

## QA Checklist
- [ ] Benign/Phish 采样比例
- [ ] 域名/品牌泄漏检测
- [ ] 缺失模态占比与分布
