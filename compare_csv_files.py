#!/usr/bin/env python
"""Compare master_v2.csv and master_v2_backup.csv"""

import pandas as pd

# Read both files
df1 = pd.read_csv("data/processed/master_v2.csv")
df2 = pd.read_csv("data/processed/master_v2_backup.csv")

print("=" * 80)
print("CSV文件对比分析")
print("=" * 80)

print("\n1. 文件基本信息:")
print(f"   master_v2.csv: {len(df1):,} 行, {len(df1.columns)} 列")
print(f"   master_v2_backup.csv: {len(df2):,} 行, {len(df2.columns)} 列")
print(f"   行数差异: {len(df2) - len(df1):,} 行 (backup多 {len(df2) - len(df1)} 行)")

print("\n2. 列信息对比:")
cols1 = set(df1.columns)
cols2 = set(df2.columns)
print(f"   master_v2.csv 的列: {sorted(cols1)}")
print(f"   master_v2_backup.csv 的列: {sorted(cols2)}")
print(f"\n   master_v2.csv 独有的列: {cols1 - cols2}")
print(f"   master_v2_backup.csv 独有的列: {cols2 - cols1}")

print("\n3. 标签分布对比:")
print("   master_v2.csv:")
print(
    f"     正样本 (label=1): {(df1['label'] == 1).sum():,} ({((df1['label'] == 1).sum() / len(df1) * 100):.1f}%)"
)
print(
    f"     负样本 (label=0): {(df1['label'] == 0).sum():,} ({((df1['label'] == 0).sum() / len(df1) * 100):.1f}%)"
)
print("   master_v2_backup.csv:")
print(
    f"     正样本 (label=1): {(df2['label'] == 1).sum():,} ({((df2['label'] == 1).sum() / len(df2) * 100):.1f}%)"
)
print(
    f"     负样本 (label=0): {(df2['label'] == 0).sum():,} ({((df2['label'] == 0).sum() / len(df2) * 100):.1f}%)"
)

print("\n4. split列对比:")
print("   master_v2.csv split值分布:")
print(df1["split"].value_counts())
print("\n   master_v2_backup.csv split值分布:")
print(df2["split"].value_counts())

print("\n5. 数据源对比:")
print("   master_v2.csv source分布 (前10):")
print(df1["source"].value_counts().head(10))
print("\n   master_v2_backup.csv source分布 (前10):")
print(df2["source"].value_counts().head(10))

print("\n6. timestamp_original字段:")
if "timestamp_original" in df1.columns:
    non_null = df1["timestamp_original"].notna().sum()
    print(
        f"   master_v2.csv: 有该列, {non_null} 个非空值 ({non_null/len(df1)*100:.1f}%)"
    )
else:
    print("   master_v2.csv: 无该列")

if "timestamp_original" in df2.columns:
    non_null = df2["timestamp_original"].notna().sum()
    print(
        f"   master_v2_backup.csv: 有该列, {non_null} 个非空值 ({non_null/len(df2)*100:.1f}%)"
    )
else:
    print("   master_v2_backup.csv: 无该列")

print("\n7. 品牌信息对比:")
print(f"   master_v2.csv 唯一品牌数: {df1['brand'].nunique()}")
print(f"   master_v2_backup.csv 唯一品牌数: {df2['brand'].nunique()}")

print("\n8. 时间戳字段检查:")
print(
    f"   master_v2.csv timestamp非空: {df1['timestamp'].notna().sum()} ({df1['timestamp'].notna().sum()/len(df1)*100:.1f}%)"
)
print(
    f"   master_v2_backup.csv timestamp非空: {df2['timestamp'].notna().sum()} ({df2['timestamp'].notna().sum()/len(df2)*100:.1f}%)"
)

print("\n" + "=" * 80)
print("结论和建议")
print("=" * 80)
print(f"\n1. master_v2.csv 是当前使用的数据集（{len(df1):,} 行）")
print("   - 包含 timestamp_original 列")
print("   - 所有配置文件都指向此文件")
print(f"   - split列值: {df1['split'].unique().tolist()}")

print(f"\n2. master_v2_backup.csv 是备份文件（{len(df2):,} 行）")
print("   - 不包含 timestamp_original 列")
print(f"   - 多 {len(df2) - len(df1)} 行数据")
print(f"   - split列值: {df2['split'].unique().tolist()}")

print("\n3. S0实验应该使用: master_v2.csv")
print("   - 所有S0工具脚本都接受 --in 参数指定输入文件")
print("   - 配置文件默认使用 master_v2.csv")
print("   - 包含完整的时间戳信息（timestamp_original）")

print("\n" + "=" * 80)


