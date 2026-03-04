#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只跑原始变量相关性热图（不跑模型）
使用原始 t1/t2 变量，不使用交互特征
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================== 路径配置 ====================
DATA_PATH = r"D:\临时文件\文档\文献\DATA_LPA.xlsx"
OUTPUT_BASE_DIR = r"D:\临时文件\文档\文献\数据图表"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# 切换到 Agg 后端，避免 tkinter 警告
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Arial'

print("=" * 80)
print("Correlation Heatmap of Original Variables (t1/t2)")
print(f"Data Source: {DATA_PATH}")
print(f"Output Directory: {OUTPUT_BASE_DIR}")
print("=" * 80)

# ==================== 1. 加载原始数据 ====================
print("\n【Step 1】Loading raw data...")
df_raw = pd.read_excel(DATA_PATH, sheet_name="无空缺值数据")
print(f"✅ Data loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

# ==================== 2. 提取原始 t1/t2 变量（排除交互/统计特征） ====================
print("\n【Step 2】Extracting original t1/t2 variables...")

# 筛选出所有以 _t1 或 _t2 结尾的原始变量（排除 norm/angle/mean/max/min 等统计特征）
original_cols = []
for col in df_raw.columns:
    if col.endswith('_t1') or col.endswith('_t2'):
        # 排除包含 norm/angle/mean/max/min 的列，只保留原始 t1/t2
        if not any(kw in col for kw in ['norm', 'angle', 'mean', 'max', 'min', 'change', 'increasing', 'stable']):
            original_cols.append(col)

print(f"Found {len(original_cols)} original t1/t2 variables:")
for col in sorted(original_cols):
    print(f"  - {col}")

# 提取这些列做相关性分析
df_corr = df_raw[original_cols].copy()

# 处理缺失值（用中位数填充，保证相关性计算稳定）
df_corr = df_corr.fillna(df_corr.median())

# ==================== 3. 计算并绘制相关性热力图 ====================
print("\n【Step 3】Calculating and plotting correlation heatmap...")

# 计算相关系数矩阵
corr_matrix = df_corr.corr()

# 绘制热力图
plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    annot=False,        # 不显示具体数值（变量多，避免拥挤）
    cmap='coolwarm',    # 红蓝配色，区分正负相关
    center=0,           # 0 为中心，蓝色负相关，红色正相关
    linewidths=0.5,     # 格子间的分隔线
    vmin=-1, vmax=1,    # 颜色范围 [-1, 1]
    cbar=True
)

plt.title('Correlation Heatmap of Original t1/t2 Variables', fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()

# 保存图片
heatmap_path = os.path.join(OUTPUT_BASE_DIR, 'original_variables_correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close('all')

print(f"✅ Correlation heatmap saved to: {heatmap_path}")
print("\n✅ All done!")