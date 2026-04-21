import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====================================================================
# 0. 设定文件夹和文件路径
# ====================================================================
folder_name = 'data_stability'  # 您的目标文件夹名称

# 如果文件夹不存在，可以自动创建一个（以防万一）
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"已自动创建文件夹: {folder_name}")

input_csv = os.path.join(folder_name, 'OPNs-Kmeans - stability.csv')
output_png = os.path.join(folder_name, 'ACC_Iterations_Stability_Comparison_LargeFont.pdf')

# 1. 加载数据并预处理
try:
    df = pd.read_csv(input_csv)
    print(f"成功读取数据文件: {input_csv}")
except FileNotFoundError:
    print(f"❌ 找不到文件 {input_csv}！")
    print(f"请确保 'OPNs-Kmeans - stability.csv' 文件已经放在了 '{folder_name}' 文件夹下。")
    exit()

df['数据集'] = df['数据集'].fillna(method='ffill') # 填补因合并单元格产生的NaN

# 清理并确保数据类型为数值型
df['平均迭代次数'] = pd.to_numeric(df['平均迭代次数'], errors='coerce')
df['迭代_Std'] = pd.to_numeric(df['迭代_Std'], errors='coerce')
df['ACC平均'] = pd.to_numeric(df['ACC平均'], errors='coerce')
df['ACC_Std'] = pd.to_numeric(df['ACC_Std'], errors='coerce')

# 2. 挑选最具代表性的 5 个数据集
selected_datasets = ['iris', 'weather', 'segment', 'sonar', 'heart']
df_filtered = df[df['数据集'].isin(selected_datasets)]
algorithms = df['算法'].dropna().unique()

# ====================================================================
# 全局图表样式设置 (字体极度放大，完全适合双栏/单栏顶级期刊排版)
# ====================================================================
plt.rcParams.update({
    'font.size': 26,                          # 全局基准字体
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],        # 强制使用新罗马字体
    'axes.labelsize': 30,                     # X/Y 轴标签字体
    'axes.titlesize': 34,                     # 子图标题字体
    'xtick.labelsize': 28,                    # 刻度数字体
    'ytick.labelsize': 30,
    'legend.fontsize': 24,                    # 图例字体
    'legend.title_fontsize': 26,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

# 创建 1x2 并排子图，尺寸拉宽以容纳大字体
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 11))

# 为 11 种算法分配颜色，保证两边子图颜色一一对应
colors = [
    '#AEC7E8',  # 1. 柔和蓝 (Soft Blue)
    '#FFBB78',  # 2. 柔和橙 (Soft Orange)
    '#98DF8A',  # 3. 柔和绿 (Soft Green)
    "#C5DF36",  # 3. 柔和绿 (Soft Green)
    '#9EDAE5',  # 4. 柔和青 (Soft Cyan)
    '#C5B0D5',  # 5. 柔和紫 (Soft Purple)
    '#C49C94',  # 6. 柔和棕 (Soft Brown)
    '#F7B6D2',  # 7. 柔和粉 (Soft Pink)
    '#C7C7C7',  # 8. 柔和灰 (Soft Gray)
    '#DBDB8D',  # 9. 柔和橄榄绿 (Soft Olive)
    '#CC0000'   # 10. 醒目深红/学术红 (Emphasized Red) - 强对比强调色
]
color_map = {algo: colors[i % len(colors)] for i, algo in enumerate(algorithms)}

# ====================================================================
# 子图 (a)：迭代次数的均值与标准差
# ====================================================================
# 过滤掉没有迭代次数数据的算法（如部分深度学习算法）
iter_algos = df_filtered.dropna(subset=['平均迭代次数'])['算法'].unique()
df_iter = df_filtered[df_filtered['算法'].isin(iter_algos)]

mean_iter = df_iter.pivot(index='数据集', columns='算法', values='平均迭代次数').reindex(index=selected_datasets, columns=iter_algos)
std_iter = df_iter.pivot(index='数据集', columns='算法', values='迭代_Std').reindex(index=selected_datasets, columns=iter_algos)
iter_colors = [color_map[algo] for algo in iter_algos]

mean_iter.plot(kind='bar', yerr=std_iter, ax=ax1, capsize=6, width=0.85, color=iter_colors,
               edgecolor='black', linewidth=1, error_kw={'elinewidth': 2, 'alpha': 0.8})

ax1.set_title('(a) Convergence Stability: Iterations Comparison', pad=20, fontweight='bold')
ax1.set_ylabel('Number of Iterations', fontweight='bold')
ax1.set_xlabel('Datasets', fontweight='bold')
ax1.tick_params(axis='x', rotation=0)

# ====================================================================
# 子图 (b)：ACC准确率的均值与标准差
# ====================================================================
mean_acc = df_filtered.pivot(index='数据集', columns='算法', values='ACC平均').reindex(index=selected_datasets, columns=algorithms)
std_acc = df_filtered.pivot(index='数据集', columns='算法', values='ACC_Std').reindex(index=selected_datasets, columns=algorithms)

mean_acc.plot(kind='bar', yerr=std_acc, ax=ax2, capsize=6, width=0.85, color=[color_map[a] for a in algorithms],
              edgecolor='black', linewidth=1, error_kw={'elinewidth': 2, 'alpha': 0.8})

ax2.set_title('(b) Accuracy Robustness: ACC Comparison', pad=20, fontweight='bold')
ax2.set_ylabel('Clustering Accuracy (ACC)', fontweight='bold')
ax2.set_xlabel('Datasets', fontweight='bold')
ax2.tick_params(axis='x', rotation=0)

# 将Y轴上限设置大一点以防图例挡住柱子
ax2.set_ylim(0, 1.3)
ax2.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

# 将图例移出右侧绘图区以免遮挡数据
ax2.legend(title='Algorithms', loc='upper left', bbox_to_anchor=(1.02, 1.0))

# 同步清除左侧图例，只保留右侧的全局图例
ax1.get_legend().remove()

# ====================================================================
# 保存高质量图片到目标文件夹
# ====================================================================
plt.tight_layout()
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"🎉 成功生成高清大字体对比图，并已保存至: {output_png}")
plt.show()