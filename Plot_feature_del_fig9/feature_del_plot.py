import os
import matplotlib.pyplot as plt
import numpy as np

# === 关键修改：设置全局字体为 Times New Roman ===
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# （可选）确保图中的某些特殊减号或符号也遵循类似的衬线风格
plt.rcParams['mathtext.fontset'] = 'stix' 

# 确保目标文件夹存在
os.makedirs('data_feature_del', exist_ok=True)

# 你的数据集特征与最优配对字典
data = {
    'DS6': (1.8, 6), 'DS7': (4.3, 7), 'DS8': (2.1, 7), 'DS9': (3.2, 7),
    'DS10': (2.5, 8), 'DS11': (2.9, 8), 'DS12': (2.1, 9), 'DS13': (6.3, 13),
    'DS14': (4.5, 13), 'DS15': (2.6, 13), 'DS16': (3.8, 13), 'DS17': (5.6, 13),
    'DS18': (14.9, 16), 'DS19': (4.8, 18), 'DS20': (2.7, 18), 'DS21': (7.7, 19),
    'DS22': (3.2, 60), 'DS23': (5.9, 64)
}

labels = list(data.keys())
opns_pairs = [v[0] for v in data.values()]
original_features = [v[1] for v in data.values()]
x = np.arange(len(labels))

# ---------------------------------------------------------
# 方案一：横向双柱状图 (Horizontal Bar Chart)
# ---------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(10, 14), dpi=300)
height = 0.35

rects1 = ax1.barh(x + height/2, original_features, height, label='Original Features', color='#b3b3b3', edgecolor='black')
rects2 = ax1.barh(x - height/2, opns_pairs, height, label='Optimal OPNs Pairs', color='#4a7bc7', edgecolor='black')

ax1.set_xlabel('Number of Features', fontsize=22, fontweight='bold')
ax1.set_ylabel('Datasets', fontsize=22, fontweight='bold')
ax1.set_yticks(x)
ax1.set_yticklabels(labels, fontsize=18)
ax1.tick_params(axis='x', labelsize=18)
ax1.legend(fontsize=18, loc='lower right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', linestyle='--', alpha=0.7)

# 为横向柱添加标签
for rect in rects1:
    w = rect.get_width()
    ax1.annotate(f'{int(w)}', xy=(w, rect.get_y() + rect.get_height() / 2),
                 xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=16)
for rect in rects2:
    w = rect.get_width()
    ax1.annotate(f'{w:.1f}', xy=(w, rect.get_y() + rect.get_height() / 2),
                 xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=16)

plt.tight_layout()
plt.savefig('data_feature_del/Figure_11_Horizontal.pdf', format='pdf')
plt.close()

# ---------------------------------------------------------
# 方案二：折线面积对比图 (Line & Area Chart) 
# ---------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(16, 7), dpi=300)

ax2.plot(x, original_features, marker='s', markersize=10, linewidth=3, linestyle='--', color='#9e9e9e', label='Original Features')
ax2.plot(x, opns_pairs, marker='o', markersize=10, linewidth=3, color='#4a7bc7', label='Optimal OPNs Pairs')
ax2.fill_between(x, opns_pairs, original_features, color='#b3b3b3', alpha=0.2, label='Dimensionality Reduction')

ax2.set_ylabel('Number of Features', fontsize=20, fontweight='bold')
ax2.set_xlabel('Datasets', fontsize=20, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=18)
ax2.tick_params(axis='y', labelsize=18)
ax2.legend(fontsize=18, loc='upper left')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='both', linestyle='--', alpha=0.5)

# 给折线加上下交错标签防止重叠
for i, (orig, opns) in enumerate(zip(original_features, opns_pairs)):
    ax2.annotate(f'{int(orig)}', xy=(i, orig), xytext=(0, 10), textcoords="offset points", ha='center', va='bottom', fontsize=14, color='#5c5c5c')
    ax2.annotate(f'{opns:.1f}', xy=(i, opns), xytext=(0, -20), textcoords="offset points", ha='center', va='top', fontsize=14, color='#1d428a')

plt.tight_layout()
plt.savefig('data_feature_del/Figure_11_LineArea.pdf', format='pdf')
plt.close()

# ---------------------------------------------------------
# 方案三：超大字体的经典垂直柱状图 (Vertical Bar Large Font)
# ---------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(20, 8), dpi=300)
width = 0.35

rects1_v = ax3.bar(x - width/2, opns_pairs, width, label='Optimal OPNs Pairs', color='#4a7bc7', edgecolor='black')
rects2_v = ax3.bar(x + width/2, original_features, width, label='Original Features', color='#b3b3b3', edgecolor='black')

ax3.set_ylabel('Number of Features', fontsize=24, fontweight='bold')
ax3.set_xlabel('Datasets', fontsize=24, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=20)
ax3.tick_params(axis='y', labelsize=20)
ax3.legend(fontsize=20, loc='upper left')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# 数值标签：为了防止重叠，蓝色小柱子的标签改为了纵向竖排 (rotation=90)
for rect in rects1_v:
    h = rect.get_height()
    ax3.annotate(f'{h:.1f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=16, rotation=90)
for rect in rects2_v:
    h = rect.get_height()
    ax3.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h),
                 xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=16)

ax3.set_ylim(0, max(original_features) * 1.25)
plt.tight_layout()
plt.savefig('data_feature_del/Figure_11_Vertical_Large.pdf', format='pdf')
plt.close()

print("三种排版优化方案均已生成，存为矢量图格式PDF，所有字体均为 Times New Roman。")
