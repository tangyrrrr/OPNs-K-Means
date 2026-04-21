import matplotlib.pyplot as plt
import numpy as np
import os

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# 严格指定并创建保存目录
output_dir = 'data_time_cost'
os.makedirs(output_dir, exist_ok=True)

# 替换 DS3 为 DS22，按维度从小到大排列
datasets = ['DS1 (d=4)', 'DS10 (d=8)', 'DS17 (d=13)', 'DS22 (d=60)']

# 完全按照用户提供的最新准确数据提取
data = {
    'ARI': {
        'EDK': [0.6769, 0.1127, 0.1009, 0.0128],
        'MDK': [0.6888, 0.1074, 0.0904, 0.0055],
        'CDK': [0.6804, 0.1281, 0.1045, 0.0218],
        'TDK': [0.6134, 0.1220 , 0.1076 , 0.0140],
        'KKM(p)': [0.6673, 0.1524, 0.1041, 0.0040],
        'KKM(l)': [0.6078, 0.1252, 0.0963, -0.0003],
        'KKM(r)': [0.6191, 0.1211, 0.0988, 0.0148],
        'TDEC': [0.6574, 0.1522, 0.0627, 0.0832],
        'ZEUS': [0.8515, 0.0167, 0.0346, 0.0105],
        'IDC': [0.8862, 0.2902, 0.3707, 0.5311],
        'OPNs(a)': [0.7886, 0.1008, np.nan, np.nan],
        'OPNs(b)': [0.8929, 0.2489, 0.3164, 0.2839],
    },
    'Time (s)': {
        'EDK': [0.0021, 0.1026, 0.8624, 0.0756],
        'MDK': [0.1616, 0.8648, 10.4227, 0.1820],
        'CDK': [0.1503, 1.3207, 11.6948, 0.1889],
        'TDK': [0.0821, 0.4332, 3.6126 , 0.0521],
        'KKM(p)': [0.0292, 0.9191, 3.8841, 0.0308],
        'KKM(l)': [0.0089, 0.3266, 1.7099, 0.0244],
        'KKM(r)': [0.0446, 0.2978, 3.9486, 0.0836],
        'TDEC': [0.3207, 0.4898, 24.2875, 0.2784],
        'ZEUS': [0.2209, 0.5684, 0.6010, 0.0217],
        'IDC': [11.5727, 24.4028, 33.3324, 12.4177],
        'OPNs(a)': [2.2807, 574.7033, np.nan, np.nan],
        'OPNs(b)': [0.9996, 3.2159, 0.2292, 20.4452],
    }
}

algorithms = list(data['ARI'].keys())

# 设置颜色，OPNs 使用醒目的暖色调
colors = plt.cm.tab10(np.linspace(0, 1, 10))
color_dict = {alg: colors[i] for i, alg in enumerate([a for a in algorithms if not a.startswith('OPNs')])}
color_dict['OPNs(a)'] = '#E57C23' 
color_dict['OPNs(b)'] = '#D83F31' 

# === 放大画布尺寸 ===
fig, axes = plt.subplots(1, 4, figsize=(48, 12))
axes = axes.flatten()

# === 放大所有字体字号 ===
TITLE_SIZE = 40
LABEL_SIZE = 40
TICK_SIZE = 38
LEGEND_SIZE = 40

for idx, ds_name in enumerate(datasets):
    ax = axes[idx]
    for alg in algorithms:
        x_val, y_val = data['ARI'][alg][idx], data['Time (s)'][alg][idx]
        if np.isnan(x_val) or np.isnan(y_val): continue
            
        if alg.startswith('OPNs'):
            ax.scatter(x_val, y_val, marker='*', s=2500, c=color_dict[alg], 
                       edgecolors='black', linewidth=3.0, label=alg, zorder=5)
        else:
            ax.scatter(x_val, y_val, marker='o', s=800, c=[color_dict[alg]], 
                       alpha=0.7, edgecolors='white', linewidth=1.5, label=alg, zorder=3)
            
    ax.set_title(f'{ds_name} - Performance vs Cost', fontsize=TITLE_SIZE, fontweight='bold', pad=25)
    ax.set_xlabel('ARI', fontsize=LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontsize=LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=0)

handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ordered_labels = [l for l in by_label.keys() if l.startswith('OPNs')] + \
                 [l for l in by_label.keys() if not l.startswith('OPNs')]
ordered_handles = [by_label[l] for l in ordered_labels]

# 调整图表边距，留出更充足的底部空间给单行图例
plt.subplots_adjust(left=0.04, right=0.99, top=0.90, bottom=0.25, wspace=0.35)

# 【核心修改区】设置 ncol=12 强制一行，下调 y 轴坐标 (-0.05) 拉开距离，并适当缩减列间距以防超出边缘
fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=12, 
           fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, -0.05), markerscale=1.5,
           columnspacing=1.0, handletextpad=0.4)

output_path = os.path.join(output_dir, 'scatter_ari_vs_time_1x4_super_large.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)

print(f"再次优化的散点图已成功保存至: {output_path}")

plt.show()
