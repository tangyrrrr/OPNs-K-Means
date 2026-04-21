import os
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches

# 自动创建 inter_pairs 文件夹
output_dir = 'inter_pairs'
os.makedirs(output_dir, exist_ok=True)

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

# 数据集与特征定义
datasets = {
    'Heart (DS13)': {
        'pairs': [(11, 12), (2, 11), (2, 10), (10, 11), (2, 8), (8, 10)],
        'labels': {2: 'Chest Pain\n(cp)', 8: 'Exercise Angina\n(exang)',
                   10: 'ST Slope\n(slope)', 11: 'Major Vessels\n(ca)', 12: 'Thalassemia\n(thal)'}
    },
    'Cleve (DS14)': {
        'pairs': [(5, 12), (9, 12), (11, 12), (1, 5)],
        'labels': {1: 'Sex', 5: 'Fasting Blood Sugar\n(fbs)', 9: 'ST Depression\n(oldpeak)',
                   11: 'Major Vessels\n(ca)', 12: 'Thalassemia\n(thal)'}
    },
    'Pima (DS10)': {
        'pairs': [(1, 5), (5, 7), (0, 7)],
        'labels': {0: 'Pregnancies', 1: 'Glucose', 5: 'BMI', 7: 'Age'}
    },
    'Wine (DS16)': {
        'pairs': [(6, 10), (0, 2), (0, 7), (7, 10)],
        'labels': {0: 'Alcohol', 2: 'Ash', 6: 'Flavanoids',
                   7: 'Nonflavanoid phenols', 10: 'Color Hue'}
    }
}

# 增加整个画布的宽度，适应 1x4 排版且保证每个子图自身更宽
fig, axes = plt.subplots(1, 4, figsize=(48, 14))

colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#D199FF', '#FF99CC', '#99FFFF', '#FFFF99']

# 【关键修改】：加入 enumerate(..., start=0) 来获取当前是第几个子图
for idx, (ax, (title, data)) in enumerate(zip(axes, datasets.items())):
    G = nx.Graph()
    G.add_edges_from(data['pairs'])
    
    # 将 k 设置为 1.0 适当缩短网络节点之间的引线距离，使网络图显得更紧凑
    pos = nx.spring_layout(G, seed=42, k=1.0)
    node_colors = [colors[list(data['labels'].keys()).index(node) % len(colors)] for node in G.nodes()]
    
    # 设置节点大小和边界
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                           node_size=6000, edgecolors='gray', linewidths=3.0)
    
    # 画边
    nx.draw_networkx_edges(G, pos, ax=ax, width=4.0, alpha=0.7, edge_color='gray')
    
    # 画节点标签，字体大且粗
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=32, font_weight='bold', font_color='black')
    
    # 准备图例项目
    legend_handles = []
    for i, (node_id, label_text) in enumerate(data['labels'].items()):
        color = colors[i % len(colors)]
        label_text_clean = label_text.replace('\n', ' ')
        patch = mpatches.Patch(color=color, label=f"{node_id}: {label_text_clean}")
        legend_handles.append(patch)
    
    # 【关键修改】：判断是否为第一个子图 (idx == 0)
    if idx == 0:
        # 第一个子图：放置在图内的【右下角】 (0.98, 0.02)
        leg = ax.legend(handles=legend_handles, loc='lower right',
                        bbox_to_anchor=(0.98, 0.02), frameon=True, shadow=True, ncol=1,
                        prop={'weight': 'bold', 'size': 24})
    else:
        # 其他子图：放置在图内的【左上角】 (0.02, 0.98)
        leg = ax.legend(handles=legend_handles, loc='upper left',
                        bbox_to_anchor=(0.02, 0.98), frameon=True, shadow=True, ncol=1,
                        prop={'weight': 'bold', 'size': 24})
                        
    leg.set_title("Features", prop={'weight': 'bold', 'size': 28})
    
    # 因为图例移到了图内部 (内部锚点)，不再往外撑开，所以 pad 调回正常的 30 即可防重叠标题
    ax.set_title(title, fontsize=42, fontweight='bold', pad=30)
    
    ax.axis('off')
    
    # 为绘图区添加四周额外的边界填充余量 (Margins)
    # y=0.35 意味着把网络图向上和向下各压缩留出 35% 空间，以腾出角落位置让给图例
    ax.margins(x=0.2, y=0.35)

# 精准调控图与图之间的间距 (wspace) 和外侧边框的距离
plt.subplots_adjust(wspace=0.15, top=0.90, bottom=0.05, left=0.02, right=0.98) 

output_path = os.path.join(output_dir, 'optimal_feature_pairing_custom_legend.pdf')
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)

print(f"文件成功保存至: {output_path}")

plt.show()
