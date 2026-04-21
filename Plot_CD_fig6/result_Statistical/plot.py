import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math

# ==========================================
# 1. 目录设置与全局绘图参数 (学术排版)
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.dirname(current_dir)
out_dir = os.path.join(data_dir, "Visualization_Results")
os.makedirs(out_dir, exist_ok=True)

# 强制使用新罗马字体 (Times New Roman) 和超大字号
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 50})

metrics = ["ACC", "ARI", "F1"]

print(f"📁 数据读取目录: {data_dir}")
print(f"📁 图表保存目录: {out_dir}\n" + "="*50)

# ==========================================
# 2. 创建 1x3 的横向画板
# ==========================================
# 宽度设为 28，高度设为 8，横向跨度拉大，避免三张图挤在一起
fig, axes = plt.subplots(1, 3, figsize=(28, 7))

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    search_pattern = os.path.join(data_dir, f"*{metric}*.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"⚠️ 未找到包含 {metric} 的 CSV 文件，跳过...")
        continue
        
    file_path = files[0]
    print(f"⏳ 正在处理 {metric} 指标...")
    
    df = pd.read_csv(file_path)
    
    # 3. 数据清洗与检验计算
    if 'dataset' in df.columns:
        df['dataset'] = df['dataset'].ffill()
    
    algo_cols = [c for c in df.columns if c.endswith(f'_{metric}')]
    df_mean = df.groupby('dataset')[algo_cols].mean()
    df_mean.columns = [c.replace(f'_{metric}', '') for c in df_mean.columns]
    
    ranks = df_mean.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()
    
    k = len(avg_ranks)
    N = len(df_mean)
    
    q_dict = {
        5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 
        11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391
    }
    q_alpha_0_05 = q_dict.get(k, 3.219) 
    CD = q_alpha_0_05 * np.sqrt((k * (k + 1)) / (6 * N))
    
    # ==========================================
    # 4. 开始在子图 ax 上绘制
    # ==========================================
    ax.plot([1, k], [0, 0], color='black', lw=2)
    ax.set_xlim(0, k+1)
    ax.set_ylim(-4.2, 2.2) # 调整上下留白
    ax.axis('off')
    
    # 画刻度
    for i in range(1, k+1):
        ax.plot([i, i], [0, 0.2], color='black')
        ax.text(i, 0.4, str(i), ha='center', fontsize=18)
        
    left_side = avg_ranks[:math.ceil(k/2)]
    right_side = avg_ranks[math.ceil(k/2):]
    
    # 绘制左侧算法
    for i, (algo, rank) in enumerate(left_side.items()):
        y = -0.6 - i*0.45
        ax.plot([rank, rank], [0, y], color='black')
        ax.plot([rank, 1], [y, y], color='black')
        
        is_opn = (algo == "OPNs")
        weight = 'bold' if is_opn else 'normal'
        color = '#000080' if is_opn else 'black' # 藏青色高亮
        
        # 算法名字号加大为 20
        ax.text(0.8, y, f"{algo} ({rank:.2f})", ha='right', va='center', 
                fontsize=20, fontweight=weight, color=color)

    # 绘制右侧算法
    for i, (algo, rank) in enumerate(right_side.items()):
        y = -0.6 - i*0.45
        ax.plot([rank, rank], [0, y], color='black')
        ax.plot([rank, k], [y, y], color='black')
        
        is_opn = (algo == "OPNs")
        weight = 'bold' if is_opn else 'normal'
        color = '#000080' if is_opn else 'black'
        
        ax.text(k+0.2, y, f"{algo} ({rank:.2f})", ha='left', va='center', 
                fontsize=20, fontweight=weight, color=color)

    # 绘制红色的 CD 临界线
    ax.plot([1, 1+CD], [1.4, 1.4], color='#8B0000', lw=3)
    ax.text(1+CD/2, 1.7, f"CD = {CD:.2f}", ha='center', color='#8B0000', fontsize=18, fontweight='bold')

    # 计算显著性红线
    groups = []
    sorted_ranks = avg_ranks.values
    for i in range(k):
        for j in range(k-1, i, -1):
            if sorted_ranks[j] - sorted_ranks[i] <= CD:
                groups.append((i, j))
                break

    final_groups = []
    for g in groups:
        is_subset = False
        for other in groups:
            if g != other and g[0] >= other[0] and g[1] <= other[1]:
                is_subset = True
                break
        if not is_subset:
            final_groups.append(g)

    # 画显著性红线 (深赤红)
    for idx_g, (start, end) in enumerate(final_groups):
        y = -0.15 - idx_g*0.15
        ax.plot([sorted_ranks[start], sorted_ranks[end]], [y, y], color='#8B0000', lw=3)

    # 子图标题 (加粗, 字母标注, 标题字号加大为 24)
    ax.set_title(f"({chr(97+idx)}) Critical Difference for {metric}", fontsize=24, pad=20, fontweight='bold')

# ==========================================
# 5. 统一排版与保存
# ==========================================
plt.tight_layout()
save_path_pdf = os.path.join(out_dir, "CD_Diagram_Combined_1x3.eps")
save_path_png = os.path.join(out_dir, "CD_Diagram_Combined_1x3.png")

plt.savefig(save_path_pdf, dpi=300, format='pdf', bbox_inches='tight')
plt.savefig(save_path_png, dpi=300, bbox_inches='tight')

print("🎉 1x3 横向合并大图已生成！请查看 Visualization_Results 文件夹。")
plt.close()
