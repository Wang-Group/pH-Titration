import matplotlib.pyplot as plt
import numpy as np

def plot_and_save(success, overshoot, avg_steps, models,
                  font_size=12, filename='model_comparison.svg'):
    """
    绘制双纵轴柱状图并保存为矢量图。
    
    参数:
    - success: list of float, Success Rate (%)
    - overshoot: list of float, Overshoot Incidence (%)
    - avg_steps: list of float, Average Steps
    - models: list of str, 模型名称
    - font_size: int, 全局字体大小
    - filename: str, 输出文件名（支持 .svg, .pdf 等矢量格式）
    """
    # 配置字体
    plt.rcParams['font.size'] = font_size
    plt.rcParams['font.family'] = 'Arial'
    
    # 坐标与宽度
    x = np.arange(len(models))
    width = 0.25
    colors = ['#2ca02c', '#1f77b4', '#17becf']
    
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax2 = ax1.twinx()
    
    # 绘制柱状图
    bars1 = ax1.bar(x - width, success, width,
                    label='Success Rate (%)',
                    color=colors[0], edgecolor=colors[0], linewidth=1.5)
    bars2 = ax1.bar(x, overshoot, width,
                    label='Overshoot Incidence (%)',
                    color=colors[1], edgecolor=colors[1], linewidth=1.5)
    bars3 = ax2.bar(x + width, avg_steps, width,
                    label='Avg Steps',
                    color=colors[2], edgecolor=colors[2], linewidth=1.5)
    
    # 坐标轴标签与刻度
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Percentage (%)')
    ax2.set_ylabel('Average Steps')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha='right')
    
    # 移除网格线
    ax1.grid(False)
    ax2.grid(False)
    
    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')
    
    plt.title('Model Comparison Across Three Metrics')
    plt.tight_layout()
    
    # 保存矢量图
    fig.savefig(filename, format=filename.split('.')[-1])
    plt.close(fig)
    print(f"Saved vector chart to {filename}")

# 示例调用
if __name__ == '__main__':
    models = ['Bayesian Rule-Based', 'Imitation Learning', 'Reinforcement Learning']
    success = [94.23, 93.77, 94.27]
    overshoot = [41.84, 34.41, 30.55]
    avg_steps = [12.73, 10.22, 10.21]
    # 调用时可以调整 font_size 和 filename
    plot_and_save(success, overshoot, avg_steps, models,
                  font_size=18, filename='model_comparison.svg')
    plt.show()  # 如果需要在脚本中直接展示图形