import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def plot_training_curves(log_dir):
    """
    从 TensorBoard 日志读取数据并绘图
    
    Args:
        log_dir: 日志目录，例如 "./logs/20240119-101200"
    """
    # 找到 events 文件
    event_file = None
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file = os.path.join(root, file)
                break
        if event_file:
            break
    
    if not event_file:
        print(f"未找到 TensorBoard 日志文件在 {log_dir}")
        return
    
    # 读取日志
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()
    
    # 提取数据
    tags = ea.Tags()['scalars']
    print(f"可用的指标: {tags}")
    
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        data[tag] = pd.DataFrame([
            {'step': e.step, 'epoch': e.step, 'value': e.value} 
            for e in events
        ])
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 平均回合奖励
    if 'hp/avg_ep_reward' in data:
        ax = axes[0, 0]
        df = data['hp/avg_ep_reward']
        ax.plot(df['epoch'], df['value'], linewidth=2, color='#2E86AB')
        ax.set_title('Average Episode Reward', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # 2. Actor Loss
    if 'loss/loss_actor' in data:
        ax = axes[0, 1]
        df = data['loss/loss_actor']
        ax.plot(df['epoch'], df['value'], linewidth=2, color='#A23B72')
        ax.set_title('Actor Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # 3. Critic Loss
    if 'loss/loss_critic' in data:
        ax = axes[1, 0]
        df = data['loss/loss_critic']
        ax.plot(df['epoch'], df['value'], linewidth=2, color='#F18F01')
        ax.set_title('Critic Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    # 4. Entropy Loss
    if 'loss/loss_entropy' in data:
        ax = axes[1, 1]
        df = data['loss/loss_entropy']
        ax.plot(df['epoch'], df['value'], linewidth=2, color='#6A994E')
        ax.set_title('Entropy Loss', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = os.path.join(log_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {save_path}")
    
    plt.show()
    
    return data


# 使用示例
if __name__ == "__main__":
    # 替换为你的日志目录
    log_dir = "./logs/20260205-113259"  # 或者你的实际日志路径
    data = plot_training_curves(log_dir)
