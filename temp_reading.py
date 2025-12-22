import pickle
import os
import matplotlib.pyplot as plt
import networkx as nx

# 1. 配置路径和文件参数
data_path = "./data/train_20_100"  # 替换成你的实际数据根目录
i = 0  # 对应文件夹0（可修改）
# 定义要读取的四个.pkl文件名
file_names = [
    "distance_m.pkl",
    "attraction_params.pkl",
    "graph.pkl",
    "tabu_table.pkl"
]

# 2. 批量读取文件并存储数据
all_data = {}
for file_name in file_names:
    # 拼接完整文件路径
    file_path = os.path.join(data_path, str(i), file_name)
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"警告：文件 {file_path} 不存在，跳过该文件")
        continue
    # 读取文件
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    all_data[file_name] = data

# 3. 逐个查看数据的详细信息
for fname, data in all_data.items():
    print(f"\n========== {fname} 的数据信息 ==========")
    print(f"数据类型：{type(data)}")
    print(f"数据长度/形状（若适用）：{len(data) if hasattr(data, '__len__') else '无'}")
    print(f"数据内容（前部分预览）：")
    
    # 针对graph.pkl的特殊处理：展示nodes、edges并可视化
    if fname == "graph.pkl" and isinstance(data, nx.Graph):
        # 展示所有节点
        nodes = list(data.nodes())
        print(f"图的节点（nodes）：{nodes}")
        # 展示所有边
        edges = list(data.edges())
        print(f"图的边（edges）：{edges}")
        # 展示节点数和边数
        print(f"节点数量：{data.number_of_nodes()}")
        print(f"边数量：{data.number_of_edges()}")
        
        # # 图形可视化
        # print("\n正在绘制图形...")
        # plt.figure(figsize=(8, 6))  # 设置画布大小
        # # 绘制图（with_labels=True显示节点标签，node_size设置节点大小，font_size设置字体大小）
        # nx.draw(data, with_labels=True, node_size=500, font_size=10, node_color="lightblue")
        # plt.title(f"Graph from {fname}")  # 设置标题
        # plt.show()  # 显示图形
        print("\n正在保存图形到本地...")
        plt.figure(figsize=(8, 6))  # 设置画布大小
        nx.draw(data, with_labels=True, node_size=500, font_size=10, node_color="lightblue")
        plt.title(f"Graph from {fname}")  # 设置标题
        # 保存图片到当前目录（可修改路径，如"./images/graph.png"）
        plt.savefig(f"graph_visualization.png", dpi=300, bbox_inches="tight")
        plt.close()  # 关闭画布，释放内存
        print(f"图形已保存为：graph_visualization.png（在当前文件夹中查看）")
    
    # 普通数据类型的展示逻辑
    elif isinstance(data, (list, tuple)):
        print(data[:10] if len(data) > 10 else data)
    elif isinstance(data, dict):
        print(f"字典的键：{list(data.keys())}")
        print(f"字典的前3个值：{list(data.values())[:3] if len(data) > 3 else data.values()}")
    else:
        print(data) 