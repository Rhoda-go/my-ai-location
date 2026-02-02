import pickle
import numpy as np
import networkx as nx
import pandas as pd
from scipy.spatial.distance import cdist
import os

def load_real_data(node_file, distance_file, output_path):
 
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    nodes_df = pd.read_csv(node_file)
    
    # 检查必要列是否存在
    required_columns = ['fence_id', 'center_lon', 'center_lat', 'alpha', 'beta', 'resident_population_2km']
    for col in required_columns:
        if col not in nodes_df.columns:
            raise ValueError(f"CSV文件中缺少必要列: {col}")
    
    # 获取节点数量
    n = len(nodes_df)
    print(f"找到 {n} 个节点")
    
    # 2. 读取距离矩阵数据
    print(f"正在读取距离矩阵: {distance_file}")
    dist_df = pd.read_csv(distance_file)
    
    # 确保距离矩阵的fence_id与节点数据一致
    # 首先获取fence_id的顺序（按照节点数据的顺序）
    fence_ids = nodes_df['fence_id'].values
    
    # 检查距离矩阵是否包含所有fence_id
    missing_ids = set(fence_ids) - set(dist_df['fence_id'].unique())
    if missing_ids:
        raise ValueError(f"距离矩阵中缺少以下fence_id: {missing_ids}")
    
    # 3. 准备数据
    # 节点位置 (经纬度)
    nodes_pos = nodes_df[['center_lon', 'center_lat']].values
    
    # 人口数据
    city_pop = nodes_df['resident_population_2km'].values
    
    # alpha和beta参数
    alpha = nodes_df['alpha'].values
    beta = nodes_df['beta'].values
    
    # 4. 构建距离矩阵
    # 方法1: 如果距离矩阵已经是成对距离格式
    if 'fence_id_x' in dist_df.columns and 'fence_id_y' in dist_df.columns and 'distance_km' in dist_df.columns:
        print("检测到成对距离格式，构建距离矩阵...")
        # 创建空距离矩阵
        distance_m = np.zeros((n, n))
        
        # 创建fence_id到索引的映射
        id_to_idx = {fid: idx for idx, fid in enumerate(fence_ids)}
        
        # 填充距离矩阵
        for _, row in dist_df.iterrows():
            idx_i = id_to_idx[row['fence_id_x']]
            idx_j = id_to_idx[row['fence_id_y']]
            distance_m[idx_i, idx_j] = row['distance_km']
            distance_m[idx_j, idx_i] = row['distance_km']  # 对称矩阵
        
    # 方法2: 如果距离矩阵已经是n×n的矩阵格式（行和列都是fence_id）
    else:
        print("检测到矩阵格式的距离数据...")
        # 确保距离矩阵的列名是fence_id
        dist_df = dist_df.set_index('fence_id')
        
        # 重新索引以确保顺序与nodes_df一致
        dist_df = dist_df.reindex(index=fence_ids, columns=fence_ids)
        
        # 转换为numpy数组
        distance_m = dist_df.values.astype(float)
    
    print(f"距离矩阵形状: {distance_m.shape}")
    
    # 5. 构建图
    print("正在构建图...")
    G = nx.Graph()
    
    # 添加节点
    for i in range(n):
        G.add_node(i, 
                   pos=nodes_pos[i],
                   pop=city_pop[i],
                   alpha=alpha[i],
                   beta=beta[i])
    
    # 添加边（完全图）
    # 注意：原代码是Gabriel图，但根据你的要求，这里构建完全图
    for i in range(n):
        for j in range(i+1, n):
            # 添加边，权重为距离
            G.add_edge(i, j, length=distance_m[i, j])
    
    print(f"图构建完成，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")
    
    # 6. 计算tabu_table（使用原始代码的逻辑）
    print("正在计算tabu_table...")
    tabu_table = np.ones((n, n), dtype=int)
    
    for k in range(n):
        for i in range(n):
            if i == k:
                continue
            d_ki = distance_m[k][i]
            
            condition1 = alpha[k] > alpha[i] * np.exp(beta[k] * d_ki)
            condition2 = beta[k] <= beta[i]
            
            if condition1 and condition2:
                tabu_table[k][i] = 0
    
    # 7. 准备attraction_params
    attraction_params = {
        "alpha": alpha,
        "beta": beta,
        "node_indices": np.arange(n)
    }
    
    # 8. 保存所有数据
    print(f"正在保存数据到: {output_path}")
    
    # 保存tabu_table
    with open(f"{output_path}/tabu_table.pkl", "wb") as f:
        pickle.dump(tabu_table, f)
    
    # 保存attraction_params
    with open(f"{output_path}/attraction_params.pkl", "wb") as f:
        pickle.dump(attraction_params, f)
    
    # 保存图
    with open(f"{output_path}/graph.pkl", "wb") as f:
        pickle.dump(G, f)
    
    # 保存距离矩阵
    with open(f"{output_path}/distance_m.pkl", "wb") as f:
        pickle.dump(distance_m, f)
    
    # 9. 返回结果（可选）
    return {
        'city_pop': city_pop,
        'distance_m': distance_m,
        'nodes_pos': nodes_pos,
        'alpha': alpha,
        'beta': beta,
        'tabu_table': tabu_table,
        'graph': G,
        'attraction_params': attraction_params
    }

def batch_process_real_data(input_dir, output_base_path, data_name):
    """
    批量处理真实数据（如果有多个数据集）
    
    参数:
    input_dir: 输入数据目录，包含CSV文件
    output_base_path: 输出基础路径
    data_name: 数据集的名称（用于创建输出目录）
    """
    
    # 构建文件路径
    node_file = f"{input_dir}/selected_fences_summary.csv"
    distance_file = f"{input_dir}/selected_fence_pairwise_dist_km.csv"
    output_path = f"{output_base_path}/{data_name}"
    
    # 检查文件是否存在
    if not os.path.exists(node_file):
        print(f"警告: 节点文件不存在: {node_file}")
        return
    
    if not os.path.exists(distance_file):
        print(f"警告: 距离文件不存在: {distance_file}")
        return
    
    # 处理数据
    try:
        result = load_real_data(node_file, distance_file, output_path)
        print(f"数据处理完成，保存到: {output_path}")
        
        # 打印一些统计信息
        n = len(result['city_pop'])
        print(f"数据集 '{data_name}' 统计:")
        print(f"  节点数量: {n}")
        print(f"  平均人口: {result['city_pop'].mean():.2f}")
        print(f"  平均alpha: {result['alpha'].mean():.3f}")
        print(f"  平均beta: {result['beta'].mean():.3f}")
        print(f"  tabu_table中0的比例: {(result['tabu_table'].size - result['tabu_table'].sum()) / result['tabu_table'].size:.3f}")
        
    except Exception as e:
        print(f"处理数据时出错: {e}")

if __name__ == "__main__":
    # 示例用法
    # 1. 处理单个真实数据集
    # load_real_data(
    #     node_file="./data/real/selected_fences_summary.csv",
    #     distance_file="./data/real/selected_fence_pairwise_dist_km.csv",
    #     output_path="./data/real_processed"
    # )
    
    # 2. 批量处理多个真实数据集（如果有多组数据）
    # 假设目录结构:
    # real_data/
    #   ├── dataset1/
    #   │   ├── selected_fences_summary.csv
    #   │   └── selected_fence_pairwise_dist_km.csv
    #   └── dataset2/
    #       ├── selected_fences_summary.csv
    #       └── selected_fence_pairwise_dist_km.csv
    
    # base_input_dir = "./case_data"
    # base_output_path = "./casedata"
    
    # # 列出所有数据集目录
    # if os.path.exists(base_input_dir):
    #     for dataset_name in os.listdir(base_input_dir):
    #         dataset_path = os.path.join(base_input_dir, dataset_name)
    #         if os.path.isdir(dataset_path):
    #             print(f"\n处理数据集: {dataset_name}")
    #             batch_process_real_data(dataset_path, base_output_path, f"real_{dataset_name}")
    # else:
    #     print(f"输入目录不存在: {base_input_dir}")
        
    # 3. 或者直接处理单个数据集
    print("\n处理单个真实数据集...")
    load_real_data(
        node_file="./case_data/selected_fences_summary_k100.csv",
        distance_file="./case_data/selected_fence_pairwise_dist_km_k100.csv",
        output_path="./casedata/test_100_1/0"
    )