

import argparse

import numpy as np
import torch
import yaml



def get_config(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        default="config/config.yaml",
    )
    args = parser.parse_args(args)

    with open(args.filename) as yml_file:
        try:
            config = yaml.safe_load(yml_file)
        except yaml.YAMLError as exc:
            print(exc)
    return config


def to_device(state: dict, device):
    if isinstance(state, dict):
        for k, v in state.items():
            state[k] = v.to(device)
    return state


# def get_cost(facility_list, distance_m, city_pop):
#     total_cost = torch.sum(
#         (distance_m[facility_list] * city_pop.flatten())[
#             torch.argmin(distance_m[facility_list], axis=0),
#             torch.arange(distance_m.shape[1]),
#         ]
#     )
#     return total_cost


def get_cost(facility_list, distance_m, city_pop, alpha, beta):
 

  
    #dist_from_facility = distance_m[facility_list, :]


    cost_matrix = (
        alpha[facility_list].unsqueeze(1)  # [n_fac, 1]
        * torch.exp(-beta[facility_list].unsqueeze(1) * distance_m[facility_list])  # e^(-beta×dist)[n_fac, n_node]
        * city_pop.unsqueeze(0)  #[1, n_node]
    )


    total_cost = torch.sum(cost_matrix)

    return total_cost



class DensitySampling:
    def __init__(self, exp):
        self.exp = exp

    def sample(self, city_pop, p):
        density = np.reshape(np.array(city_pop**self.exp), -1)
        density = density / np.sum(density)
        facility_list = np.random.choice(
            city_pop.numel(), size=p, p=density, replace=False
        )
        return facility_list
    
class AlphaSampling:
    def __init__(self, exp):
        self.exp = exp

    def sample(self, alpha, p):
        density = np.reshape(np.array(alpha**self.exp), -1)
        density = density / np.sum(density)
        facility_list = np.random.choice(
            alpha.numel(), size=p, p=density, replace=False
        )
        return facility_list
    

class TabuDensitySampling:        #initial with tabu_table
    def __init__(self, exp):

        self.exp = exp

    def sample(self, city_pop, p, tabu_table):

        city_pop_np = np.reshape(np.array(city_pop), -1)
        #print('city_pop_np',city_pop_np)

        tabu_table_np = np.array(tabu_table, dtype=int)
        tabu_table_min = np.minimum(tabu_table_np, tabu_table_np.T)
    


        facility_list = []  
        available_nodes = np.arange(len(city_pop_np))

       
        while len(facility_list) < p and len(available_nodes) > 0:
     
            pop_available = city_pop_np[available_nodes]
          
            density = pop_available ** self.exp
            density_sum = np.sum(density)
            density = density / density_sum

      
            selected = np.random.choice(available_nodes, size=1, p=density, replace=False)[0]
            facility_list.append(selected)

            if len(facility_list) < p:
                facility_rows = tabu_table_min[facility_list, :]
                filter_mask = np.all(facility_rows == 1, axis=0)
                filter_nodes = np.where(filter_mask)[0]
                available_nodes = np.setdiff1d(filter_nodes, facility_list)

  
        if len(facility_list) < p:
            raise ValueError(f"just {len(facility_list)} nodes selected")

    
        return np.array(facility_list)
    


class TabuAlphaSampling:
    """使用alpha权重进行设施选址的禁忌密度采样"""
    
    def __init__(self, exp):
        self.exp = exp

    def sample(self, alpha, p, tabu_table):
        """
        根据alpha值进行加权采样选择设施位置
        
        参数:
            alpha: array-like, 每个节点的alpha值（衰减系数）
            p: int, 需要选择的设施数量
            tabu_table: array-like, 禁忌表，标记节点间的约束关系
        
        返回:
            facility_list: numpy array, 选中的设施节点索引
        """
        
        # 将alpha转换为numpy数组
        alpha_np = np.reshape(np.array(alpha), -1)
        
        # 转换禁忌表
        tabu_table_np = np.array(tabu_table, dtype=int)
        tabu_table_min = np.minimum(tabu_table_np, tabu_table_np.T)
        
        facility_list = []  
        available_nodes = np.arange(len(alpha_np))
        
        # 迭代选择设施位置
        while len(facility_list) < p and len(available_nodes) > 0:
            # 获取可用节点的alpha值
            alpha_available = alpha_np[available_nodes]
            
            # 根据alpha计算密度权重
            # alpha越大，权重越高（可根据需要调整）
            density = alpha_available ** self.exp
            density_sum = np.sum(density)
            
            # 避免除零错误
            if density_sum == 0:
                # 如果所有alpha都为0，使用均匀分布
                density = np.ones_like(alpha_available) / len(alpha_available)
            else:
                density = density / density_sum
            
            # 按密度权重随机选择一个节点
            selected = np.random.choice(available_nodes, size=1, p=density, replace=False)[0]
            facility_list.append(selected)
            
            # 如果还需要继续选择，更新可用节点
            if len(facility_list) < p:
                # 获取已选设施的禁忌约束
                facility_rows = tabu_table_min[facility_list, :]
                # 找出满足所有已选设施约束的节点
                filter_mask = np.all(facility_rows == 1, axis=0)
                filter_nodes = np.where(filter_mask)[0]
                # 排除已选节点
                available_nodes = np.setdiff1d(filter_nodes, facility_list)
        
        # 检查是否选够了设施
        if len(facility_list) < p:
            raise ValueError(f"只选择了 {len(facility_list)} 个节点，无法满足 {p} 个设施的要求")
        
        return np.array(facility_list)



