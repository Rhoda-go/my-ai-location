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


# def get_cost(facility_list, distance_m, city_pop, alpha, beta):
#     total_cost = torch.sum(
#         (distance_m[facility_list] * city_pop.flatten())[
#             torch.argmin(distance_m[facility_list], axis=0),
#             torch.arange(distance_m.shape[1]),
#         ]
#     )
#     return total_cost

# def get_cost(facility_list, distance_m, city_pop, alpha, beta):
#     """
#     计算总成本：所有选址点到所有需求点的距离×人口数（可加alpha/beta衰减）
#     参数说明同场景1
#     """
 
#     facility_tensor = torch.tensor(facility_list, dtype=torch.long) if not isinstance(facility_list, torch.Tensor) else facility_list
#       # [n_fac, n_node]

#     total_cost = 0.0
#     for fac in facility_list:
#         dist_from_facility = distance_m[facility_tensor, :]
#         decayed = alpha * torch.exp(- beta * dist_from_facility)  
#         cost = torch.sum(decayed * city_pop.flatten())  # 单个选址点的成本
#         total_cost += cost

#     return total_cost

def get_cost(facility_list, distance_m, city_pop, alpha, beta):
   
  
    facility_tensor = torch.tensor(facility_list, dtype=torch.long) if not isinstance(facility_list, torch.Tensor) else facility_list
    n_fac = facility_tensor.shape[0]  # 选址点数量
    n_node = distance_m.shape[1]     # 需求点数量

    dist_from_facility = distance_m[facility_tensor, :]


    cost_matrix = (
        alpha.unsqueeze(0)  # [n_fac, n_node]
        * torch.exp(-beta.unsqueeze(0) * dist_from_facility)  # e^(-beta×dist)
        * city_pop.flatten().unsqueeze(0)  # [n_fac, n_node]
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
