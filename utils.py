

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
 

  
    dist_from_facility = distance_m[facility_list, :]


    cost_matrix = (
        alpha.unsqueeze(0)  # [n_fac, n_node]
        * torch.exp(-beta.unsqueeze(0) * dist_from_facility)  # e^(-beta×dist)
        * city_pop.flatten().unsqueeze(0)  #[n_fac, n_node]
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
    

class TabuDensitySampling:        #initial with tabu_table
    def __init__(self, exp):

        self.exp = exp

    def sample(self, city_pop, p, tabu_table):

        city_pop_np = np.reshape(np.array(city_pop), -1)
        #print('city_pop_np',city_pop_np)

        tabu_table_np = np.array(tabu_table, dtype=int)
        #print(tabu_table_np)
    


        facility_list = []  
        available_nodes = np.arange(len(city_pop_np))

       
        while len(facility_list) < p and len(available_nodes) > 0:
     
            pop_available = city_pop_np[available_nodes]
          
            density = pop_available ** self.exp
            density_sum = np.sum(density)
            density = density / density_sum

      
            selected = np.random.choice(available_nodes, size=1, p=density, replace=False)[0]
            facility_list.append(selected)

            tabu_nodes = np.where(tabu_table_np[selected] == 0)[0]  #==0 as true
       
            available_nodes = np.array([
                node for node in available_nodes
                if node not in facility_list and node not in tabu_nodes
            ])

  
        if len(facility_list) < p:
            raise ValueError(f"just {len(facility_list)} nodes selected")

    
        return np.array(facility_list)


