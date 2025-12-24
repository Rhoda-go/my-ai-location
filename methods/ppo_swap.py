
import os
import sys
import pickle
import time

import numpy as np
import torch
import torch_geometric.data as geom_data

#from swap_solver import SwapSolver 
from methods.swap_solver import SwapSolver
from results import PMPSolution
from train import PPOLightning
from utils import get_cost

# 获取项目根目录（methods文件夹的父目录）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# 将项目根目录加入sys.path
sys.path.append(project_root)

class PPOSwapSolver(SwapSolver):
    def __init__(self, iter_num, ckpt, device):
        super().__init__(iter_num)
        #self.model = torch.load(ckpt, map_location=device)
        self.model = (
            PPOLightning.load_from_checkpoint(ckpt, mode="test", weights_only=True ).float().to(device)
        )
        self.device = device
        #self.warm_up()

    def warm_up(self):
        rand_state = {
            "mask": torch.randint(
                0, 2, (self.iter_num, 50), dtype=torch.bool, device=self.device
            ),

            "tabu_table": torch.randint(
            0, 2, (self.iter_num, 50, 50), dtype=torch.bool, device=self.device
        ),

            "fac_data": geom_data.Batch.from_data_list(
                [
                    geom_data.Data(
                        x=torch.rand(50, 7, device=self.device),
                        edge_index=torch.randint(0, 50, (2, 50), device=self.device),
                        edge_attr=torch.rand(20, 1, device=self.device),
                    )
                    for _ in range(self.iter_num)
                ]
            ),
        }
        with torch.no_grad():
            self.model(rand_state)

    def _get_fac_data(
        self,
        city_pop,
        p,
        distance_m,
        facility_list,
        static_feat,
        road_net_data,
        #tabu_table,
        alpha,
        beta,
        mask,
    ):

        wdist = alpha* torch.exp(-beta * distance_m[facility_list]) * city_pop
        #point_indices = torch.argmin(distance_m[facility_list], 0)
        node_costs = torch.sum(wdist, dim=0)  #facility to all nodes
        total_cost = torch.sum(node_costs)  #objective value

        # fac_costs = torch.zeros(p, device=wdist.device)
        # fac_pop = torch.zeros(p, device=city_pop.device)

        # fac_costs.scatter_add_(0, point_indices, node_costs)
        # fac_pop.scatter_add_(0, point_indices, city_pop)

        # fac_feat = torch.cat(
        #     (
        #         # fac_pop.reshape(-1, 1) / torch.sum(city_pop),
        #         # fac_costs.reshape(-1, 1) / total_cost,
        #         fac_pop.reshape(-1, 1) / torch.max(fac_pop),
        #         fac_costs.reshape(-1, 1) / torch.max(fac_costs),
        #     ),
        #     axis=1,
        # )
        # node_fac_feat = torch.zeros(
        #     (city_pop.shape[0], fac_feat.shape[1]), device=self.device
        # )
        # node_fac_feat[facility_list] = fac_feat

        node_feat = torch.cat(
            (
                static_feat,
                mask.reshape(-1, 1),
                # node_costs.reshape(-1, 1) / total_cost,
                node_costs.reshape(-1, 1) / torch.max(node_costs),
                #node_fac_feat,
            ),
            axis=1,
        )

        fac_data = geom_data.Data(
            x=node_feat,
            edge_index=road_net_data.edge_index,
            edge_attr=road_net_data.edge_attr,
        )

        return fac_data, total_cost

    def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table,alpha,beta,reloc_step, **kwargs):

        start = time.time()
        best_sol = None
        city_pop = city_pop.to(self.device)
        distance_m = distance_m.to(self.device)
        alpha = alpha.to(self.device)
        beta = beta.to(self.device)
        coordinates = kwargs["coordinates"].to(self.device)
        road_net_data = kwargs["road_net_data"].to(self.device)
        coordinates_norm = (coordinates - torch.min(coordinates, 0)[0]) / max(
            torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0]
        )
        static_feat = torch.cat(
            # (coordinates_norm, city_pop.reshape(-1, 1) / torch.sum(city_pop)),
            (coordinates_norm, city_pop.reshape(-1, 1) / torch.max(city_pop),
              alpha.reshape(-1, 1) / torch.max(alpha), beta.reshape(-1, 1) / torch.max(beta)
              ),
            axis=1
        )
        facility_lists = np.tile(facility_list, (self.iter_num, 1))
        masks = torch.ones(
            (self.iter_num, city_pop.shape[0]), dtype=torch.bool, device=self.device
        )
        masks[:, facility_list] = 0

      
        tabu_table_batch = tabu_table.to(self.device, dtype=torch.bool).unsqueeze(0).repeat(self.iter_num, 1, 1)

        for j in range(reloc_step):
            fac_data_list = []
            for i in range(self.iter_num):
                fac_data, cost = self._get_fac_data(
                    city_pop,
                    p,
                    distance_m,
                    facility_lists[i],
                    static_feat,
                    road_net_data,
                    #tabu_table,
                    alpha,
                    beta,
                    masks[i],
                )
                fac_data_list.append(fac_data)
                if best_sol is None or cost > best_sol.cost:
                    best_sol = PMPSolution(facility_lists[i], np.nan, cost)

            state = {
                "mask": masks,
                "tabu_table": tabu_table_batch,
                "fac_data": geom_data.Batch.from_data_list(fac_data_list),
            }

            with torch.no_grad():
                action = self.model(state)[1].cpu().numpy()
            filtered_facility_lists =[] 
            fac_out = action[:, 0].astype(np.int64)  # int
            fac_in = action[:, 1].astype(np.int64)  # int    
            mask_tabu = (tabu_table == 1)  # shape: (n_nodes, n_nodes)
            n_nodes=len(tabu_table[0])
            
            for row in range(self.iter_num):
                idx = fac_out[row]
                filtered_facility_lists.append(set(facility_lists[row]) -{idx})

            
            # iteration
            for i in range(self.iter_num):
                k_list = filtered_facility_lists[i]  
                violate_tabu = False  # 
                target_col = fac_in[i]  # fac_in[i] is checked           
                for k in k_list:                 
                    if not mask_tabu[k, target_col]:  # if not TRUE (tabu_table[k, target_col] == 0)
                        violate_tabu = True      
                        break  
                if violate_tabu:
                    valid_indices = list(set(range(n_nodes))-(set(facility_lists[i]) | {fac_in[i]}))
                    fac_in[i] = np.random.choice(valid_indices)

            fac_out_index = np.where(facility_lists == fac_out[:, None])[1]
            facility_lists[np.arange(self.iter_num), fac_out_index] = fac_in

            masks[np.arange(self.iter_num), fac_out] = True
            masks[np.arange(self.iter_num), fac_in] = False

        for i in range(self.iter_num):
            wdist = alpha* torch.exp(-beta * distance_m[facility_lists[i]]) * city_pop
            # print(facility_lists[i])
            # print(alpha.shape,beta.shape,len(facility_lists[i]),city_pop.shape)
            #point_indices = torch.argmin(distance_m[facility_list], 0)
            node_costs = torch.sum(wdist, dim=0)  #facility to all nodes
            cost = torch.sum(node_costs)  #objective value
            actual_cost = get_cost(facility_lists[i], distance_m, city_pop, alpha, beta)
            assert torch.isclose(actual_cost, cost, rtol=1e-5, atol=1)
            if best_sol is None or cost > best_sol.cost:
                best_sol = PMPSolution(facility_lists[i], np.nan, cost)

        best_sol.time = time.time() - start
        print(city_pop)
        print("facility_lists[np.arange(self.iter_num)",facility_lists[np.arange(self.iter_num)])
        print(best_sol.cost)
        return best_sol




def run_ppo_swap(
    dataset, save_path, iter_num, swap_num, init_num, ckpt, device, **kwargs
):
    name = f'ppo_swap_{init_num}_{iter_num}_{swap_num}_{kwargs["name"]}'
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = PPOSwapSolver(iter_num, ckpt, device)
    for batch in dataset:
        city_id, city_pop, p, distance_m, coordinates, road_net_data, alpha, beta, tabu_table = batch[:9]
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve(
                p,
                city_pop,
                distance_m,
                swap_num,
                init_num,
                tabu_table,
                alpha,
                beta,
                coordinates=coordinates,
                road_net_data=road_net_data,
            )
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
            
    return sol_path

def run_ppo_swap_reloc(dataset, save_path, iter_num, ckpt, device, reloc_coef, **kwargs):
    name = f'ppo_swap_{iter_num}_{kwargs["name"]}'
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = PPOSwapSolver(iter_num, ckpt, device)
    for batch in dataset:
        (
            city_id,
            city_pop,
            p,
            distance_m,
            coordinates,
            road_net_data,
            alpha,
            beta,
            tabu_table,
            facility_list,
        ) = batch
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(
                city_pop,
                p,
                distance_m,
                facility_list,
                tabu_table,
                alpha,
                beta,
                int(reloc_coef * p),
                coordinates=coordinates,
                road_net_data=road_net_data,
            )
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
            
    return sol_path
