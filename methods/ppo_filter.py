
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
from utils import TabuAlphaSampling,get_cost
 

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
        


        wdist = alpha[facility_list].unsqueeze(1)* torch.exp(-beta[facility_list].unsqueeze(1) * distance_m[facility_list]) * city_pop.unsqueeze(0)
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
    

    '''
    弱禁忌预测,每次迭代后增加禁忌策略
    '''
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

        '''
        facility_list先用禁忌表过滤一下，防止生成的初始解有禁忌设施
        '''


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
         
            # get_fac_data_time = time.time() - start
            # print('get_fac_data_time',get_fac_data_time)

            state = {
                "mask": masks,
                "tabu_table": tabu_table_batch,
                "fac_data": geom_data.Batch.from_data_list(fac_data_list),
            }

            with torch.no_grad():
                action = self.model(state)[1].cpu().numpy()
            tabu_table_min= np.minimum(tabu_table, tabu_table.T)
            filtered_facility_lists =[] 
            fac_out = action[:, 0].astype(np.int64)  # int
            fac_in = action[:, 1].astype(np.int64)  # int    
            mask_tabu = (tabu_table== 1)  # shape: (n_nodes, n_nodes)
            mask_tabu_min = (tabu_table_min == 1)  # shape: (n_nodes, n_nodes)
            n_nodes=len(tabu_table[0])
            
            for row in range(self.iter_num):
                idx = fac_out[row]
                filtered_facility_lists.append(set(facility_lists[row]) -{idx})

            # tabu_start = time.time()
            # iteration
            for i in range(self.iter_num):
                k_list = filtered_facility_lists[i]  
                violate_tabu = False  # 
                target_col = fac_in[i]  # fac_in[i] is checked  
                #k_filter = np.where(np.all(mask_tabu[list(k_list), :], 0))[0]  #都不冲突的选址下标       
                sub_mask = mask_tabu[list(k_list), :]
 
                all_true_cols = sub_mask.all(0)
    
                k_filter = np.where(all_true_cols)[0]  # 全部为True的列下标集合
                
                for k in k_list:
                    if not mask_tabu[k, target_col]:  # if not TRUE (tabu_table[k, target_col] == 0)
                        violate_tabu = True      
                        break  
                if violate_tabu:
                    # valid_indices = list(set(range(n_nodes))-(set(facility_lists[i]) | {fac_in[i]}))
                    # fac_in[i]  = np.random.choice(np.intersect1d(valid_indices, k_filter))
                    valid_set = set(range(n_nodes)) - (set(facility_lists[i]) | {fac_in[i]})
                    valid_indices = np.intersect1d(list(valid_set), k_filter)
                    valid_indices_tensor = torch.from_numpy(valid_indices).long()

                    cost_matrix = (
                        alpha[valid_indices_tensor].unsqueeze(1)
                        * torch.exp(-beta[valid_indices_tensor].unsqueeze(1) * distance_m[valid_indices_tensor])
                        * city_pop.unsqueeze(0)
                    )

                    # 找总收益最大的设施
                    total_cost = cost_matrix.sum(dim=1)
                    max_fac_idx = torch.argmax(total_cost)
                    fac_in[i] = valid_indices[max_fac_idx.item()]


                if fac_in[i] in facility_lists[i]:  
                    valid_indices=np.setdiff1d(k_filter, facility_lists[i])
                    valid_indices_tensor = torch.from_numpy(valid_indices).long()

                    cost_matrix = (
                        alpha[valid_indices_tensor].unsqueeze(1)
                        * torch.exp(-beta[valid_indices_tensor].unsqueeze(1) * distance_m[valid_indices_tensor])
                        * city_pop.unsqueeze(0)
                    )

                    # 找总收益最大的设施
                    total_cost = cost_matrix.sum(dim=1)
                    max_fac_idx = torch.argmax(total_cost)
                    fac_in[i] = valid_indices[max_fac_idx.item()]


            fac_out_index = np.where(facility_lists == fac_out[:, None])[1]
            facility_lists[np.arange(self.iter_num), fac_out_index] = fac_in

            masks[np.arange(self.iter_num), fac_out] = True
            masks[np.arange(self.iter_num), fac_in] = False

        #filterd by tabu_mask,得到最终的可行解
        for i in range(self.iter_num):

            # 步骤1：检测冲突集
            conflict_sets = set()
            current_facilities = set(facility_lists[i])
            
            # 对每个设施，检查它与其他设施的冲突
            for fac in current_facilities:
                for other_fac in current_facilities:
                    if fac != other_fac and not mask_tabu[fac, other_fac]:  # tabu_table[fac, other_fac] == 0
                        conflict_sets.add(other_fac)  # other_fac 与 fac 冲突
            
            if len(conflict_sets) == 0:
                continue  # 无冲突，跳过
            
            # print(f"解 {i} 发现冲突: {conflict_sets}")
            
            # 步骤2：迭代删除冲突设施并替换
            max_iterations = len(conflict_sets) * 2  # 防止死循环
            iteration_count = 0
            
            while len(conflict_sets) > 0 and iteration_count < max_iterations:
                iteration_count += 1
                
                # 获取有效设施（排除-1）
                valid_facilities = facility_lists[i][facility_lists[i] >= 0]
                current_facilities = set(valid_facilities)
                
                # 按照冲突集的顺序逐个处理
                conflict_fac = conflict_sets.pop()  # 取出一个冲突设施
                
                if conflict_fac not in current_facilities:
                    continue  # 已被删除，跳过
                
                # 步骤2.1：删除冲突设施
                idx = np.where(facility_lists[i] == conflict_fac)[0]
                if len(idx) > 0:
                    facility_lists[i][idx[0]] = -1  # 使用-1标记
                    masks[i, conflict_fac] = True
                    current_facilities.discard(conflict_fac)
                    # print(f"  删除冲突设施 {conflict_fac}")
                
                # 步骤2.2：立即为这个空位选择替换设施
                if len(current_facilities) > 0:
                    # 计算与当前所有设施都不冲突的候选
                    sub_mask = mask_tabu_min[list(current_facilities), :]
                    all_true_cols = sub_mask.all(0)
                    k_filter = np.where(all_true_cols)[0]
                else:
                    # 如果当前没有设施，所有节点都可选
                    k_filter = np.arange(n_nodes)
                
                # 排除已在解中的设施
                valid_indices = np.setdiff1d(k_filter, list(current_facilities))
                
                if len(valid_indices) > 0:
                    # 计算每个候选设施的收益
                    valid_indices_tensor = torch.from_numpy(valid_indices).long().to(self.device)
                    cost_matrix = (
                        alpha[valid_indices_tensor].unsqueeze(1)
                        * torch.exp(-beta[valid_indices_tensor].unsqueeze(1) * distance_m[valid_indices_tensor])
                        * city_pop.unsqueeze(0)
                    )
                    total_cost = cost_matrix.sum(dim=1)
                    
                    # 选择收益最大的1个设施
                    max_fac_idx = torch.argmax(total_cost)
                    selected_facility = valid_indices[max_fac_idx.item()]
                    
                    # 将新设施填入刚才删除的位置
                    empty_slot = np.where(facility_lists[i] == -1)[0][0]  # 找到第一个-1的位置
                    facility_lists[i][empty_slot] = selected_facility
                    masks[i, selected_facility] = False
                    
                    # 更新 current_facilities
                    current_facilities.add(selected_facility)
            # print('facility_lists[i]',facility_lists[i])

        best_sol=None

       #print('facility_lists',facility_lists)

        for i in range(self.iter_num):
            if -1 in facility_lists[i]:
                continue

            # '''
            # reloc条件
            # '''
            # diff_count = len(set(facility_lists[i]) ^ set(facility_list))//2  # 对称差集
        
            # if diff_count > reloc_step:
            #     continue



            #wdist = alpha* torch.exp(-beta * distance_m[facility_lists[i]]) * city_pop
            wdist = alpha[facility_lists[i]].unsqueeze(1)* torch.exp(-beta[facility_lists[i]].unsqueeze(1) * distance_m[facility_lists[i]]) * city_pop.unsqueeze(0)
            # print(facility_lists[i])
            # print(alpha.shape,beta.shape,len(facility_lists[i]),city_pop.shape)
            #point_indices = torch.argmin(distance_m[facility_list], 0)
            node_costs = torch.sum(wdist, dim=0)  #facility to all nodes
            cost = torch.sum(node_costs)  #objective value
            actual_cost = get_cost(facility_lists[i], distance_m, city_pop, alpha, beta)
            assert torch.isclose(actual_cost, cost, rtol=1e-5, atol=1)
            if best_sol is None or cost > best_sol.cost:
                best_sol = PMPSolution(facility_lists[i], np.nan, cost)
 
        if best_sol is None:
            best_sol = PMPSolution(facility_list, np.nan, get_cost(facility_list, distance_m, city_pop, alpha, beta))

                
        best_sol.time = time.time() - start
        # print(city_pop)
        #print("facility_lists[np.arange(self.iter_num)",facility_lists[np.arange(self.iter_num)])
        # print('best_facility',best_facility)
        # print(best_sol.cost)
 
        return (best_sol)




def run_ppo_filter(
    dataset, save_path, iter_num, swap_num, init_num, ckpt, device, **kwargs
):
    name = f'ppo_filter_{init_num}_{iter_num}_{swap_num}_{kwargs["name"]}'
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

def run_ppo_filter_reloc(dataset, save_path, iter_num, ckpt, device, reloc_coef, **kwargs):
    name = f'ppo_filter_{iter_num}_{kwargs["name"]}'
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
        #facility_list = TabuAlphaSampling(exp=1).sample(city_pop, p, tabu_table)
        #print('facility_list',facility_list)
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
