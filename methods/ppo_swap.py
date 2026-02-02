
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
    

    # '''
    # 弱禁忌预测,有不可行解
    # '''
    # def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table,alpha,beta,reloc_step, **kwargs):
    #     print('facility_list',facility_list)
    #     start = time.time()
    #     best_sol = None
    #     city_pop = city_pop.to(self.device)
    #     distance_m = distance_m.to(self.device)
    #     alpha = alpha.to(self.device)
    #     beta = beta.to(self.device)
    #     coordinates = kwargs["coordinates"].to(self.device)
    #     road_net_data = kwargs["road_net_data"].to(self.device)
    #     coordinates_norm = (coordinates - torch.min(coordinates, 0)[0]) / max(
    #         torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0]
    #     )

    #     '''
    #     facility_list先用禁忌表过滤一下，防止生成的初始解有禁忌设施
    #     '''


    #     static_feat = torch.cat(
    #         # (coordinates_norm, city_pop.reshape(-1, 1) / torch.sum(city_pop)),
    #         (coordinates_norm, city_pop.reshape(-1, 1) / torch.max(city_pop),
    #           alpha.reshape(-1, 1) / torch.max(alpha), beta.reshape(-1, 1) / torch.max(beta)
    #           ),
    #         axis=1
    #     )
    #     facility_lists = np.tile(facility_list, (self.iter_num, 1))
    #     masks = torch.ones(
    #         (self.iter_num, city_pop.shape[0]), dtype=torch.bool, device=self.device
    #     )
    #     masks[:, facility_list] = 0

      
    #     tabu_table_batch = tabu_table.to(self.device, dtype=torch.bool).unsqueeze(0).repeat(self.iter_num, 1, 1)

    #     for j in range(reloc_step):
    #         fac_data_list = []
    #         for i in range(self.iter_num):
    #             fac_data, cost = self._get_fac_data(
    #                 city_pop,
    #                 p,
    #                 distance_m,
    #                 facility_lists[i],
    #                 static_feat,
    #                 road_net_data,
    #                 #tabu_table,
    #                 alpha,
    #                 beta,
    #                 masks[i],
    #             )
    #             fac_data_list.append(fac_data)
    #             if best_sol is None or cost > best_sol.cost:
    #                 best_sol = PMPSolution(facility_lists[i], np.nan, cost)
         
    #         # get_fac_data_time = time.time() - start
    #         # print('get_fac_data_time',get_fac_data_time)

    #         state = {
    #             "mask": masks,
    #             "tabu_table": tabu_table_batch,
    #             "fac_data": geom_data.Batch.from_data_list(fac_data_list),
    #         }

    #         with torch.no_grad():
    #             action = self.model(state)[1].cpu().numpy()
    #         tabu_table_min= np.minimum(tabu_table, tabu_table.T)
    #         filtered_facility_lists =[] 
    #         fac_out = action[:, 0].astype(np.int64)  # int
    #         fac_in = action[:, 1].astype(np.int64)  # int    
    #         mask_tabu = (tabu_table== 1)  # shape: (n_nodes, n_nodes)
    #         #mask_tabu = (tabu_table == 1)  # shape: (n_nodes, n_nodes)
    #         n_nodes=len(tabu_table[0])
            
    #         for row in range(self.iter_num):
    #             idx = fac_out[row]
    #             filtered_facility_lists.append(set(facility_lists[row]) -{idx})

    #         # tabu_start = time.time()
    #         # iteration
    #         for i in range(self.iter_num):
    #             k_list = filtered_facility_lists[i]  
    #             violate_tabu = False  # 
    #             target_col = fac_in[i]  # fac_in[i] is checked  
    #             #k_filter = np.where(np.all(mask_tabu[list(k_list), :], 0))[0]  #都不冲突的选址下标       
    #             sub_mask = mask_tabu[list(k_list), :]
 
    #             all_true_cols = sub_mask.all(0)
    
    #             k_filter = np.where(all_true_cols)[0]  # 全部为True的列下标集合
                
    #             for k in k_list:
    #                 if not mask_tabu[k, target_col]:  # if not TRUE (tabu_table[k, target_col] == 0)
    #                     violate_tabu = True      
    #                     break  
    #             if violate_tabu:
    #                 # valid_indices = list(set(range(n_nodes))-(set(facility_lists[i]) | {fac_in[i]}))
    #                 # fac_in[i]  = np.random.choice(np.intersect1d(valid_indices, k_filter))
    #                 valid_set = set(range(n_nodes)) - (set(facility_lists[i]) | {fac_in[i]})
    #                 valid_indices = np.intersect1d(list(valid_set), k_filter)
    #                 valid_indices_tensor = torch.from_numpy(valid_indices).long()

    #                 cost_matrix = (
    #                     alpha[valid_indices_tensor].unsqueeze(1)
    #                     * torch.exp(-beta[valid_indices_tensor].unsqueeze(1) * distance_m[valid_indices_tensor])
    #                     * city_pop.unsqueeze(0)
    #                 )

    #                 # 找总收益最大的设施
    #                 total_cost = cost_matrix.sum(dim=1)
    #                 max_fac_idx = torch.argmax(total_cost)
    #                 fac_in[i] = valid_indices[max_fac_idx.item()]


    #             if fac_in[i] in facility_lists[i]:  
    #                 valid_indices=np.setdiff1d(k_filter, facility_lists[i])
    #                 valid_indices_tensor = torch.from_numpy(valid_indices).long()

    #                 cost_matrix = (
    #                     alpha[valid_indices_tensor].unsqueeze(1)
    #                     * torch.exp(-beta[valid_indices_tensor].unsqueeze(1) * distance_m[valid_indices_tensor])
    #                     * city_pop.unsqueeze(0)
    #                 )

    #                 # 找总收益最大的设施
    #                 total_cost = cost_matrix.sum(dim=1)
    #                 max_fac_idx = torch.argmax(total_cost)
    #                 fac_in[i] = valid_indices[max_fac_idx.item()]

    #                 #fac_in[i]  = np.random.choice(np.setdiff1d(k_filter, facility_lists[i]))
               
    #         # tabu_time=time.time()-tabu_start
    #         # print('tabu_time',tabu_time)

    #         # print('distance_m',distance_m)
    #         # print('alpha',alpha)
    #         # print('beta',beta)


    #         fac_out_index = np.where(facility_lists == fac_out[:, None])[1]
    #         facility_lists[np.arange(self.iter_num), fac_out_index] = fac_in

    #         # for i in range(self.iter_num):
    #         #     # 只在第 i 行中查找 fac_out[i]
    #         #     idx = np.where(facility_lists[i] == fac_out[i])[0]
                
    #         #     if len(idx) > 0:
    #         #         pos = idx[0]  # 取第一个匹配位置
    #         #         facility_lists[i, pos] = fac_in[i]
    #         #     else:
    #         #         print(f"Warning: fac_out[{i}]={fac_out[i]} not found in facility_lists[{i}]")

    #         masks[np.arange(self.iter_num), fac_out] = True
    #         masks[np.arange(self.iter_num), fac_in] = False

    #     # print('city_pop',city_pop)

    #     '''
    #     增加最后是否被支配的检测
    #     '''

    #     # print('facility_lists',facility_lists)
    #     valid_mask = np.ones(self.iter_num, dtype=bool)

    #     for i in range(self.iter_num):
    #         check_set = np.setdiff1d(facility_lists[i], facility_list)
    #         # print('check_set',check_set)
    #         if len(check_set) > 0:
    #             # 快速检查是否违反禁忌
    #             tabu_submatrix = tabu_table[np.ix_(facility_lists[i], check_set)]
    #             # 确保使用numpy的any
    #             if (tabu_submatrix == 0).any():
    #                 valid_mask[i] = False
    #     # print(valid_mask)
    #     #best_facility = None
    #     best_sol=None

    #     for i in range(self.iter_num):

    #         if not valid_mask[i]:
    #             continue  # 跳过违反禁忌的解


    #         #wdist = alpha* torch.exp(-beta * distance_m[facility_lists[i]]) * city_pop
    #         wdist = alpha[facility_lists[i]].unsqueeze(1)* torch.exp(-beta[facility_lists[i]].unsqueeze(1) * distance_m[facility_lists[i]]) * city_pop.unsqueeze(0)
    #         # print(facility_lists[i])
    #         # print(alpha.shape,beta.shape,len(facility_lists[i]),city_pop.shape)
    #         #point_indices = torch.argmin(distance_m[facility_list], 0)
    #         node_costs = torch.sum(wdist, dim=0)  #facility to all nodes
    #         cost = torch.sum(node_costs)  #objective value
    #         actual_cost = get_cost(facility_lists[i], distance_m, city_pop, alpha, beta)
    #         assert torch.isclose(actual_cost, cost, rtol=1e-5, atol=1)
    #         if best_sol is None or cost > best_sol.cost:
    #             best_sol = PMPSolution(facility_lists[i], np.nan, cost)
    #             best_facility=facility_lists[i]
            
                


    #     best_sol.time = time.time() - start
    #     # print(city_pop)
    #     #print("facility_lists[np.arange(self.iter_num)",facility_lists[np.arange(self.iter_num)])
    #     print('best_facility',best_sol.facility_list)
    #     print(best_sol.cost)
 
    #     return (best_sol)

    '''
    混合版，边预测边启发式
    '''
    def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table,alpha,beta,reloc_step, **kwargs):
        print('facility_list', facility_list)
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
        static_feat = torch.nan_to_num(static_feat, nan=0.0, posinf=1.0, neginf=0.0)
        

        '''
        '''
        # city_pop = city_pop.to(self.device)
        # distance_m = distance_m.to(self.device)
        # alpha = alpha.to(self.device)
        # beta = beta.to(self.device)
        # coordinates = kwargs["coordinates"].to(self.device)
        # road_net_data = kwargs["road_net_data"].to(self.device)
        # coordinates_norm = (coordinates - torch.min(coordinates, 0)[0]) / max(
        #     torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0]
        # )

        # '''
        # facility_list先用禁忌表过滤一下，防止生成的初始解有禁忌设施
        # '''


        # static_feat = torch.cat(
        #     # (coordinates_norm, city_pop.reshape(-1, 1) / torch.sum(city_pop)),
        #     (coordinates_norm, city_pop.reshape(-1, 1) / torch.max(city_pop),
        #       alpha.reshape(-1, 1) / torch.max(alpha), beta.reshape(-1, 1) / torch.max(beta)
        #       ),
        #     axis=1
        # )





        facility_lists = np.tile(facility_list, (self.iter_num, 1))
        masks = torch.ones(
            (self.iter_num, city_pop.shape[0]), dtype=torch.bool, device=self.device
        )
        masks[:, facility_list] = 0

        conflict_sets = [set() for _ in range(self.iter_num)]
        tabu_table_batch = tabu_table.to(self.device, dtype=torch.bool).unsqueeze(0).repeat(self.iter_num, 1, 1)
        tabu_table_min= np.minimum(tabu_table, tabu_table.T)
        mask_tabu = (tabu_table== 1)  # shape: (n_nodes, n_nodes)
        mask_tabu_min = (tabu_table_min == 1)  # shape: (n_nodes, n_nodes)
        n_nodes=len(tabu_table[0])
  



        #print('facility_list',facility_list)
        for j in range(reloc_step):
            for i in range(self.iter_num):
                if len(conflict_sets[i]) > 0:
                    # 获取有效设施（排除-1）
                    valid_facilities = facility_lists[i][facility_lists[i] >= 0]
                    current_facilities = set(valid_facilities)                
                    # 按照冲突集的顺序逐个处理
                    for conflict_fac in list(conflict_sets[i]):  
                        if conflict_fac in current_facilities:
                            idx = np.where(facility_lists[i] == conflict_fac)[0]
                            if len(idx) > 0:
                                facility_lists[i][idx[0]] = -1  # 使用-1标记
                                masks[i, conflict_fac] = True
                                current_facilities.discard(conflict_fac)
                            
                            # 筛选不冲突点位
                            if len(current_facilities) > 0:
                                sub_mask = mask_tabu_min[list(current_facilities), :]
                                all_true_cols = sub_mask.all(0)
                                k_filter = np.where(all_true_cols)[0]
                            else:
                                k_filter = np.arange(n_nodes)
                            
                            # 排除已在解中的设施
                            valid_indices = np.setdiff1d(k_filter, list(current_facilities))
                            
                            if len(valid_indices) > 0:
                                # 找到不冲突的最大cost点位
                                valid_indices_tensor = torch.from_numpy(valid_indices).long()
                                cost_matrix = (
                                    alpha[valid_indices_tensor].unsqueeze(1)
                                    * torch.exp(-beta[valid_indices_tensor].unsqueeze(1) * distance_m[valid_indices_tensor])
                                    * city_pop.unsqueeze(0)
                                )
                                total_cost = cost_matrix.sum(dim=1)
                            
                                max_fac_idx = torch.argmax(total_cost)
                                selected_facility = valid_indices[max_fac_idx.item()]
                                
                                # 将新设施填入刚才删除的位置
                                empty_slot = np.where(facility_lists[i] == -1)[0][0]  # 找到第一个-1的位置
                                facility_lists[i][empty_slot] = selected_facility
                                masks[i, selected_facility] = False
                                
                                # 更新 current_facilities
                                current_facilities.add(selected_facility)
                            else:
                                # 无可用设施，跳过（保持-1）
                                print(f"Warning: Iteration {i}, no valid facility to replace {conflict_fac}")
                    
                    # 清空冲突集
                    conflict_sets[i].clear()

      
                if -1 in facility_lists[i]:
                    facility_lists[i]=facility_list.copy()
                    masks[i] = torch.ones(city_pop.shape[0], dtype=torch.bool, device=self.device)
                    masks[i, facility_list] = False


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
            fac_out = action[:, 0].astype(np.int64)  # int
            fac_in = action[:, 1].astype(np.int64)  # int    

            filtered_facility_lists =[]             
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
                    #print('violate_tabu')
                    # valid_indices = list(set(range(n_nodes))-(set(facility_lists[i]) | {fac_in[i]}))
                    # fac_in[i]  = np.random.choice(np.intersect1d(valid_indices, k_filter))
                    valid_set = set(range(n_nodes)) - (set(facility_lists[i]) | {fac_in[i]})
                    valid_indices = np.intersect1d(list(valid_set), k_filter)
                    if len(valid_indices)>0:
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
                else:
                    continue

                if fac_in[i] in facility_lists[i]:  
                    valid_indices=np.setdiff1d(k_filter, facility_lists[i])
                    if valid_indices>0:

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
                    else:
                        continue
                    #fac_in[i]  = np.random.choice(np.setdiff1d(k_filter, facility_lists[i]))


            fac_out_index = np.where(facility_lists == fac_out[:, None])[1]
            facility_lists[np.arange(self.iter_num), fac_out_index] = fac_in

            # for i in range(self.iter_num):
            #     # 只在第 i 行中查找 fac_out[i]
            #     idx = np.where(facility_lists[i] == fac_out[i])[0]
                
            #     if len(idx) > 0:
            #         pos = idx[0]  # 取第一个匹配位置
            #         facility_lists[i, pos] = fac_in[i]
            #     else:
            #         print(f"Warning: fac_out[{i}]={fac_out[i]} not found in facility_lists[{i}]")

            masks[np.arange(self.iter_num), fac_out] = True
            masks[np.arange(self.iter_num), fac_in] = False

            #换入设施后，根据弱禁忌集更新冲突集
            for i in range(self.iter_num):
                swap_in = fac_in[i]
                current_facilities = set(facility_lists[i])
                
                # 检查 swap_in 支配哪些设施（使用弱禁忌表，即原始 tabu_table）
                for fac in current_facilities:
                    if fac != swap_in and not mask_tabu[swap_in, fac]:  # tabu_table[swap_in, fac] == 0
                        conflict_sets[i].add(fac)

        #best_facility = None
        # best_sol=None

        # facility_list_=facility_list.copy()

        for i in range(self.iter_num):
            
            #若最后一步换入冲突，则解不可用
            if len(conflict_sets[i])>0:
                continue

            # '''
            # reloc 条件
            # '''

            # diff_count = len(set(facility_lists[i]) ^ set(facility_list)) // 2 # 对称差集
            # print(' diff_count', diff_count)
            # if diff_count > reloc_step:
            #     continue



            #wdist = alpha* torch.exp(-beta * distance_m[facility_lists[i]]) * city_pop
            wdist = alpha[facility_lists[i]].unsqueeze(1)* torch.exp(-beta[facility_lists[i]].unsqueeze(1) * distance_m[facility_lists[i]]) * city_pop.unsqueeze(0)
          
            node_costs = torch.sum(wdist, dim=0)  #facility to all nodes
            cost = torch.sum(node_costs)  #objective value
            actual_cost = get_cost(facility_lists[i], distance_m, city_pop, alpha, beta)
            assert torch.isclose(actual_cost, cost, rtol=1e-5, atol=1)
            if best_sol is None or cost > best_sol.cost:
                best_sol = PMPSolution(facility_lists[i], np.nan, cost)
                best_facility=facility_lists[i]
        
        # if best_sol is None:
        #     best_sol = PMPSolution(facility_list, np.nan, get_cost(facility_list, distance_m, city_pop, alpha, beta))
    

        best_sol.time = time.time() - start
        # print(city_pop)
        #print("facility_lists[np.arange(self.iter_num)",facility_lists[np.arange(self.iter_num)])
<<<<<<< HEAD
        print('best_facility',best_sol.facility_list)
        print(best_sol.cost)
        print('reloc_step',reloc_step)
=======
        # print('best_facility',best_sol.facility_list)
        # print(best_sol.cost)
        # print('reloc_step',reloc_step)
>>>>>>> lx
 
        return (best_sol)




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
