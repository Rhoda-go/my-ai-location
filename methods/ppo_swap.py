

# import os
# import sys
# import pickle
# import time

# import numpy as np
# import torch
# import torch_geometric.data as geom_data

# #from swap_solver import SwapSolver 
# from methods.swap_solver import SwapSolver
# from results import PMPSolution
# from train import PPOLightning
# from utils import TabuAlphaSampling,get_cost
 

# # 获取项目根目录（methods文件夹的父目录）
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # 将项目根目录加入sys.path
# sys.path.append(project_root)

# class PPOSwapSolver(SwapSolver):
#     def __init__(self, iter_num, ckpt, device):
#         super().__init__(iter_num)
#         #self.model = torch.load(ckpt, map_location=device)
#         self.model = (
#             PPOLightning.load_from_checkpoint(ckpt, mode="test", weights_only=True ).float().to(device)
#         )
#         self.device = device
#         #self.warm_up()

#     def warm_up(self):
#         rand_state = {
#             "mask": torch.randint(
#                 0, 2, (self.iter_num, 50), dtype=torch.bool, device=self.device
#             ),

#             "tabu_table": torch.randint(
#             0, 2, (self.iter_num, 50, 50), dtype=torch.bool, device=self.device
#         ),

#             "fac_data": geom_data.Batch.from_data_list(
#                 [
#                     geom_data.Data(
#                         x=torch.rand(50, 7, device=self.device),
#                         edge_index=torch.randint(0, 50, (2, 50), device=self.device),
#                         edge_attr=torch.rand(20, 1, device=self.device),
#                     )
#                     for _ in range(self.iter_num)
#                 ]
#             ),
#         }
#         with torch.no_grad():
#             self.model(rand_state)

#     def _get_fac_data(
#         self,
#         city_pop,
#         p,
#         distance_m,
#         facility_list,
#         static_feat,
#         road_net_data,
#         #tabu_table,
#         alpha,
#         beta,
#         mask,
#         facility_vectors=None,
#     ):
#         # If facility_vectors provided (precomputed on device), use it to avoid repeated exp computations
#         if facility_vectors is None:
#             wdist = alpha[facility_list].unsqueeze(1) * torch.exp(
#                 -beta[facility_list].unsqueeze(1) * distance_m[facility_list]
#             ) * city_pop.unsqueeze(0)
#         else:
#             # ensure facility_list is a torch index tensor on the same device
#             if not torch.is_tensor(facility_list):
#                 facility_list_t = torch.tensor(facility_list, dtype=torch.long, device=self.device)
#             else:
#                 if facility_list.dtype == torch.long and facility_list.device == self.device:
#                     facility_list_t = facility_list
#                 else:
#                     facility_list_t = facility_list.long().to(self.device)
#             wdist = facility_vectors[facility_list_t]
#         #point_indices = torch.argmin(distance_m[facility_list], 0)
#         node_costs = torch.sum(wdist, dim=0)  #facility to all nodes
#         total_cost = torch.sum(node_costs)  #objective value


#         node_feat = torch.cat(
#             (
#                 static_feat,
#                 mask.reshape(-1, 1),
#                 # node_costs.reshape(-1, 1) / total_cost,
#                 node_costs.reshape(-1, 1) / torch.max(node_costs),
#                 #node_fac_feat,
#             ),
#             axis=1,
#         )

#         fac_data = geom_data.Data(
#             x=node_feat,
#             edge_index=road_net_data.edge_index,
#             edge_attr=road_net_data.edge_attr,
#         )

#         return fac_data, total_cost
    

#     '''
#     混合版，边预测边启发式
#     '''
#     def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table,alpha,beta,reloc_step, **kwargs):
#         start = time.time()
#         best_sol = None

#         def _to_device_tensor(x, dtype=None):
#             if torch.is_tensor(x):
#                 return x.to(self.device, dtype=dtype) if dtype is not None else x.to(self.device)
#             return torch.tensor(x, dtype=dtype, device=self.device)

#         city_pop = _to_device_tensor(city_pop)
#         distance_m = _to_device_tensor(distance_m)
#         alpha = _to_device_tensor(alpha)
#         beta = _to_device_tensor(beta)
#         coordinates = _to_device_tensor(kwargs["coordinates"])
#         road_net_data = kwargs["road_net_data"]
#         if hasattr(road_net_data, "to"):
#             road_net_data = road_net_data.to(self.device)

#         if torch.is_tensor(tabu_table):
#             tabu_table_t = tabu_table.to(self.device, dtype=torch.bool)
#         else:
#             tabu_table_t = torch.tensor(tabu_table, dtype=torch.bool, device=self.device)

#         if torch.is_tensor(facility_list):
#             facility_list_t = facility_list.to(self.device, dtype=torch.long)
#         else:
#             facility_list_t = torch.tensor(facility_list, dtype=torch.long, device=self.device)

#         coordinates_norm = (coordinates - torch.min(coordinates, 0)[0]) / max(
#             torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0]
#         )

#         '''
#         facility_list先用禁忌表过滤一下，防止生成的初始解有禁忌设施
#         '''


#         static_feat = torch.cat(
#             # (coordinates_norm, city_pop.reshape(-1, 1) / torch.sum(city_pop)),
#             (coordinates_norm, city_pop.reshape(-1, 1) / torch.max(city_pop),
#               alpha.reshape(-1, 1) / torch.max(alpha), beta.reshape(-1, 1) / torch.max(beta)
#               ),
#             axis=1
#         )
#         static_feat = torch.nan_to_num(static_feat, nan=0.0, posinf=1.0, neginf=0.0)
        

#         '''
#         '''
#         # city_pop = city_pop.to(self.device)
#         # distance_m = distance_m.to(self.device)
#         # alpha = alpha.to(self.device)
#         # beta = beta.to(self.device)
#         # coordinates = kwargs["coordinates"].to(self.device)
#         # road_net_data = kwargs["road_net_data"].to(self.device)
#         # coordinates_norm = (coordinates - torch.min(coordinates, 0)[0]) / max(
#         #     torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0]
#         # )

#         # '''
#         # facility_list先用禁忌表过滤一下，防止生成的初始解有禁忌设施
#         # '''


#         # static_feat = torch.cat(
#         #     # (coordinates_norm, city_pop.reshape(-1, 1) / torch.sum(city_pop)),
#         #     (coordinates_norm, city_pop.reshape(-1, 1) / torch.max(city_pop),
#         #       alpha.reshape(-1, 1) / torch.max(alpha), beta.reshape(-1, 1) / torch.max(beta)
#         #       ),
#         #     axis=1
#         # )





#         facility_lists = facility_list_t.unsqueeze(0).repeat(self.iter_num, 1)
#         masks = torch.ones(
#             (self.iter_num, city_pop.shape[0]), dtype=torch.bool, device=self.device
#         )
#         masks[:, facility_list_t] = 0

#         conflict_sets = [set() for _ in range(self.iter_num)]
#         tabu_table_batch = tabu_table_t.unsqueeze(0).expand(self.iter_num, -1, -1)
#         mask_tabu = tabu_table_t
#         mask_tabu_min = torch.logical_and(tabu_table_t, tabu_table_t.t())
#         n_nodes = tabu_table_t.shape[0]
#         all_nodes_t = torch.arange(n_nodes, device=self.device)
  



#         # Precompute facility influence vectors on device to avoid repeated exp calls
#         with torch.no_grad():
#             facility_vectors = (
#                 alpha.unsqueeze(1) * torch.exp(-beta.unsqueeze(1) * distance_m) * city_pop.unsqueeze(0)
#             ).to(self.device)
#             facility_total_values = facility_vectors.sum(dim=1)
#         batch_idx_t = torch.arange(self.iter_num, device=self.device)

#         for j in range(reloc_step):
#             for i in range(self.iter_num):
#                 if len(conflict_sets[i]) > 0:
#                     # 获取有效设施（排除-1）
#                     valid_facilities = facility_lists[i][facility_lists[i] >= 0]
#                     current_facilities = set(valid_facilities.tolist())
#                     # 按照冲突集的顺序逐个处理
#                     for conflict_fac in list(conflict_sets[i]):  
#                         if conflict_fac in current_facilities:
#                             idx = torch.where(facility_lists[i] == conflict_fac)[0]
#                             if idx.numel() > 0:
#                                 facility_lists[i][idx[0]] = -1  # 使用-1标记
#                                 masks[i, conflict_fac] = True
#                                 current_facilities.discard(conflict_fac)
                            
#                             # 筛选不冲突点位
#                             if len(current_facilities) > 0:
#                                 current_t = torch.tensor(list(current_facilities), dtype=torch.long, device=self.device)
#                                 sub_mask = mask_tabu_min[current_t, :]
#                                 all_true_cols = sub_mask.all(0)
#                                 k_filter = torch.where(all_true_cols)[0]
#                             else:
#                                 k_filter = all_nodes_t
                            
#                             # 排除已在解中的设施
#                             if len(current_facilities) > 0:
#                                 current_t = torch.tensor(list(current_facilities), dtype=torch.long, device=self.device)
#                                 valid_indices = k_filter[~torch.isin(k_filter, current_t)]
#                             else:
#                                 valid_indices = k_filter
                            
#                             if valid_indices.numel() > 0:
#                                 # 找到不冲突的最大总贡献点位，使用预计算的 facility_total_values
#                                 max_fac_idx = torch.argmax(facility_total_values[valid_indices])
#                                 selected_facility = int(valid_indices[max_fac_idx].item())
                                
#                                 # 将新设施填入刚才删除的位置
#                                 empty_slot = torch.where(facility_lists[i] == -1)[0][0]  # 找到第一个-1的位置
#                                 facility_lists[i][empty_slot] = selected_facility
#                                 masks[i, selected_facility] = False
                                
#                                 # 更新 current_facilities
#                                 current_facilities.add(selected_facility)
#                             else:
#                                 # 无可用设施，跳过（保持-1）
#                                 pass
                    
#                     # 清空冲突集
#                     conflict_sets[i].clear()

      
#                 if torch.any(facility_lists[i] == -1):
#                     facility_lists[i] = facility_list_t.clone()
#                     masks[i].fill_(True)
#                     masks[i, facility_list_t] = False


#             fac_data_list = []
#             for i in range(self.iter_num):
#                 fac_data, cost = self._get_fac_data(
#                     city_pop,
#                     p,
#                     distance_m,
#                     facility_lists[i],
#                     static_feat,
#                     road_net_data,
#                     #tabu_table,
#                     alpha,
#                     beta,
#                     masks[i],
#                     facility_vectors=facility_vectors,
#                 )


#                 fac_data_list.append(fac_data)
#                 if best_sol is None or cost > best_sol.cost:
#                     best_sol = PMPSolution(facility_lists[i].detach().cpu().numpy(), np.nan, cost)


         
#             # get_fac_data_time = time.time() - start
#             # print('get_fac_data_time',get_fac_data_time)
#             state = {
#                 "mask": masks,
#                 "tabu_table": tabu_table_batch,
#                 "fac_data": geom_data.Batch.from_data_list(fac_data_list),
#             }
#             with torch.no_grad():
#                 action = self.model(state)[1].long()
#             fac_out_t = action[:, 0]
#             fac_in_t = action[:, 1]

#             filtered_facility_lists =[]             
#             for row in range(self.iter_num):
#                 idx = int(fac_out_t[row].item())
#                 filtered_facility_lists.append(set(facility_lists[row].tolist()) - {idx})

#             # tabu_start = time.time()
#             # iteration
#             for i in range(self.iter_num):
#                 k_list = filtered_facility_lists[i]  
#                 violate_tabu = False  # 
#                 target_col = int(fac_in_t[i].item())  # fac_in[i] is checked
#                 #k_filter = np.where(np.all(mask_tabu[list(k_list), :], 0))[0]  #都不冲突的选址下标       
#                 if len(k_list) > 0:
#                     k_list_t = torch.tensor(list(k_list), dtype=torch.long, device=self.device)
#                     sub_mask = mask_tabu[k_list_t, :]
#                 else:
#                     sub_mask = mask_tabu
 
#                 all_true_cols = sub_mask.all(0)
    
#                 k_filter = torch.where(all_true_cols)[0]  # 全部为True的列下标集合
                
#                 for k in k_list:
#                     if not bool(mask_tabu[k, target_col]):  # if not TRUE (tabu_table[k, target_col] == 0)
#                         violate_tabu = True      
#                         break  
#                 if violate_tabu:
#                     #print('violate_tabu')
#                     # valid_indices = list(set(range(n_nodes))-(set(facility_lists[i]) | {fac_in[i]}))
#                     # fac_in[i]  = np.random.choice(np.intersect1d(valid_indices, k_filter))
#                     existing_t = facility_lists[i]
#                     exclude_t = torch.cat([existing_t, fac_in_t[i].view(1)])
#                     valid_set_t = all_nodes_t[~torch.isin(all_nodes_t, exclude_t)]
#                     valid_indices = valid_set_t[torch.isin(valid_set_t, k_filter)]
#                     if valid_indices.numel() > 0:
#                         max_fac_idx = torch.argmax(facility_total_values[valid_indices])
#                         fac_in_t[i] = valid_indices[max_fac_idx]
#                 else:
#                     continue

#                 if torch.any(facility_lists[i] == fac_in_t[i]):
#                     valid_indices = k_filter[~torch.isin(k_filter, facility_lists[i])]
#                     if valid_indices.numel() > 0:
#                         max_fac_idx = torch.argmax(facility_total_values[valid_indices])
#                         fac_in_t[i] = valid_indices[max_fac_idx]
#                     else:
#                         continue
#                     #fac_in[i]  = np.random.choice(np.setdiff1d(k_filter, facility_lists[i]))


#             fac_out_index = torch.where(facility_lists == fac_out_t.unsqueeze(1))[1]
#             facility_lists[batch_idx_t, fac_out_index] = fac_in_t

#             # for i in range(self.iter_num):
#             #     # 只在第 i 行中查找 fac_out[i]
#             #     idx = np.where(facility_lists[i] == fac_out[i])[0]
                
#             #     if len(idx) > 0:
#             #         pos = idx[0]  # 取第一个匹配位置
#             #         facility_lists[i, pos] = fac_in[i]
#             #     else:
#             #         print(f"Warning: fac_out[{i}]={fac_out[i]} not found in facility_lists[{i}]")

#             masks[batch_idx_t, fac_out_t] = True
#             masks[batch_idx_t, fac_in_t] = False

#             #换入设施后，根据弱禁忌集更新冲突集
#             for i in range(self.iter_num):
#                 swap_in = int(fac_in_t[i].item())
#                 current_facilities = set(facility_lists[i].tolist())
                
#                 # 检查 swap_in 支配哪些设施（使用弱禁忌表，即原始 tabu_table）
#                 for fac in current_facilities:
#                     if fac != swap_in and not bool(mask_tabu[swap_in, fac]):  # tabu_table[swap_in, fac] == 0
#                         conflict_sets[i].add(fac)

#         #best_facility = None
#         # best_sol=None

#         # facility_list_=facility_list.copy()

#         for i in range(self.iter_num):
            
#             #若最后一步换入冲突，则解不可用
#             if len(conflict_sets[i])>0:
#                 continue

#             # '''
#             # reloc 条件
#             # '''

#             # diff_count = len(set(facility_lists[i]) ^ set(facility_list)) // 2 # 对称差集
#             # print(' diff_count', diff_count)
#             # if diff_count > reloc_step:
#             #     continue



#             # Use precomputed facility_vectors to compute node costs quickly
#             sel_t = facility_lists[i].long()
#             node_costs = facility_vectors[sel_t].sum(dim=0)
#             cost = float(node_costs.sum().item())
#             actual_cost = get_cost(
#                 facility_lists[i].detach().cpu().numpy(),
#                 distance_m,
#                 city_pop,
#                 alpha,
#                 beta,
#             )
#             # allow actual_cost to be tensor or numeric
#             if isinstance(actual_cost, torch.Tensor):
#                 assert torch.isclose(actual_cost, torch.tensor(cost, device=actual_cost.device), rtol=1e-5, atol=1)
#             else:
#                 assert abs(float(actual_cost) - cost) <= 1e-4 * max(1.0, abs(float(actual_cost)))
#             if best_sol is None or cost > getattr(best_sol, "cost", float("-inf")):
#                 best_sol = PMPSolution(facility_lists[i].detach().cpu().numpy(), np.nan, cost)
#                 best_facility=facility_lists[i].detach().cpu().numpy()
        
#         if best_sol is None:
#             best_sol = PMPSolution(
#                 facility_list_t.detach().cpu().numpy(),
#                 np.nan,
#                 get_cost(facility_list_t.detach().cpu().numpy(), distance_m, city_pop, alpha, beta),
#             )
    

#         best_sol.time = time.time() - start
#         # print(city_pop)
#         #print("facility_lists[np.arange(self.iter_num)",facility_lists[np.arange(self.iter_num)])
#         # print('best_facility',best_sol.facility_list)
#         # print(best_sol.cost)
#         # print('reloc_step',reloc_step)
 
#         return (best_sol)




# def run_ppo_swap(
#     dataset, save_path, iter_num, swap_num, init_num, ckpt, device, **kwargs
# ):
#     name = f'ppo_swap_{init_num}_{iter_num}_{swap_num}_{kwargs["name"]}'
#     sol_path = save_path + "/" + name
#     os.makedirs(sol_path, exist_ok=True)
#     print("Running", name)

#     solver = PPOSwapSolver(iter_num, ckpt, device)
#     for batch in dataset:
#         city_id, city_pop, p, distance_m, coordinates, road_net_data, alpha, beta, tabu_table = batch[:9]
#         if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
#             sol = solver.solve(
#                 p,
#                 city_pop,
#                 distance_m,
#                 swap_num,
#                 init_num,
#                 tabu_table,
#                 alpha,
#                 beta,
#                 coordinates=coordinates,
#                 road_net_data=road_net_data,
#             )
#             pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
            
#     return sol_path

# def run_ppo_swap_reloc(dataset, save_path, iter_num, ckpt, device, reloc_coef, **kwargs):
#     name = f'ppo_swap_{iter_num}_{kwargs["name"]}'
#     sol_path = save_path + "/" + name
#     os.makedirs(sol_path, exist_ok=True)
#     print("Running", name)

#     solver = PPOSwapSolver(iter_num, ckpt, device)
#     for batch in dataset:
#         (
#             city_id,
#             city_pop,
#             p,
#             distance_m,
#             coordinates,
#             road_net_data,
#             alpha,
#             beta,
#             tabu_table,
#             facility_list,
#         ) = batch
#         #facility_list = TabuAlphaSampling(exp=1).sample(city_pop, p, tabu_table)
#         #print('facility_list',facility_list)
#         if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
#             sol = solver.solve_reloc(
#                 city_pop,
#                 p,
#                 distance_m,
#                 facility_list,
#                 tabu_table,
#                 alpha,
#                 beta,
#                 int(reloc_coef * p),
#                 coordinates=coordinates,
#                 road_net_data=road_net_data,
#             )
#             pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
         
            
#     return sol_path

# # import os
# # import sys
# # import pickle
# # import time

# # import numpy as np
# # import torch
# # import torch_geometric.data as geom_data

# # #from swap_solver import SwapSolver 
# # from methods.swap_solver import SwapSolver
# # from results import PMPSolution
# # from train import PPOLightning
# # from utils import TabuAlphaSampling,get_cost
 

# # # 获取项目根目录（methods文件夹的父目录）
# # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# # # 将项目根目录加入sys.path
# # sys.path.append(project_root)

# # class PPOSwapSolver(SwapSolver):
# #     def __init__(self, iter_num, ckpt, device):
# #         super().__init__(iter_num)
# #         #self.model = torch.load(ckpt, map_location=device)
# #         self.model = (
# #             PPOLightning.load_from_checkpoint(ckpt, mode="test", weights_only=True ).float().to(device)
# #         )
# #         self.device = device
# #         #self.warm_up()

# #     def warm_up(self):
# #         rand_state = {
# #             "mask": torch.randint(
# #                 0, 2, (self.iter_num, 50), dtype=torch.bool, device=self.device
# #             ),

# #             "tabu_table": torch.randint(
# #             0, 2, (self.iter_num, 50, 50), dtype=torch.bool, device=self.device
# #         ),

# #             "fac_data": geom_data.Batch.from_data_list(
# #                 [
# #                     geom_data.Data(
# #                         x=torch.rand(50, 7, device=self.device),
# #                         edge_index=torch.randint(0, 50, (2, 50), device=self.device),
# #                         edge_attr=torch.rand(20, 1, device=self.device),
# #                     )
# #                     for _ in range(self.iter_num)
# #                 ]
# #             ),
# #         }
# #         with torch.no_grad():
# #             self.model(rand_state)

# #     def _get_fac_data(
# #         self,
# #         city_pop,
# #         p,
# #         distance_m,
# #         facility_list,
# #         static_feat,
# #         road_net_data,
# #         #tabu_table,
# #         alpha,
# #         beta,
# #         mask,
# #         facility_vectors=None,
# #     ):
# #         # If facility_vectors provided (precomputed on device), use it to avoid repeated exp computations
# #         if facility_vectors is None:
# #             wdist = alpha[facility_list].unsqueeze(1) * torch.exp(
# #                 -beta[facility_list].unsqueeze(1) * distance_m[facility_list]
# #             ) * city_pop.unsqueeze(0)
# #         else:
# #             # ensure facility_list is a torch index tensor on the same device
# #             if not torch.is_tensor(facility_list):
# #                 facility_list_t = torch.from_numpy(facility_list).long().to(self.device)
# #             else:
# #                 facility_list_t = facility_list.long().to(self.device)
# #             wdist = facility_vectors[facility_list_t]
# #         #point_indices = torch.argmin(distance_m[facility_list], 0)
# #         node_costs = torch.sum(wdist, dim=0)  #facility to all nodes
# #         total_cost = torch.sum(node_costs)  #objective value


# #         node_feat = torch.cat(
# #             (
# #                 static_feat,
# #                 mask.reshape(-1, 1),
# #                 # node_costs.reshape(-1, 1) / total_cost,
# #                 node_costs.reshape(-1, 1) / torch.max(node_costs),
# #                 #node_fac_feat,
# #             ),
# #             axis=1,
# #         )

# #         fac_data = geom_data.Data(
# #             x=node_feat,
# #             edge_index=road_net_data.edge_index,
# #             edge_attr=road_net_data.edge_attr,
# #         )

# #         return fac_data, total_cost
    

# #     '''
# #     混合版，边预测边启发式
# #     '''
# #     def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table,alpha,beta,reloc_step, **kwargs):
# #         print('facility_list', facility_list)
# #         start = time.time()
# #         best_sol = None

 
# #         city_pop = city_pop.to(self.device)
# #         distance_m = distance_m.to(self.device)
# #         alpha = alpha.to(self.device)
# #         beta = beta.to(self.device)
# #         coordinates = kwargs["coordinates"].to(self.device)
# #         road_net_data = kwargs["road_net_data"].to(self.device)
# #         coordinates_norm = (coordinates - torch.min(coordinates, 0)[0]) / max(
# #             torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0]
# #         )

# #         '''
# #         facility_list先用禁忌表过滤一下，防止生成的初始解有禁忌设施
# #         '''


# #         static_feat = torch.cat(
# #             # (coordinates_norm, city_pop.reshape(-1, 1) / torch.sum(city_pop)),
# #             (coordinates_norm, city_pop.reshape(-1, 1) / torch.max(city_pop),
# #               alpha.reshape(-1, 1) / torch.max(alpha), beta.reshape(-1, 1) / torch.max(beta)
# #               ),
# #             axis=1
# #         )
# #         static_feat = torch.nan_to_num(static_feat, nan=0.0, posinf=1.0, neginf=0.0)
        

# #         '''
# #         '''
# #         # city_pop = city_pop.to(self.device)
# #         # distance_m = distance_m.to(self.device)
# #         # alpha = alpha.to(self.device)
# #         # beta = beta.to(self.device)
# #         # coordinates = kwargs["coordinates"].to(self.device)
# #         # road_net_data = kwargs["road_net_data"].to(self.device)
# #         # coordinates_norm = (coordinates - torch.min(coordinates, 0)[0]) / max(
# #         #     torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0]
# #         # )

# #         # '''
# #         # facility_list先用禁忌表过滤一下，防止生成的初始解有禁忌设施
# #         # '''


# #         # static_feat = torch.cat(
# #         #     # (coordinates_norm, city_pop.reshape(-1, 1) / torch.sum(city_pop)),
# #         #     (coordinates_norm, city_pop.reshape(-1, 1) / torch.max(city_pop),
# #         #       alpha.reshape(-1, 1) / torch.max(alpha), beta.reshape(-1, 1) / torch.max(beta)
# #         #       ),
# #         #     axis=1
# #         # )





# #         facility_lists = np.tile(facility_list, (self.iter_num, 1))
# #         masks = torch.ones(
# #             (self.iter_num, city_pop.shape[0]), dtype=torch.bool, device=self.device
# #         )
# #         masks[:, facility_list] = 0

# #         conflict_sets = [set() for _ in range(self.iter_num)]
# #         tabu_table_batch = tabu_table.to(self.device, dtype=torch.bool).unsqueeze(0).repeat(self.iter_num, 1, 1)
# #         tabu_table_min= np.minimum(tabu_table, tabu_table.T)
# #         mask_tabu = (tabu_table== 1)  # shape: (n_nodes, n_nodes)
# #         mask_tabu_min = (tabu_table_min == 1)  # shape: (n_nodes, n_nodes)
# #         n_nodes=len(tabu_table[0])
  



# #         # Precompute facility influence vectors on device to avoid repeated exp calls
# #         with torch.no_grad():
# #             facility_vectors = (
# #                 alpha.unsqueeze(1) * torch.exp(-beta.unsqueeze(1) * distance_m) * city_pop.unsqueeze(0)
# #             ).to(self.device)
# #             facility_total_values = facility_vectors.sum(dim=1)

# #         for j in range(reloc_step):
# #             for i in range(self.iter_num):
# #                 if len(conflict_sets[i]) > 0:
# #                     # 获取有效设施（排除-1）
# #                     valid_facilities = facility_lists[i][facility_lists[i] >= 0]
# #                     current_facilities = set(valid_facilities)                
# #                     # 按照冲突集的顺序逐个处理
# #                     for conflict_fac in list(conflict_sets[i]):  
# #                         if conflict_fac in current_facilities:
# #                             idx = np.where(facility_lists[i] == conflict_fac)[0]
# #                             if len(idx) > 0:
# #                                 facility_lists[i][idx[0]] = -1  # 使用-1标记
# #                                 masks[i, conflict_fac] = True
# #                                 current_facilities.discard(conflict_fac)
                            
# #                             # 筛选不冲突点位
# #                             if len(current_facilities) > 0:
# #                                 sub_mask = mask_tabu_min[list(current_facilities), :]
# #                                 all_true_cols = sub_mask.all(0)
# #                                 k_filter = np.where(all_true_cols)[0]
# #                             else:
# #                                 k_filter = np.arange(n_nodes)
                            
# #                             # 排除已在解中的设施
# #                             valid_indices = np.setdiff1d(k_filter, list(current_facilities))
                            
# #                             if len(valid_indices) > 0:
# #                                 # 找到不冲突的最大总贡献点位，使用预计算的 facility_total_values
# #                                 valid_idx_t = torch.from_numpy(valid_indices).long().to(self.device)
# #                                 max_fac_idx = torch.argmax(facility_total_values[valid_idx_t])
# #                                 selected_facility = int(valid_indices[max_fac_idx.item()])
                                
# #                                 # 将新设施填入刚才删除的位置
# #                                 empty_slot = np.where(facility_lists[i] == -1)[0][0]  # 找到第一个-1的位置
# #                                 facility_lists[i][empty_slot] = selected_facility
# #                                 masks[i, selected_facility] = False
                                
# #                                 # 更新 current_facilities
# #                                 current_facilities.add(selected_facility)
# #                             else:
# #                                 # 无可用设施，跳过（保持-1）
# #                                 print(f"Warning: Iteration {i}, no valid facility to replace {conflict_fac}")
                    
# #                     # 清空冲突集
# #                     conflict_sets[i].clear()

      
# #                 if -1 in facility_lists[i]:
# #                     facility_lists[i]=facility_list.copy()
# #                     masks[i] = torch.ones(city_pop.shape[0], dtype=torch.bool, device=self.device)
# #                     masks[i, facility_list] = False


# #             fac_data_list = []
# #             for i in range(self.iter_num):
# #                 fac_data, cost = self._get_fac_data(
# #                     city_pop,
# #                     p,
# #                     distance_m,
# #                     facility_lists[i],
# #                     static_feat,
# #                     road_net_data,
# #                     #tabu_table,
# #                     alpha,
# #                     beta,
# #                     masks[i],
# #                     facility_vectors=facility_vectors,
# #                 )


# #                 fac_data_list.append(fac_data)
# #                 if best_sol is None or cost > best_sol.cost:
# #                     best_sol = PMPSolution(facility_lists[i], np.nan, cost)


         
# #             # get_fac_data_time = time.time() - start
# #             # print('get_fac_data_time',get_fac_data_time)
# #             state = {
# #                 "mask": masks,
# #                 "tabu_table": tabu_table_batch,
# #                 "fac_data": geom_data.Batch.from_data_list(fac_data_list),
# #             }
# #             with torch.no_grad():
# #                 action = self.model(state)[1].cpu().numpy()
# #             fac_out = action[:, 0].astype(np.int64)  # int
# #             fac_in = action[:, 1].astype(np.int64)  # int    

# #             filtered_facility_lists =[]             
# #             for row in range(self.iter_num):
# #                 idx = fac_out[row]
# #                 filtered_facility_lists.append(set(facility_lists[row]) -{idx})

# #             # tabu_start = time.time()
# #             # iteration
# #             for i in range(self.iter_num):
# #                 k_list = filtered_facility_lists[i]  
# #                 violate_tabu = False  # 
# #                 target_col = fac_in[i]  # fac_in[i] is checked  
# #                 #k_filter = np.where(np.all(mask_tabu[list(k_list), :], 0))[0]  #都不冲突的选址下标       
# #                 sub_mask = mask_tabu[list(k_list), :]
 
# #                 all_true_cols = sub_mask.all(0)
    
# #                 k_filter = np.where(all_true_cols)[0]  # 全部为True的列下标集合
                
# #                 for k in k_list:
# #                     if not mask_tabu[k, target_col]:  # if not TRUE (tabu_table[k, target_col] == 0)
# #                         violate_tabu = True      
# #                         break  
# #                 if violate_tabu:
# #                     #print('violate_tabu')
# #                     # valid_indices = list(set(range(n_nodes))-(set(facility_lists[i]) | {fac_in[i]}))
# #                     # fac_in[i]  = np.random.choice(np.intersect1d(valid_indices, k_filter))
# #                     valid_set = set(range(n_nodes)) - (set(facility_lists[i]) | {fac_in[i]})
# #                     valid_indices = np.intersect1d(list(valid_set), k_filter)
# #                     if len(valid_indices)>0:
# #                         valid_idx_t = torch.from_numpy(valid_indices).long().to(self.device)
# #                         max_fac_idx = torch.argmax(facility_total_values[valid_idx_t])
# #                         fac_in[i] = int(valid_indices[max_fac_idx.item()])
# #                 else:
# #                     continue

# #                 if fac_in[i] in facility_lists[i]:  
# #                     valid_indices=np.setdiff1d(k_filter, facility_lists[i])
# #                     if valid_indices.size > 0:
# #                         valid_idx_t = torch.from_numpy(valid_indices).long().to(self.device)
# #                         max_fac_idx = torch.argmax(facility_total_values[valid_idx_t])
# #                         fac_in[i] = int(valid_indices[max_fac_idx.item()])
# #                     else:
# #                         continue
# #                     #fac_in[i]  = np.random.choice(np.setdiff1d(k_filter, facility_lists[i]))


# #             fac_out_index = np.where(facility_lists == fac_out[:, None])[1]
# #             facility_lists[np.arange(self.iter_num), fac_out_index] = fac_in

# #             # for i in range(self.iter_num):
# #             #     # 只在第 i 行中查找 fac_out[i]
# #             #     idx = np.where(facility_lists[i] == fac_out[i])[0]
                
# #             #     if len(idx) > 0:
# #             #         pos = idx[0]  # 取第一个匹配位置
# #             #         facility_lists[i, pos] = fac_in[i]
# #             #     else:
# #             #         print(f"Warning: fac_out[{i}]={fac_out[i]} not found in facility_lists[{i}]")

# #             masks[np.arange(self.iter_num), fac_out] = True
# #             masks[np.arange(self.iter_num), fac_in] = False

# #             #换入设施后，根据弱禁忌集更新冲突集
# #             for i in range(self.iter_num):
# #                 swap_in = fac_in[i]
# #                 current_facilities = set(facility_lists[i])
                
# #                 # 检查 swap_in 支配哪些设施（使用弱禁忌表，即原始 tabu_table）
# #                 for fac in current_facilities:
# #                     if fac != swap_in and not mask_tabu[swap_in, fac]:  # tabu_table[swap_in, fac] == 0
# #                         conflict_sets[i].add(fac)

# #         #best_facility = None
# #         # best_sol=None

# #         # facility_list_=facility_list.copy()

# #         for i in range(self.iter_num):
            
# #             #若最后一步换入冲突，则解不可用
# #             if len(conflict_sets[i])>0:
# #                 continue

# #             # '''
# #             # reloc 条件
# #             # '''

# #             # diff_count = len(set(facility_lists[i]) ^ set(facility_list)) // 2 # 对称差集
# #             # print(' diff_count', diff_count)
# #             # if diff_count > reloc_step:
# #             #     continue



# #             # Use precomputed facility_vectors to compute node costs quickly
# #             if not torch.is_tensor(facility_lists[i]):
# #                 sel_t = torch.from_numpy(facility_lists[i]).long().to(self.device)
# #             else:
# #                 sel_t = facility_lists[i].long().to(self.device)
# #             node_costs = facility_vectors[sel_t].sum(dim=0)
# #             cost = float(node_costs.sum().item())
# #             actual_cost = get_cost(facility_lists[i], distance_m, city_pop, alpha, beta)
# #             # allow actual_cost to be tensor or numeric
# #             if isinstance(actual_cost, torch.Tensor):
# #                 assert torch.isclose(actual_cost, torch.tensor(cost, device=actual_cost.device), rtol=1e-5, atol=1)
# #             else:
# #                 assert abs(float(actual_cost) - cost) <= 1e-4 * max(1.0, abs(float(actual_cost)))
# #             if best_sol is None or cost > getattr(best_sol, "cost", float("-inf")):
# #                 best_sol = PMPSolution(facility_lists[i], np.nan, cost)
# #                 best_facility=facility_lists[i]
        
# #         if best_sol is None:
# #             best_sol = PMPSolution(facility_list, np.nan, get_cost(facility_list, distance_m, city_pop, alpha, beta))
    

# #         best_sol.time = time.time() - start
# #         # print(city_pop)
# #         #print("facility_lists[np.arange(self.iter_num)",facility_lists[np.arange(self.iter_num)])
# #         # print('best_facility',best_sol.facility_list)
# #         # print(best_sol.cost)
# #         # print('reloc_step',reloc_step)
 
# #         return (best_sol)




# # def run_ppo_swap(
# #     dataset, save_path, iter_num, swap_num, init_num, ckpt, device, **kwargs
# # ):
# #     name = f'ppo_swap_{init_num}_{iter_num}_{swap_num}_{kwargs["name"]}'
# #     sol_path = save_path + "/" + name
# #     os.makedirs(sol_path, exist_ok=True)
# #     print("Running", name)

# #     solver = PPOSwapSolver(iter_num, ckpt, device)
# #     for batch in dataset:
# #         city_id, city_pop, p, distance_m, coordinates, road_net_data, alpha, beta, tabu_table = batch[:9]
# #         if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
# #             sol = solver.solve(
# #                 p,
# #                 city_pop,
# #                 distance_m,
# #                 swap_num,
# #                 init_num,
# #                 tabu_table,
# #                 alpha,
# #                 beta,
# #                 coordinates=coordinates,
# #                 road_net_data=road_net_data,
# #             )
# #             pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
            
# #     return sol_path

# # def run_ppo_swap_reloc(dataset, save_path, iter_num, ckpt, device, reloc_coef, **kwargs):
# #     name = f'ppo_swap_{iter_num}_{kwargs["name"]}'
# #     sol_path = save_path + "/" + name
# #     os.makedirs(sol_path, exist_ok=True)
# #     print("Running", name)

# #     solver = PPOSwapSolver(iter_num, ckpt, device)
# #     for batch in dataset:
# #         (
# #             city_id,
# #             city_pop,
# #             p,
# #             distance_m,
# #             coordinates,
# #             road_net_data,
# #             alpha,
# #             beta,
# #             tabu_table,
# #             facility_list,
# #         ) = batch
# #         #facility_list = TabuAlphaSampling(exp=1).sample(city_pop, p, tabu_table)
# #         #print('facility_list',facility_list)
# #         if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
# #             sol = solver.solve_reloc(
# #                 city_pop,
# #                 p,
# #                 distance_m,
# #                 facility_list,
# #                 tabu_table,
# #                 alpha,
# #                 beta,
# #                 int(reloc_coef * p),
# #                 coordinates=coordinates,
# #                 road_net_data=road_net_data,
# #             )
# #             pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))
         
            
# #     return sol_path
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


def _resolve_torch_device(device):
    """Resolve runtime device and gracefully fall back to CPU when CUDA is unavailable."""
    req_device = device if isinstance(device, torch.device) else torch.device(device)
    if req_device.type != "cuda":
        return req_device

    if not torch.cuda.is_available():
        print(f"[PPOSwap] CUDA unavailable, fallback to CPU (requested: {req_device}).")
        return torch.device("cpu")

    if req_device.index is not None and req_device.index >= torch.cuda.device_count():
        print(
            f"[PPOSwap] Invalid CUDA index {req_device.index}, "
            f"fallback to cuda:0 (available count={torch.cuda.device_count()})."
        )
        return torch.device("cuda:0")

    return req_device

class PPOSwapSolver(SwapSolver):
    def __init__(self, iter_num, ckpt, device):
        super().__init__(iter_num)
        self.device = _resolve_torch_device(device)
        #self.model = torch.load(ckpt, map_location=device)
        self.model = (
            PPOLightning.load_from_checkpoint(
                ckpt,
                mode="test",
                map_location=self.device,
                weights_only=True,
            )
            .float()
            .to(self.device)
        )
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
        facility_vectors=None,
    ):
        # If facility_vectors provided (precomputed on device), use it to avoid repeated exp computations
        if facility_vectors is None:
            wdist = alpha[facility_list].unsqueeze(1) * torch.exp(
                -beta[facility_list].unsqueeze(1) * distance_m[facility_list]
            ) * city_pop.unsqueeze(0)
        else:
            # ensure facility_list is a torch index tensor on the same device
            if not torch.is_tensor(facility_list):
                facility_list_t = torch.tensor(facility_list, dtype=torch.long, device=self.device)
            else:
                facility_list_t = facility_list.long().to(self.device)
            wdist = facility_vectors[facility_list_t]
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
    

    '''
    混合版，边预测边启发式
    '''
    def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table,alpha,beta,reloc_step, **kwargs):
        print('facility_list', facility_list)
        start = time.time()
        best_sol = None

        def _to_device_tensor(x, dtype=None):
            if torch.is_tensor(x):
                return x.to(self.device, dtype=dtype) if dtype is not None else x.to(self.device)
            return torch.tensor(x, dtype=dtype, device=self.device)

        city_pop = _to_device_tensor(city_pop)
        distance_m = _to_device_tensor(distance_m)
        alpha = _to_device_tensor(alpha)
        beta = _to_device_tensor(beta)
        coordinates = _to_device_tensor(kwargs["coordinates"])
        road_net_data = kwargs["road_net_data"]
        if hasattr(road_net_data, "to"):
            road_net_data = road_net_data.to(self.device)

        if torch.is_tensor(tabu_table):
            tabu_table_t = tabu_table.to(self.device, dtype=torch.bool)
            tabu_table_np = tabu_table.detach().cpu().numpy()
        else:
            tabu_table_np = np.asarray(tabu_table)
            tabu_table_t = torch.tensor(tabu_table_np, dtype=torch.bool, device=self.device)

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
        tabu_table_batch = tabu_table_t.unsqueeze(0).repeat(self.iter_num, 1, 1)
        tabu_table_min= np.minimum(tabu_table_np, tabu_table_np.T)
        mask_tabu = (tabu_table_np == 1)  # shape: (n_nodes, n_nodes)
        mask_tabu_min = (tabu_table_min == 1)  # shape: (n_nodes, n_nodes)
        n_nodes=len(tabu_table_np[0])
  



        # Precompute facility influence vectors on device to avoid repeated exp calls
        with torch.no_grad():
            facility_vectors = (
                alpha.unsqueeze(1) * torch.exp(-beta.unsqueeze(1) * distance_m) * city_pop.unsqueeze(0)
            ).to(self.device)
            facility_total_values = facility_vectors.sum(dim=1)

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
                                # 找到不冲突的最大总贡献点位，使用预计算的 facility_total_values
                                valid_idx_t = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
                                max_fac_idx = torch.argmax(facility_total_values[valid_idx_t])
                                selected_facility = int(valid_indices[max_fac_idx.item()])
                                
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
                    facility_vectors=facility_vectors,
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
                        valid_idx_t = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
                        max_fac_idx = torch.argmax(facility_total_values[valid_idx_t])
                        fac_in[i] = int(valid_indices[max_fac_idx.item()])
                else:
                    continue

                if fac_in[i] in facility_lists[i]:  
                    valid_indices=np.setdiff1d(k_filter, facility_lists[i])
                    if valid_indices.size > 0:
                        valid_idx_t = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
                        max_fac_idx = torch.argmax(facility_total_values[valid_idx_t])
                        fac_in[i] = int(valid_indices[max_fac_idx.item()])
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



            # Use precomputed facility_vectors to compute node costs quickly
            if not torch.is_tensor(facility_lists[i]):
                sel_t = torch.tensor(facility_lists[i], dtype=torch.long, device=self.device)
            else:
                sel_t = facility_lists[i].long().to(self.device)
            node_costs = facility_vectors[sel_t].sum(dim=0)
            cost = float(node_costs.sum().item())
            actual_cost = get_cost(facility_lists[i], distance_m, city_pop, alpha, beta)
            # allow actual_cost to be tensor or numeric
            if isinstance(actual_cost, torch.Tensor):
                assert torch.isclose(actual_cost, torch.tensor(cost, device=actual_cost.device), rtol=1e-5, atol=1)
            else:
                assert abs(float(actual_cost) - cost) <= 1e-4 * max(1.0, abs(float(actual_cost)))
            if best_sol is None or cost > getattr(best_sol, "cost", float("-inf")):
                best_sol = PMPSolution(facility_lists[i], np.nan, cost)
                best_facility=facility_lists[i]
        
        if best_sol is None:
            best_sol = PMPSolution(facility_list, np.nan, get_cost(facility_list, distance_m, city_pop, alpha, beta))
    

        best_sol.time = time.time() - start
        # print(city_pop)
        #print("facility_lists[np.arange(self.iter_num)",facility_lists[np.arange(self.iter_num)])
        print('best_facility',best_sol.facility_list)
        # print(best_sol.cost)
        # print('reloc_step',reloc_step)
 
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