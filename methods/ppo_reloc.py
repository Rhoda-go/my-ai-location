
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
        facility_vectors=None,
    ):

        if facility_vectors is None:
            wdist = alpha[facility_list].unsqueeze(1)* torch.exp(-beta[facility_list].unsqueeze(1) * distance_m[facility_list]) * city_pop.unsqueeze(0)
        else:
            facility_list_t = torch.as_tensor(facility_list, dtype=torch.long, device=self.device)
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
    弱禁忌预测,有不可行解
    '''
    def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table,alpha,beta,reloc_step, **kwargs):
        print('facility_list',facility_list)
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

        if torch.is_tensor(facility_list):
            facility_list_np = facility_list.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            facility_list_np = np.asarray(facility_list, dtype=np.int64)

        if torch.is_tensor(tabu_table):
            tabu_table_np = tabu_table.detach().cpu().numpy()
            tabu_table_t = tabu_table.to(self.device, dtype=torch.bool)
        else:
            tabu_table_np = np.asarray(tabu_table)
            tabu_table_t = torch.as_tensor(tabu_table_np, dtype=torch.bool, device=self.device)

        facility_lists = np.tile(facility_list_np, (self.iter_num, 1))
        facility_list_t = torch.as_tensor(facility_list_np, dtype=torch.long, device=self.device)
        masks = torch.ones(
            (self.iter_num, city_pop.shape[0]), dtype=torch.bool, device=self.device
        )
        masks[:, facility_list_t] = 0

      
        tabu_table_batch = tabu_table_t.unsqueeze(0).expand(self.iter_num, -1, -1)

        tabu_table_min = np.minimum(tabu_table_np, tabu_table_np.T)
        mask_tabu = (tabu_table_np == 1)
        n_nodes = tabu_table_np.shape[0]
        all_nodes = np.arange(n_nodes, dtype=np.int64)
        batch_idx = np.arange(self.iter_num)

        with torch.no_grad():
            facility_vectors = (
                alpha.unsqueeze(1)
                * torch.exp(-beta.unsqueeze(1) * distance_m)
                * city_pop.unsqueeze(0)
            )
            facility_total_values = facility_vectors.sum(dim=1)

        static_batch = static_feat.unsqueeze(0).expand(self.iter_num, -1, -1)
        feat_dim = static_feat.shape[1] + 2
        fac_data_batch = geom_data.Batch.from_data_list(
            [
                geom_data.Data(
                    x=torch.zeros((city_pop.shape[0], feat_dim), device=self.device),
                    edge_index=road_net_data.edge_index,
                    edge_attr=road_net_data.edge_attr,
                )
                for _ in range(self.iter_num)
            ]
        )

        for j in range(reloc_step):
            facility_lists_t = torch.as_tensor(facility_lists, dtype=torch.long, device=self.device)
            wdist_batch = facility_vectors[facility_lists_t]
            node_costs_batch = torch.sum(wdist_batch, dim=1)
            total_costs = torch.sum(node_costs_batch, dim=1)

            best_idx = torch.argmax(total_costs)
            best_cost = total_costs[best_idx]
            if best_sol is None or best_cost > best_sol.cost:
                best_sol = PMPSolution(facility_lists[int(best_idx.item())], np.nan, best_cost)

            norm_den = torch.max(node_costs_batch, dim=1, keepdim=True)[0].clamp(min=1e-12)
            norm_node = (node_costs_batch / norm_den).unsqueeze(2)
            node_feat = torch.cat((static_batch, masks.reshape(self.iter_num, -1, 1), norm_node), dim=2)
            fac_data_batch.x = node_feat.reshape(-1, feat_dim)
         
            # get_fac_data_time = time.time() - start
            # print('get_fac_data_time',get_fac_data_time)

            state = {
                "mask": masks,
                "tabu_table": tabu_table_batch,
                "fac_data": fac_data_batch,
            }

            with torch.no_grad():
                action = self.model(state)[1].cpu().numpy()
            filtered_facility_lists =[] 
            fac_out = action[:, 0].astype(np.int64)  # int
            fac_in = action[:, 1].astype(np.int64)  # int    
            
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
                    if len(valid_indices) > 0:
                        valid_indices_tensor = torch.as_tensor(valid_indices, dtype=torch.long, device=self.device)
                        max_fac_idx = torch.argmax(facility_total_values[valid_indices_tensor])
                        fac_in[i] = valid_indices[max_fac_idx.item()]


                if fac_in[i] in facility_lists[i]:  
                    valid_indices=np.setdiff1d(k_filter, facility_lists[i])
                    if len(valid_indices) > 0:
                        valid_indices_tensor = torch.as_tensor(valid_indices, dtype=torch.long, device=self.device)
                        max_fac_idx = torch.argmax(facility_total_values[valid_indices_tensor])
                        fac_in[i] = valid_indices[max_fac_idx.item()]

                    #fac_in[i]  = np.random.choice(np.setdiff1d(k_filter, facility_lists[i]))
               
            # tabu_time=time.time()-tabu_start
            # print('tabu_time',tabu_time)

            # print('distance_m',distance_m)
            # print('alpha',alpha)
            # print('beta',beta)


            fac_out_index = np.where(facility_lists == fac_out[:, None])[1]
            facility_lists[batch_idx, fac_out_index] = fac_in

            # for i in range(self.iter_num):
            #     # 只在第 i 行中查找 fac_out[i]
            #     idx = np.where(facility_lists[i] == fac_out[i])[0]
                
            #     if len(idx) > 0:
            #         pos = idx[0]  # 取第一个匹配位置
            #         facility_lists[i, pos] = fac_in[i]
            #     else:
            #         print(f"Warning: fac_out[{i}]={fac_out[i]} not found in facility_lists[{i}]")

            masks[batch_idx, fac_out] = True
            masks[batch_idx, fac_in] = False

        # print('city_pop',city_pop)

        '''
        增加最后是否被支配的检测
        '''

        # print('facility_lists',facility_lists)
        valid_mask = np.ones(self.iter_num, dtype=bool)

        for i in range(self.iter_num):
            check_set = np.setdiff1d(facility_lists[i], facility_list_np)
            # print('check_set',check_set)
            if len(check_set) > 0:
                # 快速检查是否违反禁忌
                tabu_submatrix = tabu_table_np[np.ix_(facility_lists[i], check_set)]
                # 确保使用numpy的any
                if (tabu_submatrix == 0).any():
                    valid_mask[i] = False
        # print(valid_mask)
        #best_facility = None
        best_sol=None

        for i in range(self.iter_num):

            if not valid_mask[i]:
                continue  # 跳过违反禁忌的解


            #wdist = alpha* torch.exp(-beta * distance_m[facility_lists[i]]) * city_pop
            sel_t = torch.as_tensor(facility_lists[i], dtype=torch.long, device=self.device)
            wdist = facility_vectors[sel_t]
            # print(facility_lists[i])
            # print(alpha.shape,beta.shape,len(facility_lists[i]),city_pop.shape)
            #point_indices = torch.argmin(distance_m[facility_list], 0)
            node_costs = torch.sum(wdist, dim=0)  #facility to all nodes
            cost = torch.sum(node_costs)  #objective value
            if best_sol is None or cost > best_sol.cost:
                best_sol = PMPSolution(facility_lists[i], np.nan, cost)
                best_facility=facility_lists[i]

        if best_sol is None:
            fallback_sel = torch.as_tensor(facility_list_np, dtype=torch.long, device=self.device)
            fallback_cost = torch.sum(facility_vectors[fallback_sel].sum(dim=0))
            best_sol = PMPSolution(facility_list_np, np.nan, fallback_cost)
        
        best_sol.time = time.time() - start
        # print(city_pop)
        #print("facility_lists[np.arange(self.iter_num)",facility_lists[np.arange(self.iter_num)])
        print('best_facility',best_sol.facility_list)
        print(best_sol.cost)
 
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

def run_ppo_reloc(dataset, save_path, iter_num, ckpt, device, reloc_coef, **kwargs):
    name = f'ppo_reloc_{iter_num}_{kwargs["name"]}'
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
#     ):

#         wdist = alpha[facility_list].unsqueeze(1)* torch.exp(-beta[facility_list].unsqueeze(1) * distance_m[facility_list]) * city_pop.unsqueeze(0)
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
#     弱禁忌预测,有不可行解
#     '''
#     def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table, alpha, beta, reloc_step, **kwargs):
#         """Relocation solver that tries to improve `facility_list`.

#         The original implementation relied heavily on NumPy loops and
#         conversions between NumPy and torch.  This rewritten version keeps
#         all of the computation on the device, vectorizes the inner loops, and
#         avoids repeatedly rebuilding tensors that can be reused across
#         iterations.  The result is a significant reduction in Python overhead
#         when `iter_num` or `reloc_step` is large.
#         """

#         print("facility_list", facility_list)
#         start = time.time()

#         # push everything to the right device up front
#         city_pop = city_pop.to(self.device)
#         distance_m = distance_m.to(self.device)
#         alpha = alpha.to(self.device)
#         beta = beta.to(self.device)
#         coordinates = kwargs["coordinates"].to(self.device)
#         road_net_data = kwargs["road_net_data"].to(self.device)

#         # normalise coords and build static node features
#         coordinates_norm = (coordinates - torch.min(coordinates, 0)[0]) / (
#             torch.max(coordinates, 0)[0] - torch.min(coordinates, 0)[0]
#         ).clamp(min=1e-12)
#         static_feat = torch.cat(
#             (
#                 coordinates_norm,
#                 city_pop.reshape(-1, 1) / city_pop.max(),
#                 alpha.reshape(-1, 1) / alpha.max(),
#                 beta.reshape(-1, 1) / beta.max(),
#             ),
#             dim=1,
#         )

#         n_nodes = city_pop.shape[0]

#         # work with torch tensors rather than NumPy arrays throughout
#         facility_list = torch.as_tensor(facility_list, dtype=torch.long, device=self.device)
#         facility_lists = facility_list.unsqueeze(0).repeat(self.iter_num, 1)  # (iter_num, p)

#         masks = torch.ones((self.iter_num, n_nodes), dtype=torch.bool, device=self.device)
#         masks[torch.arange(self.iter_num, device=self.device).unsqueeze(1), facility_lists] = False

#         tabu_dev = tabu_table.to(self.device, dtype=torch.bool)
#         tabu_batch = tabu_dev.unsqueeze(0).repeat(self.iter_num, 1, 1)

#         # precompute vectors used in cost calculations
#         with torch.no_grad():
#             facility_vectors = (
#                 alpha.unsqueeze(1)
#                 * torch.exp(-beta.unsqueeze(1) * distance_m)
#                 * city_pop.unsqueeze(0)
#             )  # (n_nodes, n_nodes)
#             facility_total_values = facility_vectors.sum(dim=1)  # (n_nodes,)

#         membership = torch.zeros((self.iter_num, n_nodes), dtype=torch.bool, device=self.device)
#         membership.scatter_(1, facility_lists, True)

#         best_sol = None

#         rows = torch.arange(self.iter_num, device=self.device)

#         for _ in range(reloc_step):
#             # compute current costs in a fully vectorised way
#             wdist = facility_vectors[facility_lists]  # (iter_num, p, n_nodes)
#             node_costs = wdist.sum(dim=1)  # (iter_num, n_nodes)
#             total_costs = node_costs.sum(dim=1)  # (iter_num,)

#             # record best solution seen so far
#             iter_max, iter_idx = total_costs.max(dim=0)
#             if best_sol is None or iter_max.item() > best_sol.cost:
#                 best_sol = PMPSolution(
#                     facility_lists[iter_idx].cpu().numpy().copy(),
#                     np.nan,
#                     iter_max.item(),
#                 )

#             # prepare input for the model
#             max_node = node_costs.max(dim=1, keepdim=True)[0]
#             norm_node = node_costs / (max_node + 1e-12)
#             static_batch = static_feat.unsqueeze(0).expand(self.iter_num, -1, -1)
#             mask_float = masks.float().unsqueeze(2)
#             node_feat = torch.cat((static_batch, mask_float, norm_node.unsqueeze(2)), dim=2)

#             fac_data = geom_data.Batch.from_data_list(
#                 [
#                     geom_data.Data(
#                         x=node_feat[i],
#                         edge_index=road_net_data.edge_index,
#                         edge_attr=road_net_data.edge_attr,
#                     )
#                     for i in range(self.iter_num)
#                 ]
#             )

#             state = {"mask": masks, "tabu_table": tabu_batch, "fac_data": fac_data}
#             with torch.no_grad():
#                 action_t = self.model(state)[1]  # keep on device

#             fac_out = action_t[:, 0].long()
#             fac_in = action_t[:, 1].long()

#             # compute mask of selected facilities excluding the one being moved
#             membership_except = membership.clone()
#             membership_except[rows, fac_out] = False

#             # allowed columns = those that are permitted by all remaining facilities
#             prohib = ~tabu_dev  # True means prohibited
#             combined = prohib.unsqueeze(0) & membership_except.unsqueeze(2)
#             invalid_cols = combined.any(dim=1)  # (iter_num, n_nodes)
#             k_filter_mask = ~invalid_cols

#             # fix proposals that violate tabu or select an already‑chosen node
#             invalid_in = (~k_filter_mask[rows, fac_in]) | membership[rows, fac_in]
#             if invalid_in.any():
#                 idxs = torch.nonzero(invalid_in, as_tuple=False).squeeze(1)
#                 cand_mask = k_filter_mask[idxs] & ~membership[idxs]
#                 vals = facility_total_values.unsqueeze(0).expand(idxs.size(0), -1).clone()
#                 vals[~cand_mask] = -float("inf")
#                 _, new_pos = vals.max(dim=1)
#                 fac_in[idxs] = new_pos

#             # perform the relocation
#             cols = torch.argmax((facility_lists == fac_out.unsqueeze(1)).long(), dim=1)
#             facility_lists[rows, cols] = fac_in
#             membership[rows, fac_out] = False
#             membership[rows, fac_in] = True
#             masks[rows, fac_out] = True
#             masks[rows, fac_in] = False

#         if best_sol is None:
#             best_sol = PMPSolution(
#                 facility_list.cpu().numpy().copy(),
#                 np.nan,
#                 get_cost(facility_list.cpu().numpy(), distance_m, city_pop, alpha, beta),
#             )

#         best_sol.time = time.time() - start
#         return best_sol




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

# def run_ppo_reloc(dataset, save_path, iter_num, ckpt, device, reloc_coef, **kwargs):
#     name = f'ppo_reloc_{iter_num}_{kwargs["name"]}'
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