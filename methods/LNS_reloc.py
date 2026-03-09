import os
import pickle
import time
import numpy as np
import torch

from methods.swap_solver import SwapSolver
from results import PMPSolution
from utils import get_cost

class RandomSwapSolver(SwapSolver):
     def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table, alpha, beta, reloc_step, **kwargs):
        print('facility_list', facility_list)
        start = time.time()
        best_sol = None

        # 初始化解
        facility_lists = np.tile(facility_list, (self.iter_num, 1))
        masks = np.ones((self.iter_num, len(city_pop)), dtype=bool)
        masks[:, facility_list] = False

        tabu_table_min = np.minimum(tabu_table, tabu_table.T)  # 弱禁忌表
        mask_tabu = (tabu_table == 1)  # 禁忌表的布尔形式
        n_nodes = len(tabu_table[0])

        for j in range(reloc_step):
            for i in range(self.iter_num):
                # 随机选择换出的设施
                fac_out_idx = np.random.choice(range(len(facility_lists[i])))
                fac_out = facility_lists[i][fac_out_idx]

                sub_mask = mask_tabu[list(facility_lists[i]), :]
 
                all_true_cols = sub_mask.all(0)
    
                k_filter = np.where(all_true_cols)[0]  # 全部为True的列下标集合

                valid_indices = np.setdiff1d(k_filter, list(facility_lists[i]))

                if len(valid_indices) > 0:
                    # 使用大邻域搜索选择收益最大的设施
                    valid_indices_tensor = torch.from_numpy(valid_indices).long()
                    cost_matrix = (
                        alpha[valid_indices_tensor].unsqueeze(1)
                        * torch.exp(-beta[valid_indices_tensor].unsqueeze(1) * distance_m[valid_indices_tensor])
                        * city_pop.unsqueeze(0)
                    )
                    total_cost = cost_matrix.sum(dim=1)

                    # 选择收益最大的设施
                max_fac_idx = torch.argmax(total_cost)
                fac_in = valid_indices[max_fac_idx.item()]

                # 替换设施
                facility_lists[i][fac_out_idx] = fac_in
                masks[i, fac_out] = True
                masks[i, fac_in] = False

        # 最后是否被支配检测
        valid_mask = np.ones(self.iter_num, dtype=bool)
  
        for i in range(self.iter_num):
            check_set = np.setdiff1d(facility_lists[i], facility_list)
            if len(check_set) > 0:
                # 快速检查是否违反禁忌
                tabu_submatrix = tabu_table[np.ix_(facility_lists[i], check_set)]
                if (tabu_submatrix == 0).any():  # 如果有违反禁忌的情况
                    valid_mask[i] = False
        best_sol=None
        # 计算最终的最优解
        for i in range(self.iter_num):
            if not valid_mask[i]:  # 跳过违反禁忌的解
                continue

            cost = get_cost(facility_lists[i], distance_m, city_pop, alpha, beta)
            # assert np.isclose(actual_cost, cost, rtol=1e-5, atol=1)
            if best_sol is None or cost > best_sol.cost:
                best_sol = PMPSolution(facility_lists[i], np.nan, cost)
        if best_sol is None:
            best_sol = PMPSolution(facility_list, np.nan, get_cost(facility_list, distance_m, city_pop, alpha, beta))


        best_sol.time = time.time() - start
        return best_sol





def run_LNS_reloc(dataset, save_path, iter_num, reloc_coef, **kwargs):
    name = f"LNS_reloc_{iter_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = RandomSwapSolver(iter_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, alpha, beta, tabu_table, facility_list = batch
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(
                city_pop, p, distance_m, facility_list, tabu_table, alpha, beta, int(reloc_coef * p)
            )
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path