

import os
import pickle
import time

import numpy as np
import torch

from methods.swap_solver import SwapSolver
from results import PMPSolution
from utils import get_cost


class GreedySwapSolver(SwapSolver):
    def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table, alpha, beta, reloc_step, **kwargs):
        start = time.time()
        best_sol = None

        mask = np.ones(city_pop.numel(), dtype=bool)
        mask[facility_list] = 0
        swaps = []
        # print('reloc_step', reloc_step)

        tabu_table_min = np.minimum(tabu_table, tabu_table.T)
        mask_tabu = (tabu_table == 1)  # 弱禁忌表
        mask_tabu_min = (tabu_table_min == 1)  # 强禁忌表
        n_nodes = len(tabu_table)
        conflict_set = set()

        for step in range(reloc_step):
            # ========== 阶段1：检测并处理冲突 ==========

            if len(conflict_set) > 0:
                # print(f"Step {step}: 检测到冲突设施 {conflict_set}")
                
                # 将 facility_list 转换为可以标记 -1 的数组
                facility_list_temp = facility_list.copy()
                
                # 获取有效设施（排除-1）
                valid_facilities = facility_list_temp[facility_list_temp >= 0]
                current_facilities = set(valid_facilities)
                
                # 按照冲突集的顺序逐个处理
                for conflict_fac in list(conflict_set):
                    if conflict_fac in current_facilities:
                        # 步骤1：删除冲突设施
                        idx = np.where(facility_list_temp == conflict_fac)[0]
                        if len(idx) > 0:
                            facility_list_temp[idx[0]] = -1  # 使用-1标记
                            mask[conflict_fac] = True
                            current_facilities.discard(conflict_fac)
                            #print(f"  删除冲突设施 {conflict_fac}")
                        
                        # 步骤2：立即为这个空位选择替换设施
                        # 筛选不冲突点位
                        if len(current_facilities) > 0:
                            sub_mask = mask_tabu_min[list(current_facilities), :]
                            all_true_cols = sub_mask.all(0)
                            k_filter = np.where(all_true_cols)[0]
                        else:
                            # 如果当前没有设施，所有节点都可选
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
                            
                            # 选择收益最大的1个设施
                            max_fac_idx = torch.argmax(total_cost)
                            selected_facility = valid_indices[max_fac_idx.item()]
                            
                            # 将新设施填入刚才删除的位置
                            empty_slot = np.where(facility_list_temp == -1)[0][0]  # 找到第一个-1的位置
                            facility_list_temp[empty_slot] = selected_facility
                            mask[selected_facility] = False
                            
                            # 更新 current_facilities，以便下一个冲突设施的过滤
                            current_facilities.add(selected_facility)
                            #print(f"  替换为设施 {selected_facility}")
                        else:
                            # 无可用设施，跳过（保持-1）
                            #print(f"  ⚠️ 警告: 无可用设施替换 {conflict_fac}")
                            continue
                # 更新 facility_list（移除所有 -1）
                facility_list = facility_list_temp[facility_list_temp >= 0]
                
                # 如果设施数量不足 p，说明有不可行的情况
                if len(facility_list) < p:
                    continue
                    #print(f"  ⚠️ 警告: 冲突处理后设施数量不足 ({len(facility_list)} < {p})，尝试补充")

            
            #阶段二：贪心选择最优换入换出
            fac_in_indices = np.where(mask == 1)[0]
            max_cost = get_cost(facility_list, distance_m, city_pop, alpha, beta)
            best_action = None

            for i, fac_out in enumerate(facility_list):
                # 过滤：与剩余设施不冲突的候选
                filtered_fac_in = []
                for j in fac_in_indices:
                    keep_j = True
                    for k in facility_list:
                        if k != fac_out:
                            if tabu_table[k][j] == 0:
                                keep_j = False
                                break
                    if keep_j:
                        filtered_fac_in.append(j)
                
                # 排除已在设施列表中的
                filtered_fac_in = np.setdiff1d(filtered_fac_in, facility_list)

                if len(filtered_fac_in) == 0:
                    continue

                # 贪心搜索最优换入
                for fac_in in filtered_fac_in:
                    facility_list_ = facility_list.copy()
                    facility_list_[i] = fac_in
                    cost = get_cost(facility_list_, distance_m, city_pop, alpha, beta)
                    if cost > max_cost:
                        max_cost = cost
                        best_action = (fac_out, fac_in)
                    del facility_list_

            if best_action == None:
                print(f"Step {step}: 无更优解，提前终止")
                break

            # 执行最优动作
            fac_out, fac_in = best_action
            facility_list[np.where(facility_list == fac_out)[0]] = fac_in
            mask[fac_in] = 0
            mask[fac_out] = 1
            swaps.append(best_action)
            
            # ========== 阶段3：检测新换入设施是否引入冲突 ==========
            for fac in facility_list:
                if fac != fac_in and not mask_tabu[fac_in, fac]:  # fac_in 支配 fac
                    conflict_set.add(fac)
            
            # if len(new_conflicts) > 0:
            #     print(f"Step {step}: 换入设施 {fac_in} 引入新冲突 {new_conflicts}")
            #     # 下一轮循环会在阶段1处理这些冲突

        best_sol = PMPSolution(facility_list, time.time() - start, max_cost)
        best_sol.swaps = swaps
        print('best_facility', best_sol.facility_list)
        print(best_sol.cost)
        return best_sol


def run_greedy_filter(dataset, save_path, swap_num, init_num, **kwargs):
    name = f"greedy_filter_{init_num}_{swap_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name) 

    solver = GreedySwapSolver(None)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, alpha, beta, tabu_table = batch[:9]
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, swap_num, init_num, tabu_table, alpha, beta, **kwargs)
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path 


def run_greedy_filter_reloc(dataset, save_path, reloc_coef, **kwargs):
    name = "greedy_filter"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = GreedySwapSolver(None)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, alpha, beta, tabu_table, facility_list = batch
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(
                city_pop, p, distance_m, facility_list, tabu_table, alpha, beta, int(reloc_coef * p))

            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path
