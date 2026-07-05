import os
import pickle
import time
import numpy as np

from methods.swap_solver import SwapSolver
from results import PMPSolution
from utils import get_cost


class RandomSwapSolver(SwapSolver):
    def solve_reloc(self, city_pop, p, distance_m, facility_list, tabu_table, alpha, beta, reloc_step, **kwargs):
        print('facility_list', facility_list)
        start = time.time()
        best_sol = None

        facility_lists = np.tile(facility_list, (self.iter_num, 1))
        masks = np.ones((self.iter_num, len(city_pop)), dtype=bool)
        masks[:, facility_list] = False

        conflict_sets = [set() for _ in range(self.iter_num)]  # 每个解的冲突集
        tabu_table_min = np.minimum(tabu_table, tabu_table.T)  # 弱禁忌表（对称）
        mask_tabu = (tabu_table == 1)  # 禁忌表的布尔形式
        mask_tabu_min = (tabu_table_min == 1)  # 弱禁忌表的布尔形式
        n_nodes = len(tabu_table[0])

        for j in range(reloc_step):
            for i in range(self.iter_num):
                # 处理冲突集中的设施
                if len(conflict_sets[i]) > 0:
                    # 获取当前解中的有效设施（排除标记为 -1 的设施）
                    valid_facilities = facility_lists[i][facility_lists[i] >= 0]
                    current_facilities = set(valid_facilities)

                    # 按照冲突集的顺序逐个处理冲突设施
                    for conflict_fac in list(conflict_sets[i]):
                        if conflict_fac in current_facilities:
                            # 删除冲突设施
                            idx = np.where(facility_lists[i] == conflict_fac)[0]
                            if len(idx) > 0:
                                facility_lists[i][idx[0]] = -1  # 使用 -1 标记删除的设施
                                masks[i, conflict_fac] = True
                                current_facilities.discard(conflict_fac)

                            # 筛选不冲突的点位
                            if len(current_facilities) > 0:
                                sub_mask = mask_tabu_min[list(current_facilities), :]
                                all_true_cols = sub_mask.all(0)
                                k_filter = np.where(all_true_cols)[0]
                            else:
                                k_filter = np.arange(n_nodes)

                            # 排除已在解中的设施
                            valid_indices = np.setdiff1d(k_filter, list(current_facilities))

                            if len(valid_indices) > 0:
                                # 随机选择一个不冲突的设施
                                selected_facility = np.random.choice(valid_indices)
                                # 将新设施填入刚才删除的位置
                                empty_slot = np.where(facility_lists[i] == -1)[0][0]  # 找到第一个 -1 的位置
                                facility_lists[i][empty_slot] = selected_facility
                                masks[i, selected_facility] = False
                                current_facilities.add(selected_facility)
                            else:
                                # 无可用设施，跳过（保持 -1）
                                print(f"Warning: Iteration {i}, no valid facility to replace {conflict_fac}")

                    # 清空冲突集
                    conflict_sets[i].clear()

                # 如果解中仍然存在 -1（即未填充的设施），重置为初始解
                if -1 in facility_lists[i]:
                    facility_lists[i] = facility_list.copy()
                    masks[i] = np.ones(len(city_pop), dtype=bool)
                    masks[i, facility_list] = False

                # 随机选择换出和换入的设施
                fac_in_indices = np.where(masks[i] == True)[0]  # 可选换入设施
                fac_out_idx = np.random.choice(range(len(facility_lists[i])))  # 随机选择换出设施的索引
                fac_out = facility_lists[i][fac_out_idx]  # 换出的设施

                fac_in = np.random.choice(fac_in_indices)  # 随机选择换入设施
                facility_lists[i][fac_out_idx] = fac_in
                masks[i, fac_in] = False
                masks[i, fac_out] = True

                # 更新冲突集
                swap_in = fac_in
                current_facilities = set(facility_lists[i])

                # 检查换入设施是否引入新的冲突
                for fac in current_facilities:
                    if fac != swap_in and not mask_tabu[swap_in, fac]:  # tabu_table[swap_in, fac] == 0
                        conflict_sets[i].add(fac)

        # 计算最终的最优解
        for i in range(self.iter_num):
            # 如果解中仍然存在冲突，则跳过
            if len(conflict_sets[i]) > 0:
                continue

            # wdist = (
            #     alpha[facility_lists[i]][:, None]
            #     * np.exp(-beta[facility_lists[i]][:, None] * distance_m[facility_lists[i]])
            #     * city_pop[None, :]
            # )
            # node_costs = np.sum(wdist, axis=0)  # 计算设施到所有节点的总收益
            # cost = np.sum(node_costs)  # 总目标值
            cost = get_cost(facility_lists[i], distance_m, city_pop, alpha, beta)
            # assert np.isclose(actual_cost, cost, rtol=1e-5, atol=1)
            if best_sol is None or cost > best_sol.cost:
                best_sol = PMPSolution(facility_lists[i], np.nan, cost)
        if best_sol is None:
            best_sol = PMPSolution(facility_list, np.nan, get_cost(facility_list, distance_m, city_pop, alpha, beta))

        best_sol.time = time.time() - start
        print('best_facility', best_sol.facility_list)
        return best_sol


def run_random(dataset, save_path, iter_num, swap_num, init_num, **kwargs):
    name = f"random_{init_num}_{iter_num}_{swap_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = RandomSwapSolver(iter_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, alpha, beta, tabu_table = batch[:9]
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, swap_num, init_num, tabu_table, alpha, beta, **kwargs)
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path



