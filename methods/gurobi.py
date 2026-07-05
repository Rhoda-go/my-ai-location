import os
import pickle
import torch
from itertools import product
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from results import PMPSolution
from utils import TabuAlphaSampling


class GurobiSolver:
    def __init__(self):
        pass

    def solve(self, p, city_pop, distance_m, alpha, beta ,**kwargs):
        gurobi_start=time.time()
        customers = list(range(city_pop.numel()))
        facilities = list(range(city_pop.numel()))

        num_customers = len(customers)
        num_facilities = len(facilities)
        cartesian_prod = list(product(range(num_facilities),range(num_customers)))

        coverage_profit = {}
        for f, c in cartesian_prod:
            coverage_profit[(f, c)] = (
                alpha[f]*np.exp(-beta[f]*distance_m[facilities[f],customers[c]]) * city_pop[customers[c]]
            )

        # coeff = {}
        # for i in range(num_facilities):
        #     for k in range(num_facilities):
        #         if i != k:
        #             exp_term = np.exp(beta[k] * distance_m[facilities[i], facilities[k]])
        #             # α_i · e^{β_k · d_ik}
        #             coeff[(i, k)] = alpha[i] * exp_term

        m = gp.Model("facility_location")
        for param, param_val in kwargs.items():
            m.setParam(param, param_val)

        select = m.addVars(num_facilities, vtype=GRB.BINARY, name="Select")
        assign = m.addVars(cartesian_prod, vtype=GRB.BINARY, name="Assign")

        m.addConstr(select.sum() == p, name="Facility_limit")
        m.addConstrs(
            (assign[(f, c)] <= select[f] for f, c in cartesian_prod), name="Setup2ship"
        )
        # m.addConstrs(
        #     (
        #         gp.quicksum(assign[(c, f)] for f in range(num_facilities)) == 1
        #         for c in range(num_customers)
        #     ),
        #     name="Demand",
        # )
        # tabu_count=0
        # for i in facilities:
        #     for k in facilities:
    
        #         # condition1: α_k > α_i * exp(β_k * d_ik)
        #         condition1 = alpha[k] > alpha[i] * np.exp(beta[k] * distance_m[k, i])
        #         # condition2: β_k ≤ β_i
        #         condition2 = beta[k] <= beta[i]

        #         if condition1 and condition2:
        #             tabu_count+=1                 
        #             m.addConstr(select[i] + select[k] <= 1, name=f"Tabu_{i}_{k}")

        m.setObjective(assign.prod(coverage_profit), GRB.MAXIMIZE)

        m.optimize()


        if m.status != GRB.OPTIMAL:
            print("Optimization was stopped with status %d" % m.status)

        facility_list = []
        for facility in select.keys():
            if select[facility].X == 1:
                facility_list.append(facility)
        
        runtime=time.time()-gurobi_start

        print('facility_list',facility_list)
        # print('optimal value',m.objVal)


        sol = PMPSolution(
            np.asarray(facility_list, dtype=int), time=runtime, cost=m.objVal
        )

        return sol
    
    def solve_reloc(self, p, reloc_step, current_facility_list, city_pop, distance_m, alpha, beta, **kwargs):
        # print('current_facility_list', current_facility_list)
        print('reloc_step', reloc_step)

        gurobi_start = time.time()

        customers = list(range(city_pop.numel()))
        facilities = list(range(city_pop.numel()))
        num_customers = len(customers)
        num_facilities = len(facilities)
        cartesian_prod = list(product(range(num_facilities), range(num_customers)))

        # 计算覆盖收益
        coverage_profit = {}
        for f, c in cartesian_prod:
            coverage_profit[(f, c)] = (
                alpha[f] * torch.exp(-beta[f] * distance_m[facilities[f], customers[c]]) * city_pop[customers[c]]
            )

        # 创建模型
        m = gp.Model("facility_relocation")
        for param, param_val in kwargs.items():
            m.setParam(param, param_val)

        # 决策变量
        select = m.addVars(num_facilities, vtype=GRB.BINARY, name="Select")
        assign = m.addVars(cartesian_prod, vtype=GRB.BINARY, name="Assign")

        # 辅助变量：标记换入的设施
        swap_in = m.addVars(num_facilities, vtype=GRB.BINARY, name="SwapIn")

        # 基本约束
        m.addConstr(select.sum() == p, name="Facility_limit")
        m.addConstrs(
            (assign[(f, c)] <= select[f] for f, c in cartesian_prod), 
            name="Setup2ship"
        )

        # 定义换入变量
        current_facility_set = set(current_facility_list)
        for f in facilities:
            if f in current_facility_set:
                # 原本在解中的设施，不是换入的
                m.addConstr(swap_in[f] == 0, name=f"SwapIn_def_{f}")
            else:
                # 原本不在解中的设施，如果被选中就是换入的
                m.addConstr(swap_in[f] == select[f], name=f"SwapIn_def_{f}")

        # 换出数量限制
        m.addConstr(
            gp.quicksum(1 - select[f] for f in current_facility_list) <= reloc_step,
            name="Relocation_budget"
        )

        # 禁忌约束：换入的设施不能被任何选中的设施支配
        # tabu_count = 0
        # for i in facilities:
        #     if i in current_facility_set:
        #         continue  # i 在当前解中，不是换入的设施，跳过
            
        #     for k in facilities:
        #         if i == k:
        #             continue
                
        #         # 检查 k 是否支配 i
        #         condition1 = alpha[k] > alpha[i] * torch.exp(beta[k] * distance_m[k, i])
        #         condition2 = beta[k] <= beta[i]

        #         if condition1 and condition2:
        #             # 如果 k 支配 i，则：
        #             # - 如果 i 被换入（swap_in[i]=1），则 k 不能被选中（select[k]=0）
        #             # 等价于：swap_in[i] + select[k] <= 1
        #             m.addConstr(
        #                 swap_in[i] + select[k] <= 1,
        #                 name=f"Tabu_{k}_dominates_swapin_{i}"
        #             )
        #             tabu_count += 1

        # print(f"Added {tabu_count} tabu constraints")

        # 目标函数
        m.setObjective(assign.prod(coverage_profit), GRB.MAXIMIZE)

        # 优化
        m.optimize()

        if m.status != GRB.OPTIMAL:
            print(f"Optimization was stopped with status {m.status}")
            return None

        # 提取解
        facility_list = [f for f in facilities if select[f].X > 0.5]
        
        # print("Current facility:", current_facility_list)
        print("Optimal facility:", facility_list)
        # print("Best value:", m.ObjVal)
        
        # 统计换出的设施
        swapped_out = [f for f in current_facility_list if f not in facility_list]
        swapped_in = [f for f in facility_list if f not in current_facility_list]
        # print(f"Swapped out ({len(swapped_out)}): {swapped_out}")
        # print(f"Swapped in ({len(swapped_in)}): {swapped_in}")
        
        # 验证换入的设施不被支配
        for i in swapped_in:
            for k in facility_list:
                if i == k:
                    continue
                condition1 = alpha[k] > alpha[i] * torch.exp(beta[k] * distance_m[k, i])
                condition2 = beta[k] <= beta[i]
                # if condition1 and condition2:
                #     print(f"⚠️ Warning: Swapped-in facility {i} is dominated by {k}")

        runtime = time.time() - gurobi_start

        sol = PMPSolution(
            np.asarray(facility_list, dtype=int), 
            time=runtime, 
            cost=m.ObjVal
        )
        
        return sol
    
    # def solve_reloc(self, p, reloc_step, current_facility_list, city_pop, distance_m, alpha, beta, **kwargs):
    #     # print("current_facility_list",current_facility_list)
    #     # print('distance_m',distance_m)
    #     # print('alpha',alpha)
    #     # print('beta',beta)
    #     # print('city_pop',city_pop)
    #     print('current_facility_list',current_facility_list)

    #     # reloc_step=round(float(reloc_step))
    #     print('reloc_step',reloc_step)


    #     gurobi_start=time.time()

    #     customers = list(range(city_pop.numel()))
    #     facilities = list(range(city_pop.numel()))

    #     num_customers = len(customers)
    #     num_facilities = len(facilities)
    #     cartesian_prod = list(product(range(num_facilities),range(num_customers)))

    #     coverage_profit = {}
    #     for f, c in cartesian_prod:
    #         coverage_profit[(f, c)] = (
    #             alpha[f]*torch.exp(-beta[f]*distance_m[facilities[f],customers[c]]) * city_pop[customers[c]]
    #         )

    #     # coeff = {}
    #     # for i in range(num_facilities):
    #     #     for k in range(num_facilities):
    #     #         if i != k:
    #     #             exp_term = np.exp(beta[k] * distance_m[facilities[i], facilities[k]])
    #     #             # α_i · e^{β_k · d_ik}
    #     #             coeff[(i, k)] = alpha[i] * exp_term

    #     m = gp.Model("facility_relocation")
    #     for param, param_val in kwargs.items():
    #         m.setParam(param, param_val)

    #     select = m.addVars(num_facilities, vtype=GRB.BINARY, name="Select")
    #     assign = m.addVars(cartesian_prod, vtype=GRB.BINARY, name="Assign")

    #     m.addConstr(select.sum() == p, name="Facility_limit")
    #     m.addConstrs(
    #         (assign[(f, c)] <= select[f] for f, c in cartesian_prod), name="Setup2ship"
    #     )

    #     # m.addConstrs(
    #     #     (
    #     #         gp.quicksum(assign[(c, f)] for f in range(num_facilities)) == 1
    #     #         for c in range(num_customers)
    #     #     ),
    #     #     name="Demand",
    #     # )
      
    #     tabu_count=0
    #     #print('facilities',facilities)
    #     #print('distance_m',distance_m)
    #     for i in facilities:
    #         for k in facilities:
    
    #                 # condition1: α_k > α_i * exp(β_k * d_ik)
    #             condition1 = alpha[k] > alpha[i] * torch.exp(beta[k] * distance_m[k, i])

    #             # condition2: β_k ≤ β_i
    #             condition2 = beta[k] <= beta[i]

    #             if condition1 and condition2:
    #                 tabu_count+=1                 
    #                 m.addConstr(select[i] + select[k] <= 1, name=f"Tabu_{i}_{k}")
                    
    #     # Add constraint for relocation budget
    #     current_facility_vars = {f: select[f] for f in current_facility_list}
    #     m.addConstr(
    #         gp.quicksum(1 - current_facility_vars[f] for f in current_facility_list) <= reloc_step,
    #         name="Relocation_budget",
    #     )

    #     m.setObjective(assign.prod(coverage_profit), GRB.MAXIMIZE)

    #     m.optimize()

    #     #print("tabu_count",tabu_count)



    #     if m.status != GRB.OPTIMAL:
    #         print("Optimization was stopped with status %d" % m.status)
               
    # # # condition1&condition2 -> select[i] = 0
    #     facility_list = []
    #     for facility in select.keys():
    #         if select[facility].X == 1:
    #             facility_list.append(facility)
        
    #     print("optimal facility", facility_list)
    #     print('best value',m.ObjVal)

    #     runtime=time.time()-gurobi_start

    #     # sol = PMPSolution(
    #     #     np.asarray(facility_list, dtype=int), time=m.Runtime, cost=m.objVal
    #     # )

    #     sol = PMPSolution(
    #         np.asarray(facility_list, dtype=int), time=runtime, cost=m.objVal
    #     )
        
    #     return sol



def run_gurobi(dataset, save_path, **kwargs):
    if "TimeLimit" not in kwargs:
        name = f"gurobi_optimal"
    else:
        name = f'gurobi_{kwargs["TimeLimit"]}'
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = GurobiSolver()
    for batch in dataset:
        #city_id, city_pop, p, distance_m, alpha, beta = batch[:6]
        city_id, city_pop, p, distance_m, coordinates, road_net_data, alpha, beta, tabu_table = batch[:9]
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, alpha, beta, **kwargs)
            #sol.eval(city_pop, distance_m)
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path

def run_gurobi_reloc(dataset, save_path, reloc_coef, **kwargs):
    if "TimeLimit" not in kwargs:
        name = f"gurobi_optimal"
    else:
        name = f'gurobi_{kwargs["TimeLimit"]}'
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = GurobiSolver()
    for batch in dataset:
        #city_id, city_pop, p, distance_m, _, _,alpha,beta, facility_list = batch
        city_id, city_pop, p, distance_m, coordinates, road_net_data, alpha, beta, tabu_table, facility_list = batch
        #facility_list=facility_list = TabuAlphaSampling(exp=1).sample(city_pop, p, tabu_table)
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(p, int(reloc_coef * p), facility_list, city_pop, distance_m,alpha,beta, **kwargs)
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path
