import os
import pickle
import torch
from itertools import product
import time

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from results import PMPSolution


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
        tabu_count=0
        for i in facilities:
            for k in facilities:
                if i < k:
                    # 检查条件是否满足
                    # condition1: α_k > α_i * exp(β_k * d_ik)
                    condition1 = alpha[k] > alpha[i] * np.exp(beta[k] * distance_m[k, i])
                    # condition2: β_k ≤ β_i
                    condition2 = beta[k] <= beta[i]

                    if condition1 and condition2:
                        tabu_count+=1                 
                        m.addConstr(select[i] + select[k] <= 1, name=f"Tabu_{i}_{k}")

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
        print('optimal value',m.objVal)


        sol = PMPSolution(
            np.asarray(facility_list, dtype=int), time=runtime, cost=m.objVal
        )

        return sol
    
    def solve_reloc(self, p, reloc_step, current_facility_list, city_pop, distance_m, alpha, beta, **kwargs):
        # print("current_facility_list",current_facility_list)
        # print('distance_m',distance_m)
        # print('alpha',alpha)
        # print('beta',beta)
        # print('city_pop',city_pop)

        gurobi_start=time.time()

        customers = list(range(city_pop.numel()))
        facilities = list(range(city_pop.numel()))

        num_customers = len(customers)
        num_facilities = len(facilities)
        cartesian_prod = list(product(range(num_facilities),range(num_customers)))

        coverage_profit = {}
        for f, c in cartesian_prod:
            coverage_profit[(f, c)] = (
                alpha[f]*torch.exp(-beta[f]*distance_m[facilities[f],customers[c]]) * city_pop[customers[c]]
            )

        # coeff = {}
        # for i in range(num_facilities):
        #     for k in range(num_facilities):
        #         if i != k:
        #             exp_term = np.exp(beta[k] * distance_m[facilities[i], facilities[k]])
        #             # α_i · e^{β_k · d_ik}
        #             coeff[(i, k)] = alpha[i] * exp_term

        m = gp.Model("facility_relocation")
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
      
        tabu_count=0
        #print('facilities',facilities)
        #print('distance_m',distance_m)
        for i in facilities:
            for k in facilities:
                #if i < k:
                    # 检查条件是否满足
                    # condition1: α_k > α_i * exp(β_k * d_ik)
                condition1 = alpha[k] > alpha[i] * torch.exp(beta[k] * distance_m[k, i])

                # condition2: β_k ≤ β_i
                condition2 = beta[k] <= beta[i]

                if condition1 and condition2:
                    tabu_count+=1                 
                    m.addConstr(select[i] + select[k] <= 1, name=f"Tabu_{i}_{k}")
                    
        # Add constraint for relocation budget
        current_facility_vars = {f: select[f] for f in current_facility_list}
        m.addConstr(
            gp.quicksum(1 - current_facility_vars[f] for f in current_facility_list) <= reloc_step,
            name="Relocation_budget",
        )

        m.setObjective(assign.prod(coverage_profit), GRB.MAXIMIZE)

        m.optimize()

        #print("tabu_count",tabu_count)



        if m.status != GRB.OPTIMAL:
            print("Optimization was stopped with status %d" % m.status)
               
    # # condition1&condition2 -> select[i] = 0
        facility_list = []
        for facility in select.keys():
            if select[facility].X == 1:
                facility_list.append(facility)
        
        print("optimal facility", facility_list)
        print('best value',m.ObjVal)

        runtime=time.time()-gurobi_start

        # sol = PMPSolution(
        #     np.asarray(facility_list, dtype=int), time=m.Runtime, cost=m.objVal
        # )

        sol = PMPSolution(
            np.asarray(facility_list, dtype=int), time=runtime, cost=m.objVal
        )
        
        return sol



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
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve_reloc(p, int(reloc_coef * p), facility_list, city_pop, distance_m,alpha,beta, **kwargs)
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path
