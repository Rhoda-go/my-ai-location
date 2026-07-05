import os
import pickle
import time
import numpy as np
import torch

from methods.swap_solver import SwapSolver
from results import PMPSolution
from utils import get_cost, TabuAlphaSampling


class GASolver(SwapSolver):
    """Genetic algorithm solver for placement problems.

    Generates candidate solutions using GA, then repairs them to be feasible
    by detecting conflicts and replacing conflicting facilities using LNS-style logic.
    """

    def solve_reloc(
        self,
        city_pop,
        p,
        distance_m,
        facility_list,
        tabu_table,
        alpha,
        beta,
        reloc_step=10,  # generations
        crossover_rate=0.8,
        mutation_rate=0.2,
        elitism=1,
        **kwargs,
    ):
        start = time.time()
        pop_size = self.iter_num

        # Initialize population
        sampler = TabuAlphaSampling(exp=1)
        population = np.array([sampler.sample(city_pop, p, tabu_table) for _ in range(pop_size)])
        costs = np.array([get_cost(ind, distance_m, city_pop, alpha, beta) for ind in population])

        best_sol = None

        for gen in range(reloc_step):
            new_pop = []

            # Elitism: keep best individuals
            if elitism > 0:
                elite_idx = np.argsort(costs)[-elitism:]
                for idx in elite_idx:
                    new_pop.append(population[idx].copy())

            # Generate rest of population
            while len(new_pop) < pop_size:
                # Select two parents
                p1, p2 = np.random.choice(pop_size, 2, replace=False)
                parent1 = population[p1]
                parent2 = population[p2]

                # Crossover
                if np.random.rand() < crossover_rate:
                    child = self._crossover(parent1, parent2, p, len(city_pop))
                else:
                    child = parent1.copy()

                # Mutation
                if np.random.rand() < mutation_rate:
                    self._mutate(child, len(city_pop), p)

                # Repair: make feasible by resolving conflicts
                child = self._repair(child, city_pop, p, distance_m, tabu_table, alpha, beta)

                new_pop.append(child)

            population = np.array(new_pop)
            costs = np.array([get_cost(ind, distance_m, city_pop, alpha, beta) for ind in population])

            # Update best solution
            best_idx = np.argmax(costs)
            if best_sol is None or costs[best_idx] > best_sol.cost:
                best_sol = PMPSolution(population[best_idx], np.nan, costs[best_idx])

        if best_sol is None:
            best_sol = PMPSolution(
                facility_list,
                np.nan,
                get_cost(facility_list, distance_m, city_pop, alpha, beta),
            )

        best_sol.time = time.time() - start
        print('best_facility', best_sol.facility_list)
        return best_sol

    def _crossover(self, p1, p2, p, n_nodes):
        """Uniform crossover with deduplication."""
        child = np.full(p, -1, dtype=int)
        for i in range(p):
            if np.random.rand() < 0.5:
                child[i] = p1[i]
            else:
                child[i] = p2[i]

        # Deduplicate
        seen = set()
        for i in range(p):
            if child[i] in seen or child[i] == -1:
                available = [x for x in range(n_nodes) if x not in child and x not in seen]
                if available:
                    child[i] = np.random.choice(available)
            seen.add(child[i])
        return child

    def _mutate(self, ind, n_nodes, p):
        """Swap mutation."""
        if np.random.rand() < 0.1:  # mutation probability
            idx = np.random.randint(p)
            available = [x for x in range(n_nodes) if x not in ind]
            if available:
                ind[idx] = np.random.choice(available)

    def _repair(self, facility_list, city_pop, p, distance_m, tabu_table, alpha, beta):
        """Repair by detecting conflicts and replacing using LNS logic."""
        # Detect conflict set
        conflict_set = set()
        for j in facility_list:
            for i in facility_list:
                if i != j and tabu_table[i][j] == 0:
                    conflict_set.add(j)
                    break

        if not conflict_set:
            return facility_list

        # Repair conflicts
        facility_list = np.array(facility_list)
        current_facilities = set(facility_list)
        tabu_table_min = np.minimum(tabu_table, tabu_table.T)
        mask_tabu_min = (tabu_table_min == 1)
        n_nodes = len(tabu_table)

        for conflict_fac in list(conflict_set):
            if conflict_fac not in current_facilities:
                continue
            # Remove conflicting facility
            idx = np.where(facility_list == conflict_fac)[0][0]
            facility_list[idx] = -1
            current_facilities.discard(conflict_fac)

            # Find valid replacements
            if current_facilities:
                sub_mask = mask_tabu_min[list(current_facilities), :]
                all_true_cols = sub_mask.all(0)
                k_filter = np.where(all_true_cols)[0]
            else:
                k_filter = np.arange(n_nodes)

            valid_indices = np.setdiff1d(k_filter, list(current_facilities))

            if valid_indices.size > 0:
                # Select best by cost
                valid_indices_tensor = torch.from_numpy(valid_indices).long()
                cost_matrix = (
                    alpha[valid_indices_tensor].unsqueeze(1)
                    * torch.exp(-beta[valid_indices_tensor].unsqueeze(1) * distance_m[valid_indices_tensor])
                    * city_pop.unsqueeze(0)
                )
                total_cost = cost_matrix.sum(dim=1)
                max_fac_idx = torch.argmax(total_cost)
                selected_facility = valid_indices[max_fac_idx.item()]
            else:
                # No valid, choose random available
                available = np.setdiff1d(np.arange(n_nodes), list(current_facilities))
                if available.size > 0:
                    selected_facility = np.random.choice(available)
                else:
                    # Fallback, but shouldn't happen
                    selected_facility = np.random.randint(n_nodes)

            facility_list[idx] = selected_facility
            current_facilities.add(selected_facility)

        return facility_list


def run_ga(dataset, save_path, iter_num, swap_num, init_num, **kwargs):
    name = f"GA_{init_num}_{iter_num}_{swap_num}"
    sol_path = save_path + "/" + name
    os.makedirs(sol_path, exist_ok=True)
    print("Running", name)

    solver = GASolver(iter_num)
    for batch in dataset:
        city_id, city_pop, p, distance_m, _, _, alpha, beta, tabu_table = batch[:9]
        if not os.path.isfile(f"{sol_path}/{city_id}_{p}.pkl"):
            sol = solver.solve(p, city_pop, distance_m, swap_num, init_num, tabu_table, alpha, beta, **kwargs)
            pickle.dump(sol, open(f"{sol_path}/{city_id}_{p}.pkl", "wb"))

    return sol_path