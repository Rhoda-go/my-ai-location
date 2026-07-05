import multiprocessing as mp
import os
import pickle
import time

import networkx as nx
import numpy as np


def gen_gabriel_graph(data_path, seed, n, k1=3, k2=6):

    #SCALE_FACTOR = 5.0  # 1×1→10km×10km
    #SCALE_FACTOR = 8.0  # 1×1→10km×10km 100 200 nodes
    SCALE_FACTOR = 10.0  # 1×1→10km×10km 500 nodes
    #SCALE_FACTOR = 15.0  # 1×1→10km×10km 1000 nodes


    class Edge:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class EdgeIntersector:
        def __init__(self):
            self.edges = []

        def add_edge(self, new_edge: Edge) -> bool:
            for edge in self.edges:
                if self.intersects(edge, new_edge):
                    return False
            self.edges.append(new_edge)
            return True

        def intersects(self, edge1: Edge, edge2: Edge) -> bool:
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

            A, B = edge1.start, edge1.end
            C, D = edge2.start, edge2.end
            if (A == C).all() or (A == D).all() or (B == C).all() or (B == D).all():
                return False
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    np.random.seed(seed)
    os.makedirs(data_path, exist_ok=True)

    center = (0.5, 0.5)
    std = 0.15
    nodes = np.clip(np.random.normal(center, std, size=(n, 2)), 0, 1)
    nodes = nodes * SCALE_FACTOR     # SCALE [0,1]->[0,10]
    eu_dist = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)

    intersector = EdgeIntersector()

    # Gabriel graph
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, pos=nodes[i])
        for j in range(i):
            M = (nodes[i] + nodes[j]) / 2
            dist_k = np.linalg.norm(M - nodes, axis=1)
            if dist_k.min() >= np.linalg.norm(M - nodes[i]):
                #G.add_edge(i, j, length=eu_dist[i, j])
                G.add_edge(i, j, length=eu_dist[i, j] * SCALE_FACTOR) #SCALE
                assert intersector.add_edge(Edge(nodes[i], nodes[j]))

    # Additional edges
    for i in range(n):
        # Sort distances and get indices of k nearest neighbors
        deg = max(np.random.randint(k1, k2 + 1) - G.degree(i), 0)
        indices = np.argsort(eu_dist[i])[1 : deg + 1]  # Exclude self (distance of 0)
        for j in indices:
            if j not in G.neighbors(i):
                if intersector.add_edge(Edge(nodes[i], nodes[j])):
                    #G.add_edge(i, j, length=eu_dist[i, j])
                    G.add_edge(i, j, length=eu_dist[i, j] * SCALE_FACTOR) #SCALE

    # Distance matrix
    p = dict(nx.shortest_path_length(G, weight="length"))
    distance_m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            distance_m[i, j] = p[i][j]

    # Generate population
    total_pop = 3000000
    city_pop = np.random.uniform(1, 100, n)  # noise
    eigenvector_centrality = nx.eigenvector_centrality(G, 50000)
    mid = total_pop / sum(eigenvector_centrality.values())
    for i in range(n):
        city_pop[i] += max(
            0,
            np.random.normal(
                eigenvector_centrality[i] * mid,
                eigenvector_centrality[i] * mid / 10,
            ),
        )
    for i, pop in enumerate(city_pop):
        G.nodes[i]["pop"] = pop
        
        
    # Generate attraction parameters
    #alpha = np.random.uniform(0, 0.6, n)  # initial basic coefficient
    '''
    origin A1+B1
    '''
    # alpha = np.random.beta(a=1.2, b=8, size=n)
    # alpha = np.clip(alpha, 0.015, 0.86)

    # beta = np.random.normal(0.32, 0.31, n)  # mean=0.32, std=0.31
    # beta = np.clip(beta, 0.05, 1.80)  # 基于实际最小值和最大值


    '''
    origin A2+B2
    '''
    # alpha = np.random.beta(a=2.0, b=5.0, size=n)
    # alpha = np.clip(alpha, 0.015, 0.86)

    # beta = np.random.normal(0.25, 0.20, n)  # mean=0.25, std=0.20
    # beta = np.clip(beta, 0.05, 1.80)

    '''
    origin A1+B2
    '''
    # alpha = np.random.beta(a=1.2, b=8, size=n)
    # alpha = np.clip(alpha, 0.015, 0.86)

    # beta = np.random.normal(0.25, 0.20, n)  # mean=0.25, std=0.20
    # beta = np.clip(beta, 0.05, 1.80)

    '''
    origin A2+B1
    '''
    alpha = np.random.beta(a=2.0, b=5.0, size=n)
    alpha = np.clip(alpha, 0.015, 0.86)

    beta = np.random.normal(0.32, 0.31, n)  # mean=0.32, std=0.31
    beta = np.clip(beta, 0.05, 1.80)  # 基于实际最小值和最大值



    # '''
    # new
    # '''
    # alpha = np.random.beta(a=1.5, b=6, size=n)
    # alpha = np.clip(alpha, 0.045, 0.65)

    # beta = np.random.normal(0.25, 0.6, n)  # mean=0.32, std=0.31
    # beta = np.clip(beta, 0.05, 2)  # 基于实际最小值和最大值

    # centrality_normalized = np.array(list(eigenvector_centrality.values())) / max(eigenvector_centrality.values())
    # alpha = alpha * (1+0.2 * centrality_normalized) #alpha bigger in more centralized area

    # beta = np.random.normal(1.5, 0.6, n)  # initial decay coefficient
    # beta = np.clip(beta, 0.3, 2.2)
    # beta = np.random.normal(0.32, 0.31, n)  # mean=0.32, std=0.31
    # beta = np.clip(beta, 0.05, 1.80)  # 基于实际最小值和最大值

    # pop_normalized = city_pop / city_pop.max()
    # beta = beta * (1 - 0.2 * pop_normalized)  #beta bigger in an area with more population

    for i in range(n):
        G.nodes[i]["alpha"] = alpha[i]
        G.nodes[i]["beta"] = beta[i]

        attraction_params = {
            "alpha": alpha,
            "beta": beta,
            "node_indices": np.arange(n)
        }


    tabu_table = np.ones((n, n), dtype=int)

    for k in range(n):
        for i in range(n):
            if i == k:
                continue  #  
            d_ki= distance_m[k][i] 
        
            condition1 = alpha[k] > alpha[i] * np.exp(beta[k] * d_ki)
            condition2 = beta[k] <= beta[i]
         
            if condition1 and condition2:
                tabu_table[k][i] = 0


    pickle.dump(tabu_table, open(f"{data_path}/tabu_table.pkl", "wb"))

    pickle.dump(attraction_params, open(f"{data_path}/attraction_params.pkl", "wb"))
    pickle.dump(G, open(f"{data_path}/graph.pkl", "wb"))
    pickle.dump(distance_m, open(f"{data_path}/distance_m.pkl", "wb"))

    return city_pop, distance_m, nodes, alpha, beta, tabu_table
        
        

def batch_gen(data_path: str, n: int, graph_num: int):
    with mp.Pool(10) as pool:
        for i in range(graph_num):
            if os.path.exists(f"{data_path}/{i}/graph.pkl"):
                continue
            pool.apply_async(
                gen_gabriel_graph,
                args=(f"{data_path}/{i}/", int(time.time() + i), n),
            )
        pool.close()
        pool.join()


if __name__ == "__main__":
    # batch_gen("./data/train_100_500/", 100, 500)
    # batch_gen("./data/test_100_20/", 100, 20)
    # batch_gen("./data/train_200_1000/", 200, 1000)
    # batch_gen("./data/test_200_20/", 200, 20)
    # batch_gen("./data/train_300_1000/", 300, 1000)
    # batch_gen("./data/test_300_20/", 300, 20)
    #batch_gen("./data/train_100_999/", 100, 999)
    # batch_gen("./data/train_100_1000/", 100, 1000)
    batch_gen("./data/testA2+B1_500_20/", 500, 20)


