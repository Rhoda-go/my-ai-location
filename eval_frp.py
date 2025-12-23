# from dataset import GraphImpDataset
# # from methods.br import run_br
# # from methods.fr2fp import run_FR2FP
# # from methods.greedy_swap import run_greedy_swap_reloc
# # from methods.gurobi import run_gurobi_reloc
# # from methods.random_swap import run_random_swap_reloc
# from methods.ppo_swap import run_ppo_swap_reloc
# from methods.swap_solver import run_original
# from results import save_avg, save_frp_results
# from utils import get_config


# def eval_frp(config):
#     data_path = config['data_path']
    
#     reloc_coef = config['reloc_coef']
#     dataset = GraphImpDataset(data_path, "range(5, 10)")
#     save_path = f"{data_path}/results_frp_{reloc_coef}/"

#     # res_list = {}
#     # baseline = "Gurobi"
#     # optimal_path = run_gurobi_reloc(dataset=dataset, save_path=save_path, reloc_coef=reloc_coef, OutputFlag=0, MIPGap=0)
#     # res_list[baseline] = optimal_path
#     # original_path = run_original(dataset=dataset, save_path=save_path)
#     # save_avg(optimal_path, dataset, optimal_path, reloc=original_path)
#     # for k, v in config["methods"].items():
#     #     run_fn = eval(v["run_fn"])
#     #     sol_path = run_fn(dataset=dataset, save_path=save_path, reloc_coef=reloc_coef, **v)
#     #     res_list[k] = sol_path
#     #     save_avg(sol_path, dataset, optimal_path, reloc=original_path)

#     # save_frp_results(save_path, res_list, dataset.p)

#     #只跑test
#     res_list = {}
#     # 仅运行原始解法（无依赖，无需权重）
#     original_path = run_original(dataset=dataset, save_path=save_path)
#     res_list["Original"] = original_path
#     save_avg(original_path, dataset, original_path, reloc=original_path)
#     save_frp_results(save_path, res_list, dataset.p)
#     run_ppo_swap_reloc


# if __name__ == "__main__":
#     config = get_config(["-c", "config/eval_frp.yaml"])
#     eval_frp(config)


from dataset import GraphImpDataset
from methods.ppo_swap import run_ppo_swap_reloc  # 确保导入了该函数
from methods.swap_solver import run_original
from results import save_avg, save_frp_results
from utils import get_config

def eval_frp(config):
    data_path = config['data_path']
    reloc_coef = config['reloc_coef']
    dataset = GraphImpDataset(data_path, "range(5, 10)")
    save_path = f"{data_path}/results_frp_{reloc_coef}/"

    res_list = {}
    # 1. 运行原始解法（基础基线）
    original_path = run_original(dataset=dataset, save_path=save_path)
    res_list["Original"] = original_path
    save_avg(original_path, dataset, original_path, reloc=original_path)

    # 2. 核心：调用run_ppo_swap_reloc（从配置文件读取参数）
    for k, v in config["methods"].items():
        # 动态获取函数（对应配置文件中的run_fn: run_ppo_swap_reloc）
        run_fn = eval(v["run_fn"])
        # 执行PPO-swap-reloc算法，传入所需参数
        sol_path = run_fn(
            dataset=dataset,
            save_path=save_path,
            reloc_coef=reloc_coef,
            **v  # 从配置文件读取iter_num、ckpt、device、name等参数
        )
        res_list[k] = sol_path
        # 保存PPO算法与原始解的对比结果
        save_avg(sol_path, dataset, original_path, reloc=original_path)

    # 3. 保存所有结果（原始解+PPO解）
    save_frp_results(save_path, res_list, dataset.p)

if __name__ == "__main__":
    config = get_config(["-c", "config/eval_frp.yaml"])
    eval_frp(config)