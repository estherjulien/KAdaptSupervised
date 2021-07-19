from ShortestPath.Environment.Env import Graph
from ShortestPath.ProblemMILPs.functions import *
from KAdaptabilityAlgorithm.Random import algorithm as algorithm_random
from joblib import Parallel, delayed
import pickle

num_cores = 8
N = 100
env_list = [Graph(N=N, gamma_perc=0.01, first_stage_ratio=0.3, max_degree=5, throw_away_perc=0.8, inst_num=i) for i in np.arange(num_cores*10)]

K = 4
time_limit = 2*60*60

# results = algorithm_random(K, env_list[0], scenario_fun_build, scenario_fun_update, separation_fun,
#                            time_limit=time_limit, print_info=True, k_adapt_centroid_tau=k_adapt_centroid_tau)

results = Parallel(n_jobs=num_cores)(delayed(algorithm_random)(K, env, scenario_fun_build, scenario_fun_update,
                                                                    separation_fun, k_adapt_centroid_tau=k_adapt_centroid_tau,
                                                                    time_limit=time_limit, print_info=True,
                                                                    problem_type=f"sp_K{K}_N{N}")
                                                                    for env in env_list)

# for i in np.arange(8):
#     with open(f"Results/Decisions/final_results_sp_K8_N100_inst{i}.pickle", "rb") as handle:
#         env, results = pickle.load(handle)
#     env.plot_graph_solutions(K, results["y"], tau=results["tau"])

# results_list = dict()
# env_list = dict()

# for i in np.arange(8):
#     with open(f"Results/Decisions/tmp_results_sp_K4_N100_inst{i}.pickle", "rb") as handle:
#         env_list[i], results_list[i] = pickle.load(handle)
#     env_list[i].plot_graph_solutions(K, results_list[i]["y"], tau=results_list[i]["tau"], extra=True, tmp=True, it=1)
