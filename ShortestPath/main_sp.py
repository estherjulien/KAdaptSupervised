from ShortestPath.Environment.Env import Graph
# from ShortestPath.ProblemMILPs.functions import *
from ShortestPath.ProblemMILPs.functions_2s import *
from KAdaptabilityAlgorithm.Random import algorithm as algorithm_random
from joblib import Parallel, delayed
import pickle

from ShortestPath.Supervised.data_functions import train_data_prep

num_cores = 8
num_instances = num_cores
N = 100
# env_list = [Graph(N=N, gamma_perc=0.01, first_stage_ratio=0.3, max_degree=5, throw_away_perc=0.3, inst_num=i) for i in np.arange(num_instances)]

K = 4
time_limit = 1*60*60
problem_type = f"sp_2s_K{K}_N{N}"

# results = algorithm_random(K, env_list[0], scenario_fun_build, scenario_fun_update, separation_fun,
#                            time_limit=time_limit, print_info=True, k_adapt_centroid_tau=None)
#
# results = Parallel(n_jobs=num_cores)(delayed(algorithm_random)(K, env, scenario_fun_build, scenario_fun_update,
#                                                                     separation_fun, k_adapt_centroid_tau=k_adapt_centroid_tau,
#                                                                     time_limit=time_limit, print_info=True,
#                                                                     problem_type=problem_type)
#                                                                     for env in env_list)

X, Y = train_data_prep(problem_type)
