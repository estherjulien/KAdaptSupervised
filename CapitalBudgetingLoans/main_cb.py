from CapitalBudgetingLoans.Environment.Env import ProjectsInstance
from CapitalBudgetingLoans.ProblemMILPs.functions_loans import *
from KAdaptabilityAlgorithm.Random import algorithm as algorithm_random
from joblib import Parallel, delayed

num_cores = 4
N = 10
xi_dim = 2
env_list = [ProjectsInstance(N=N, xi_dim=xi_dim, inst_num=i) for i in range(num_cores*5)]

K = 4
time_limit = 20*60

# results = algorithm_random(K, env_list[0], scenario_fun_build, scenario_fun_update, separation_fun,
#                                                                 k_adapt_centroid_tau=k_adapt_centroid_tau,
#                                                                 time_limit=time_limit, print_info=True)

results = Parallel(n_jobs=num_cores)(delayed(algorithm_random)(K, env_list[i], scenario_fun_build, scenario_fun_update,
                                                               separation_fun,
                                                               k_adapt_centroid_tau=k_adapt_centroid_tau,
                                                               time_limit=time_limit, print_info=True,
                                                               problem_type=f"cp_K{K}_N{N}_d{xi_dim}")
                                                                for i in range(num_cores*5))