from CapitalBudgetingLoans.Output.centroid_voronoi import *

from CapitalBudgetingLoans.Environment.Env import ProjectsInstance
from CapitalBudgetingLoans.KAdaptabilityAlgorithm.Normal import algorithm as algorithm_normal
from CapitalBudgetingLoans.KAdaptabilityAlgorithm.RandomDepthFirst import algorithm as algorithm_random
from joblib import Parallel, delayed

num_cores = 4
N = 10
xi_dim = 2
# env_list = [ProjectsInstance(N=N, xi_dim=xi_dim, inst_num=i) for i in range(num_cores)]

K = 4
time_limit = 10*60

# results = algorithm_random(K=K, env=env_list[0], time_limit=time_limit, print_info=True)

# results = Parallel(n_jobs=num_cores)(delayed(algorithm_random)(env_list[i], K, time_limit, print_info=True) for i in range(num_cores))
# results = Parallel(n_jobs=num_cores)(delayed(i)(env_list[j], K, time_limit, print_info=True) for i in [algorithm_normal, algorithm_random] for j in range(4))

for i in range(4):
    file_name = f"Results/Decisions/final_results_cp_rand_K4_N10_d2_inst{i}.pickle"
    tau_to_centroid(K, file_name, grid=0.1)