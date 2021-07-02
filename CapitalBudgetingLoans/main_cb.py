from CapitalBudgetingLoans.Environment.Env import ProjectsInstance
from CapitalBudgetingLoans.KAdaptabilityAlgorithm.Normal import algorithm as algorithm_normal
from CapitalBudgetingLoans.KAdaptabilityAlgorithm.RandomDepthFirst import algorithm as algorithm_random
N = 10
xi_dim = 2
env = ProjectsInstance(N=N, xi_dim=xi_dim)
K = 4
time_limit = 10*60

results = algorithm_normal(K=K, env=env, time_limit=time_limit, print_info=True)