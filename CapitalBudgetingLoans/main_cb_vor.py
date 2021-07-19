from CapitalBudgetingLoans.ProblemMILPs.centroid_voronoi import *
K = 4
vr_list = []
for i in range(20):
    file_name = f"Results/Decisions/final_results_cp_rand_K4_N10_d2_inst{i}.pickle"
    _, vor_real = k_adapt_centroid(K, file_name)
    vr_list.append(vor_real)

print(f"avg: {np.mean(vr_list)}")
