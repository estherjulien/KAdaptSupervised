import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle

num_inst = 4

res_norm = dict()
res_rand = dict()
for i in np.arange(num_inst):
    with open(f"Results/Decisions/final_results_cp_normal_K4_N10_inst{i}.pickle", "rb") as handle:
        res_norm[i] = pickle.load(handle)[1]
    with open(f"Results/Decisions/final_results_cp_rand_K4_N10_inst{i}.pickle", "rb") as handle:
        res_rand[i] = pickle.load(handle)[1]

t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 10*60+15, 15)])
num_grids = len(t_grid)
obj_norm = pd.DataFrame(index=t_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float)
obj_rand = pd.DataFrame(index=t_grid, columns=[f"rand_{i}" for i in np.arange(num_inst)], dtype=np.float)

for i in np.arange(num_inst):
    # normal
    t_norm = np.zeros(num_grids)
    theta_final = res_rand[i]["theta"]
    for t, theta in res_norm[i]["inc_thetas"].items():
        t_norm[t_grid > t] = theta/theta_final
    obj_norm[f"norm_{i}"] = t_norm

    # random
    t_rand = np.zeros(num_grids)
    for t, theta in res_rand[i]["inc_thetas"].items():
        t_rand[t_grid > t] = theta/theta_final
    obj_rand[f"rand_{i}"] = t_rand

sn.set_style("whitegrid")
# plot results
avg_norm = np.mean(obj_norm, axis=1)
plt.plot(t_grid, avg_norm, label="Normal")

avg_random = np.mean(obj_rand, axis=1)
plt.plot(t_grid, avg_random, label="Random")
plt.ylim([0.9, 1.025])
plt.legend()
plt.show()
