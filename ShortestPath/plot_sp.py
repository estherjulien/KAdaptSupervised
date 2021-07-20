import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import pickle

num_inst = 8

results = dict()
results_vor = dict()
for i in np.arange(num_inst):
    with open(f"Results/Decisions/final_results_sp_2s_K4_N100_inst{i}.pickle", "rb") as handle:
        results[i] = pickle.load(handle)[1]
    results_vor[i] = results[i]["vor"]

t_grid = np.array([*np.arange(0, 65, 5), *np.arange(60, 1*60*60+15, 15)])
num_grids = len(t_grid)
obj_norm = pd.DataFrame(index=t_grid, columns=[f"norm_{i}" for i in np.arange(num_inst)], dtype=np.float)
obj_vor = pd.DataFrame(index=t_grid, columns=[f"vor_{i}" for i in np.arange(num_inst)], dtype=np.float)
for i in np.arange(num_inst):
    # normal
    t_norm = np.zeros(num_grids)
    theta_final = results[i]["theta"]
    for t, theta in results[i]["inc_thetas"].items():
        t_norm[t_grid > t] = theta/theta_final
    obj_norm[f"norm_{i}"] = t_norm

    t_vor = np.zeros(num_grids)
    theta_final = results[i]["theta"]
    for t, theta in results_vor[i]["inc_thetas"].items():
        t_vor[t_grid > t] = theta/theta_final
    obj_vor[f"vor_{i}"] = t_vor
obj_norm[obj_norm == 0] = np.nan
obj_vor[obj_vor == 0] = np.nan
sn.set_style("whitegrid")
# plot results
avg_norm = np.mean(obj_norm, axis=1)
plt.plot(t_grid, obj_norm[f"norm_{0}"], "k", label="Normal")
for i in np.arange(1, num_inst):
    plt.plot(t_grid, obj_norm[f"norm_{i}"], "k")

avg_vor = np.mean(obj_vor, axis=1)
plt.plot(t_grid, obj_vor[f"vor_{0}"], "r", label="Voronoi")
for i in np.arange(1, num_inst):
    plt.plot(t_grid, obj_vor[f"vor_{i}"], "r")
# plt.plot(t_grid, avg_vor, label="Voronoi")


plt.ylim([0.95, 1.025])
plt.legend()
plt.show()
