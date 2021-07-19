from CapitalBudgetingLoans.Environment.Env import *
from CapitalBudgetingLoans.ProblemMILPs.functions_loans import robust_counterpart_voronoi
from OutputFunctions.hyperplanes import voronoi_hyperlanes
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import seaborn as sn
import numpy as np
import pickle
import time


def k_adapt_centroid_sample(K, file_name, grid=0.1):
    start_time = time.time()

    with open(file_name, "rb") as handle:
        env, results = pickle.load(handle)

    x_input = results["x"]
    y_input = results["y"]
    Xi, Xi_subset = tau_grid(K, grid, env, x_input, y_input)

    subset_centroid = np.zeros([K, env.xi_dim])
    for k in np.arange(K):
        Xi_k = Xi[Xi_subset == k]
        subset_centroid[k] = np.mean(Xi_k, axis=0)

    # voronoi diagram
    output = robust_counterpart_voronoi(K, subset_centroid, env)
    runtime = time.time() - start_time
    output.update({"runtime": runtime})
    theta_real = results["theta"]
    theta_vor = output["theta"]
    vor_real = theta_vor/theta_real
    print(f"Instance {env.inst_num}: {vor_real}, theta_real = {theta_real}, theta_vor = {theta_vor}")
    return output, vor_real


def tau_to_centroid(K, file_name, grid=0.1):
    with open(file_name, "rb") as handle:
        env, results = pickle.load(handle)

    x_input = results["x"]
    y_input = results["y"]
    Xi, Xi_subset = tau_grid(K, grid, env, x_input, y_input)

    subset_centroid = np.zeros([K, env.xi_dim])
    for k in np.arange(K):
        Xi_k = Xi[Xi_subset == k]
        subset_centroid[k] = np.mean(Xi_k, axis=0)

    # voronoi diagram
    coef, b = voronoi_hyperlanes(K, subset_centroid, env)

    plot_tau_grid(Xi, Xi_subset, K, env, subset_centroid, coef, b)


def tau_grid(K, grid, env, x_input, y_input):
    x_0, x = x_input
    y_0, y = y_input
    Xi = np.mgrid[[np.s_[-1:1+grid:grid] for i in np.arange(env.xi_dim)]].reshape(env.xi_dim, -1).T
    Xi_subset = np.zeros(len(Xi), dtype=np.int8)
    xi_num = 0
    for xi in Xi:
        obj_set = []
        for k in np.arange(K):
            const_set = []
            # constraints
            const_set.append(sum(cost_fun(env.projects[p], xi) * (x[p]) for p in np.arange(env.N)) <= env.budget + x_0)
            const_set.append(sum(cost_fun(env.projects[p], xi) * (x[p] + y[k][p]) for p in np.arange(env.N)) <= (env.budget + x_0 + y_0[k]))
            if all(const_set):
                obj_set.append(-(sum(rev_fun(env.projects[p], xi) * (x[p] + env.kappa*y[k][p]) for p in np.arange(env.N)) - env.lam*(x_0 + env.mu*y_0[k])))
            else:
                obj_set.append(0)
        Xi_subset[xi_num] = np.argmin(obj_set)
        xi_num += 1
    return Xi, Xi_subset


def plot_tau_grid(Xi, Xi_subset, K, env, subset_centroid=None, coef=None, b=None):
    sn.set_style("whitegrid")
    # plot grid subset results
    plt.scatter(x=Xi[:, 0], y=Xi[:, 1], c=Xi_subset)

    if subset_centroid is not None: # plot centroids
        # plt.scatter(x=subset_centroid[:, 0], y=subset_centroid[:, 1], c=cols, s=200, edgecolors=cols)
        plt.scatter(x=subset_centroid[:, 0], y=subset_centroid[:, 1], c=np.arange(K), s=100, edgecolors="k")
    if coef is not None:
        X = np.array([-2, 2])
        for k in np.arange(K):
            for i in np.arange(K-1):
                Y = -(b[k][i] + X*coef[k][i][0])/coef[k][i][1]
                plt.plot(X, Y, "k")
    plt.axis('square')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()
    plt.savefig(f"./Results/Plots/plot_cb_K{K}_N{env.N}_d{env.xi_dim}_inst{env.inst_num}")
    plt.close()
