from CapitalBudgetingLoans.Environment.Env import *

from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from gurobipy import GRB
import gurobipy as gp
import seaborn as sn
import numpy as np
import pickle


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


def voronoi_hyperlanes(K, centroids, env):
    # try voronoi
    # try:
    #     vor = Voronoi(centroids)
    #     search = {i: [] for i in np.arange(K)}
    #     for i in np.arange(K):
    #         j = 0
    #         for row in vor.ridge_points:
    #             if i in row:
    #                 search[i].append(j)
    #             j += 1
    #     vor_allowed = True
    #     fig = voronoi_plot_2d(vor)
    #     fig.close()
    # except:
    vor_allowed = False

    coef = {i: [] for i in np.arange(K)}
    b = {i: [] for i in np.arange(K)}
    # bisection
    i = 0
    for z in centroids:
        l = 0
        for z_b in centroids:
            if i == l:
                l += 1
                continue
            # elif vor_allowed and l not in np.unique(vor.ridge_points[search[i]]):
            #     l += 1
            #     continue
            coef[i].append(z_b - z)
            b[i].append(1/2*sum([z_b[j]**2 - z[j]**2 for j in np.arange(env.xi_dim)]))
            l += 1
        i += 1
    return coef, b


def plot_tau_grid(Xi, Xi_subset, K, env, subset_centroid=None, coef=None, b=None):
    sn.set_style("whitegrid")
    # plot grid subset results
    plt.scatter(x=Xi[:, 0], y=Xi[:, 1], c=Xi_subset)

    if subset_centroid is not None: # plot centroids
        plt.scatter(x=subset_centroid[:, 0], y=subset_centroid[:, 1], c=np.arange(K), s=100, edgecolors="k")
    if coef is not None:
        X = np.array([-2, 2])
        for k in np.arange(K):
            hp_num = len(coef[k])
            for i in np.arange(hp_num):
                Y = -(b[k][i] + X*coef[k][i][0])/coef[k][i][1]
                plt.plot(X, Y, "k")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()
    plt.savefig(f"./Results/Plots/plot_cb_K{K}_N{env.N}_d{env.xi_dim}_inst{env.inst_num}")
    plt.close()


def robust_counterpart_voronoi(K, tau, env):
    coef, b, _, scen_bag = voronoi_hyperlanes(K, tau)
    rc = gp.Model("Robust counterpart voronoi diagram")
    rc.Params.OutputFlag = 0
    N = graph.N
    # variables
    theta = rc.addVar(lb=0, vtype=GRB.CONTINUOUS)
    y_index = [(k, a) for a in np.arange(graph.num_arcs) for k in np.arange(K)]
    y = rc.addVars(y_index, vtype=GRB.BINARY)
    lamb = dict()
    for k in np.arange(K):
        lamb[k] = dict()
        for i in scen_bag[k]:
            lamb[k][i] = dict()
            lamb[k][i][0] = rc.addVar(lb=0, vtype=GRB.CONTINUOUS)
            lamb[k][i][1] = rc.addVars(graph.num_arcs, lb=0, vtype=GRB.CONTINUOUS)
            lamb[k][i][2] = rc.addVars(len(b[i]), lb=0, vtype=GRB.CONTINUOUS)

    # objective function
    rc.setObjective(theta, GRB.MINIMIZE)

    # robust counterparts of objective function
    for k in np.arange(K):
        for i in scen_bag[k]:
            # constants
            rc.addConstr(gp.quicksum(graph.distances_array[a]*y[k, a] for a in np.arange(graph.num_arcs)) +
                         graph.gamma*lamb[k][i][0] +
                         gp.quicksum(lamb[k][i][1][a] for a in np.arange(graph.num_arcs)) +
                         gp.quicksum(lamb[k][i][2][j]*b[i][j] for j in np.arange(len(b[i]))) <= theta)
            # other constraints
            rc.addConstrs((1/2*graph.distances_array[a]*y[k, a] - lamb[k][i][0] - lamb[k][i][1][a]
                           - gp.quicksum(lamb[k][i][2][j]*coef[i][j][a] for j in np.arange(len(b[i]))) <= 0) for a in np.arange(graph.num_arcs))

    # constraints without uncertainty
    for k in np.arange(K):
        for j in np.arange(graph.N):
            if j == 0:
                rc.addConstr(
                    gp.quicksum(y[k, a] for a in graph.arcs_out[j])
                    - gp.quicksum(y[k, a] for a in graph.arcs_in[j]) >= 1)
                continue
            if j == N-1:
                rc.addConstr(
                    gp.quicksum(y[k, a] for a in graph.arcs_out[j])
                    - gp.quicksum(y[k, a] for a in graph.arcs_in[j]) >= -1)
                continue
            rc.addConstr(
                gp.quicksum(y[k, a] for a in graph.arcs_out[j])
                - gp.quicksum(y[k, a] for a in graph.arcs_in[j]) >= 0)
    # solve
    rc.optimize()
    y_sol = {i: var.X for i, var in y.items()}
    theta_sol = theta.X
    # check lambda results
    lamb_sol = dict()
    for k in np.arange(K):
        lamb_sol[k] = dict()
        for i in scen_bag[k]:
            lamb_sol[k][i] = dict()
            lamb_sol[k][i][0] = lamb[k][i][0].X
            lamb_sol[k][i][1] = {i: var.X for i, var in lamb[k][i][1].items()}
            lamb_sol[k][i][2] = {i: var.X for i, var in lamb[k][i][2].items()}

    theta_list = np.array([max(sum(graph.distances_array[a] * y_sol[k, a] for a in np.arange(graph.num_arcs)) +
                 graph.gamma * lamb_sol[k][i][0] + sum(lamb_sol[k][i][1][a] for a in np.arange(graph.num_arcs)) +
                 sum(lamb_sol[k][i][2][j] * b[i][j] for j in np.arange(len(b[i]))) for i in scen_bag[k]) for k in np.arange(K)])
    k_vor = np.argmin(theta_list)

    return theta_sol, [], y_sol, k_vor
