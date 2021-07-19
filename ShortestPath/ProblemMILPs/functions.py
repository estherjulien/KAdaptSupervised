from OutputFunctions.hyperplanes import voronoi_hyperlanes

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def scenario_fun_update(K, k_new, xi_new, graph, scen_model=None):
    # use same model and just add new constraint
    y = dict()
    for k in np.arange(K):
        y[k] = {a: scen_model.getVarByName("y_{}[{}]".format(k, a)) for a in np.arange(graph.num_arcs)}
    theta = scen_model.getVarByName("theta")

    scen_model.addConstr(gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * y[k_new][a]
                                     for a in np.arange(graph.num_arcs)) <= theta)
    scen_model.update()

    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    return theta_sol, None, y_sol, scen_model


def scenario_fun_build(K, tau, graph, return_model=True):
    scen_model = gp.Model("Scenario-Based K-Adaptability Problem")
    arcs = graph.arcs
    N = graph.N
    # variables
    theta = scen_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    y = dict()
    for k in np.arange(K):
        y[k] = scen_model.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y_{}".format(k))
    # objective function
    scen_model.setObjective(theta, GRB.MINIMIZE)

    # deterministic constraints
    for k in np.arange(K):
        for j in np.arange(graph.N):
            if j == 0:
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_out[j]) >= 1)
                continue
            if j == N - 1:
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_in[j]) >= 1)
                continue
            scen_model.addConstr(
                gp.quicksum(y[k][a] for a in graph.arcs_out[j])
                - gp.quicksum(y[k][a] for a in graph.arcs_in[j]) >= 0)

    # uncertain constraints
    for k in np.arange(K):
        for xi in tau[k]:
            scen_model.addConstr(gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * y[k][a]
                                             for a in np.arange(graph.num_arcs)) <= theta)

    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    if return_model:
        return theta_sol, None, y_sol, scen_model
    else:
        return theta_sol, None, y_sol


def separation_fun(K, x, y, theta, graph, tau):
    sep_model = gp.Model("Separation Problem")
    sep_model.Params.OutputFlag = 0
    # variables
    zeta = sep_model.addVar(lb=-graph.bigM, name="zeta", vtype=GRB.CONTINUOUS)
    xi = sep_model.addVars(graph.num_arcs, lb=0, ub=1, name="xi", vtype=GRB.CONTINUOUS)

    # objective function
    sep_model.setObjective(zeta, GRB.MAXIMIZE)

    # uncertainty set
    sep_model.addConstr(gp.quicksum(xi[a] for a in np.arange(graph.num_arcs)) <= graph.gamma)

    for k in np.arange(K):
        if tau[k]:
            sep_model.addConstr(zeta <= gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * y[k][a]
                                                    for a in np.arange(graph.num_arcs)) - theta)

    # solve
    sep_model.optimize()
    zeta_sol = zeta.X
    xi_sol = np.array([var.X for i, var in xi.items()])
    return zeta_sol, xi_sol, []


def robust_counterpart_voronoi(K, centroids, graph):
    coef, b = voronoi_hyperlanes(K, centroids, graph)
    rc = gp.Model("Robust counterpart voronoi diagram")
    rc.Params.OutputFlag = 0

    N = graph.N
    # variables
    theta = rc.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    y = dict()
    for k in np.arange(K):
        y[k] = rc.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y_{}".format(k))
    # dual variables
    d_1 = dict()
    d_2 = rc.addVars(K)
    d_3 = dict()
    for k in np.arange(K):
        d_1[k] = rc.addVars(graph.num_arcs)
        d_3[k] = rc.addVars(K-1)

    # objective function
    rc.setObjective(theta, GRB.MINIMIZE)

    # deterministic constraints
    for k in np.arange(K):
        for j in np.arange(graph.N):
            if j == 0:
                rc.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_out[j]) >= 1)
                continue
            if j == N - 1:
                rc.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_in[j]) >= 1)
                continue
            rc.addConstr(
                gp.quicksum(y[k][a] for a in graph.arcs_out[j])
                - gp.quicksum(y[k][a] for a in graph.arcs_in[j]) >= 0)

    # uncertain constraints
    rc.addConstrs(gp.quicksum(y[k][a]*graph.distances_array[a] + d_1[k][a] for a in np.arange(graph.num_arcs))
                  + d_2[k]*graph.gamma
                  + gp.quicksum(d_3[k][j]*b[k][j] for j in np.arange(K-1))
                  <= theta for k in np.arange(K))
    rc.addConstrs(graph.distances_array[a]/2*y[k][a] - d_1[k][a] - d_2[k]
                  + gp.quicksum(coef[k][j][a]*d_3[k][j] for j in np.arange(K-1))
                  <= 0 for a in np.arange(graph.num_arcs) for k in np.arange(K))

    # solve model
    rc.optimize()
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = theta.X

    output = {"theta": theta_sol, "x": None, "y": y_sol}
    return output


def k_adapt_centroid_tau(K, env, tau):
    subset_centroid = np.array([np.mean(np.array(tau[k]), axis=0) for k in np.arange(K)])

    # voronoi diagram
    theta, x, y = robust_counterpart_voronoi(K, subset_centroid, env).values()
    return theta, x, y
