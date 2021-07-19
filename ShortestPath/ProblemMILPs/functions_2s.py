import numpy as np
import gurobipy as gp
from gurobipy import GRB


def scenario_fun_update(K, k_new, xi_new, graph, scen_model=None):
    # use same model and just add new constraints
    x = {a: scen_model.getVarByName("x[{}]".format(a)) for a in np.arange(graph.num_arcs)}
    y = dict()
    for k in np.arange(K):
        y[k] = {a: scen_model.getVarByName("y_{}[{}]".format(k, a)) for a in np.arange(graph.num_arcs)}
    theta = scen_model.getVarByName("theta")

    # first stage constraint
    scen_model.addConstr(
        gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * x[a] for a in np.arange(graph.num_arcs))
        >= graph.max_first_stage)
    # objective constraint
    scen_model.addConstr(gp.quicksum((1 + xi_new[a] / 2) * graph.distances_array[a] * (y[k_new][a] + x[a])
                                     for a in np.arange(graph.num_arcs)) <= theta)
    scen_model.update()
    # solve model
    scen_model.optimize()
    x_sol = {i: var.X for i, var in x.items()}
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    return theta_sol, x_sol, y_sol, scen_model


def scenario_fun_build(K, tau, graph, return_model=True):
    scen_model = gp.Model("Scenario-Based K-Adaptability Problem")
    N = graph.N
    # variables
    theta = scen_model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
    x = scen_model.addVars(graph.num_arcs, vtype=GRB.BINARY, name="x")
    s = scen_model.addVars(graph.N, vtype=GRB.BINARY, name="s")
    y = dict()
    for k in np.arange(K):
        y[k] = scen_model.addVars(graph.num_arcs, vtype=GRB.BINARY, name="y_{}".format(k))

    # objective function
    scen_model.setObjective(theta, GRB.MINIMIZE)

    # deterministic constraints

    # first stage
    scen_model.addConstr(gp.quicksum(x[a] for a in graph.arcs_out[0]) >= 1)
    # switch point without extra variable
    # scen_model.addConstr(gp.quicksum(gp.quicksum(x[a] for a in graph.arcs_in[j]) - gp.quicksum(x[a] for a in graph.arcs_out[j]) for j in np.arange(1, graph.N-1)) == 1)
    # outside range
    scen_model.addConstrs(x[a] == 0 for j in graph.outside_range for a in graph.arcs_out[j])

    # switch point
    scen_model.addConstr(gp.quicksum(s[j] for j in np.arange(graph.N)) == 1)
    # x in
    scen_model.addConstrs(gp.quicksum(x[a] for a in graph.arcs_in[j]) - gp.quicksum(x[a] for a in graph.arcs_out[j])
                          == s[j] for j in np.arange(1, graph.N-1))
    # y out
    scen_model.addConstrs(gp.quicksum(y[k][a] for a in graph.arcs_out[j]) - gp.quicksum(y[k][a] for a in graph.arcs_in[j])
                          == s[j] for j in np.arange(1, graph.N-1) for k in np.arange(K))
    # valid inequality switch point
    scen_model.addConstrs(s[j] == 0 for j in graph.inside_range)
    scen_model.addConstrs(s[j] == 0 for j in graph.outside_range)
    # second stage
    for k in np.arange(K):
        # only one outgoing for each node
        scen_model.addConstrs(
            gp.quicksum(x[a] + y[k][a] for a in graph.arcs_out[j]) <= 1 for j in np.arange(graph.N - 1))
        # all ys inside range have to be zero
        scen_model.addConstrs(y[k][a] == 0 for j in graph.inside_range for a in graph.arcs_in[j])
        # total sum of arcs is smaller than N-1
        scen_model.addConstr(gp.quicksum(x[a] + y[k][a] for a in np.arange(graph.N)) <= graph.N - 1)
        for j in np.arange(graph.N):
            if j == 0:
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_out[j]) <= 0)
                continue
            if j == N - 1:
                scen_model.addConstr(gp.quicksum(y[k][a] for a in graph.arcs_in[j]) >= 1)
                continue
            # normal shortest path constraint
            scen_model.addConstr(
                gp.quicksum(y[k][a] + x[a] for a in graph.arcs_out[j])
                - gp.quicksum(y[k][a] + x[a] for a in graph.arcs_in[j]) >= 0)

            # no vertex can have ingoing y and outgoing x
            scen_model.addConstr(
                gp.quicksum(y[k][a] for a in graph.arcs_in[j])
                + gp.quicksum(x[a] for a in graph.arcs_out[j]) <= 1)

    # uncertain constraints
    for k in np.arange(K):
        for xi in tau[k]:
            # first stage constraint
            scen_model.addConstr(
                gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * x[a] for a in np.arange(graph.num_arcs))
                >= graph.max_first_stage)
            # objective function
            scen_model.addConstr(gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * (x[a] + y[k][a])
                                             for a in np.arange(graph.num_arcs)) <= theta)

    # solve model
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    x_sol = {i: var.X for i, var in x.items()}
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = {i: var.X for i, var in y[k].items()}
    theta_sol = scen_model.getVarByName("theta").X

    if return_model:
        return theta_sol, [], y_sol, scen_model
    else:
        return theta_sol, [], y_sol


def separation_fun(K, x, y, theta, graph, tau):
    sep_model = gp.Model("Separation Problem")
    sep_model.Params.OutputFlag = 0
    # variables
    zeta = sep_model.addVar(lb=-graph.bigM, name="zeta", vtype=GRB.CONTINUOUS)
    xi = sep_model.addVars(graph.num_arcs, lb=0, ub=1, name="xi", vtype=GRB.CONTINUOUS)

    num_consts = 2
    z = dict()
    for k in np.arange(K):
        z[k] = sep_model.addVars(num_consts, vtype=GRB.BINARY)

    # objective function
    sep_model.setObjective(zeta, GRB.MAXIMIZE)

    # z constraint
    sep_model.addConstrs(gp.quicksum(z[k][c] for c in np.arange(num_consts)) == 1 for k in np.arange(K))

    # uncertainty set
    sep_model.addConstr(gp.quicksum(xi[a] for a in np.arange(graph.num_arcs)) <= graph.gamma)

    for k in np.arange(K):
        if tau[k]:
            # first stage constraint
            sep_model.addConstr(zeta <= -(gp.quicksum((1 + xi[a] / 2) * graph.distances_array[a] * x[a] for a in np.arange(graph.num_arcs)))
                                 + graph.max_first_stage + graph.bigM * (1 - z[k][0]))
            # objective constraint
            sep_model.addConstr(zeta <= gp.quicksum((1 + xi[a] / 2) *
                                graph.distances_array[a] * (x[a] + y[k][a]) for a in np.arange(graph.num_arcs))
                                - theta + graph.bigM * (1 - z[k][1]))

    # solve
    sep_model.optimize()
    zeta_sol = zeta.X
    xi_sol = np.array([var.X for i, var in xi.items()])
    return zeta_sol, xi_sol, []
