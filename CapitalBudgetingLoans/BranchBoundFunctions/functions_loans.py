import gurobipy as gp
from gurobipy import GRB
from CapitalBudgetingLoans.Environment.Env import *


def scenario_fun_update(K, k_new, xi_new, env, scen_model):
    projects = env.projects
    N = env.N

    # load variables
    x_0 = scen_model.getVarByName("x0")
    x = {p: scen_model.getVarByName(f"x[{p}]") for p in np.arange(N)}
    y_0 = {k: scen_model.getVarByName(f"y0[{k}]") for k in np.arange(K)}
    y = {k: {p: scen_model.getVarByName(f"y_{k}[{p}]") for p in np.arange(N)} for k in np.arange(K)}
    theta = scen_model.getVarByName("theta")

    # add new constraints
    # objective constraint
    scen_model.addConstr(-(gp.quicksum(rev_fun(projects[p], xi_new) * (x[p] + env.kappa * y[k_new][p])
                                       for p in np.arange(N)) - env.lam * (x_0 + env.mu * y_0[k_new])) <= theta)
    # budget constraint
    scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi_new) * (x[p]) for p in np.arange(N))
                         <= env.budget + x_0)
    scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi_new) * (x[p] + y[k_new][p]) for p in np.arange(N))
                         <= (env.budget + x_0 + y_0[k_new]))

    # solve
    scen_model.optimize()
    x_0_sol = x_0.X
    x_sol = np.array([var.X for i, var in x.items()])
    y_0_sol = np.array([var.X for i, var in y_0.items()])
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = np.array([var.X for i, var in y[k].items()])
    theta_sol = theta.X

    return theta_sol, [x_0_sol, x_sol], [y_0_sol, y_sol], scen_model


def scenario_fun_build(K, tau, env, return_model=False):
    projects = env.projects
    N = env.N
    scen_model = gp.Model("Scenario-Based K-Adaptability Problem")
    # variables
    theta = scen_model.addVar(lb=-env.theta_lb, ub=0, name="theta")
    x_0 = scen_model.addVar(lb=0, name="x0")
    x = scen_model.addVars(N, vtype=GRB.BINARY, name="x")
    y = dict()
    for k in np.arange(K):
        y[k] = scen_model.addVars(N, vtype=GRB.BINARY, name=f"y_{k}")
    y_0 = scen_model.addVars(K, lb=0, name="y0")

    # objective function
    scen_model.setObjective(theta, GRB.MINIMIZE)

    # constraints
    for k in np.arange(K):
        for xi in tau[k]:
            # objective constraint
            scen_model.addConstr(-(gp.quicksum(rev_fun(projects[p], xi) * (x[p] + env.kappa*y[k][p])
                                    for p in np.arange(N)) - env.lam*(x_0 + env.mu*y_0[k])) <= theta)
            # budget constraint
            scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi) * (x[p]) for p in np.arange(N))
                                      <= env.budget + x_0)
            scen_model.addConstr(gp.quicksum(cost_fun(projects[p], xi) * (x[p] + y[k][p]) for p in np.arange(N))
                                      <= (env.budget + x_0 + y_0[k]))

        # other constraints
        scen_model.addConstrs(x[p] + y[k][p] <= 1 for p in np.arange(N))

    # solve
    scen_model.Params.OutputFlag = 0
    scen_model.optimize()
    x_0_sol = x_0.X
    x_sol = np.array([var.X for i, var in x.items()])
    y_0_sol = np.array([var.X for i, var in y_0.items()])
    y_sol = dict()
    for k in np.arange(K):
        y_sol[k] = np.array([var.X for i, var in y[k].items()])
    theta_sol = theta.X

    if return_model:
        return theta_sol, [x_0_sol, x_sol], [y_0_sol, y_sol], scen_model
    else:
        return theta_sol, [x_0_sol, x_sol], [y_0_sol, y_sol]


def separation_fun(K, x_input, y_input, theta, env):
    x_0, x = x_input
    y_0, y = y_input
    N =env.N
    projects = env.projects
    # model
    sep_model = gp.Model("Separation Problem")
    sep_model.Params.OutputFlag = 0
    # variables
    zeta = sep_model.addVar(lb=-env.bigM, name="zeta")
    xi = sep_model.addVars(env.xi_dim, lb=-1, ub=1, name="xi")
    z_index = [(k, i) for k in np.arange(K) for i in [0, 1, 2]]
    z = sep_model.addVars(z_index, name="z", vtype=GRB.BINARY)

    # objective function
    sep_model.setObjective(zeta, GRB.MAXIMIZE)
    # z constraint
    sep_model.addConstrs(gp.quicksum(z[k, l] for l in np.arange(3)) == 1 for k in np.arange(K))
    # objective constraint
    sep_model.addConstrs((zeta + env.bigM*z[k, 2] <= -(gp.quicksum(rev_fun(projects[p], xi) *
                                                    (x[p] + env.kappa*y[k][p]) for p in np.arange(N)) -
                                                    env.lam*(x_0 + env.mu*y_0[k])) - theta + env.bigM)
                                                    for k in np.arange(K))
    # budget constraints
    sep_model.addConstrs((zeta + env.bigM*z[k, 1] <= gp.quicksum(cost_fun(projects[p], xi) * (x[p])
                                                    for p in np.arange(N)) - env.budget - x_0 + env.bigM)
                                                    for k in np.arange(K))
    sep_model.addConstrs((zeta + env.bigM*z[k, 0] <= gp.quicksum(cost_fun(projects[p], xi) * (x[p] + y[k][p])
                                                    for p in np.arange(N)) - env.budget - x_0 - y_0[k] + env.bigM)
                                                    for k in np.arange(K))
    # solve
    sep_model.optimize()
    zeta_sol = zeta.X
    xi_sol = np.array([var.X for i, var in xi.items()])
    z_sol = np.array([var.X for i, var in z.items()])

    return zeta_sol, xi_sol, z_sol
