import time
from datetime import datetime
import pickle
import numpy as np
import copy

from CapitalBudgetingLoans.BranchBoundFunctions.functions_loans import scenario_fun_build, separation_fun, scenario_fun_update


def algorithm(env, K=4, time_limit=20*60, print_info=False):
    # Initialize
    env.plot_graph()
    N_set = [{k: [] for k in np.arange(K)}]
    N_set[0][0].append(env.init_uncertainty)
    tau_i = copy.deepcopy(N_set[0])
    iteration = 0
    start_time = time.time()
    # initialization for saving stuff
    inc_thetas = dict()
    inc_tau = dict()
    inc_x = dict()
    inc_y = dict()
    prune_count = 0
    inc_tot_nodes = dict()
    cum_tot_nodes = dict()
    tot_nodes = 0
    inc_tot_nodes[0] = 0
    cum_tot_nodes[0] = 0
    prev_save_time = 0
    mp_time = 0
    sp_time = 0
    # initialization of lower and upper bounds
    theta, x, y = (env.upper_bound, [], [])
    theta_i, x_i, y_i = (copy.copy(theta), copy.copy(x), copy.copy(y))
    inc_lb = dict()
    inc_lb[0] = 0
    # K-branch and bound algorithm
    now = datetime.now().time()
    xi_new, k_new = None, None
    rt = 0
    print("Instance {} started at {}".format(env.inst_num, now))
    while N_set and time.time() - start_time < time_limit:
        if xi_new is None:
            # take new node
            tau = N_set.pop(0)
            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_build(K, tau, env)
            mp_time += time.time() - start_mp
        else:
            # make new tau from k_new
            tot_nodes += 1
            tau = copy.deepcopy(tau)
            adj_tau_k = copy.deepcopy(tau[k_new])
            adj_tau_k.append(xi_new)
            tau[k_new] = adj_tau_k
            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_update(K, k_new, xi_new, env, scen_model=model)
            mp_time += time.time() - start_mp
        # prune if theta higher than current robust theta
        if theta - theta_i > -1e-8:
            prune_count += 1
            xi_new = None
            k_new = None
            continue

        # subproblem
        start_sp = time.time()
        zeta, xi, z = separation_fun(K, x, y, theta, env, tau)
        sp_time += time.time() - start_sp

        # check if robust
        if zeta <= 1e-04:
            if print_info:
                now = datetime.now().time()
                print("Instance {}: ROBUST at iteration {} ({}) (time {})   :theta = {},    Xi{},   prune count = {}".format(
                    env.inst_num, iteration, np.round(time.time()-start_time, 3), now, theta, [len(t) for t in tau.values()], prune_count))
            env.plot_graph_solutions(K, y, tau, x=x, tmp=True, it=iteration)
            theta_i, x_i, y_i = (copy.copy(theta), copy.copy(x), copy.copy(y))
            tau_i = copy.deepcopy(tau)
            inc_thetas[time.time() - start_time] = theta_i
            inc_tau[time.time() - start_time] = tau_i
            inc_x[time.time() - start_time] = x_i
            inc_y[time.time() - start_time] = y_i
            prune_count += 1
            xi_new = None
            k_new = None
            if K == 1:
                break
            else:
                continue
        else:
            xi_new = xi
            gap = time.time() - start_time - rt
            rt = time.time() - start_time
            print("Instance {}, it {} gap {} rt {}: Xi{}, theta = {}, zeta = {}".format(env.inst_num, iteration,
                                                                                 np.round(gap, 3), np.round(rt, 3),
                                                                           [len(t) for t in tau.values()], theta, zeta))

        # Create new branches
        full_list = [k for k in np.arange(K) if tau[k]]
        if not full_list:
            K_set = [0]
        elif len(full_list) == K:
            K_prime = K
            K_set = np.arange(K_prime)
        else:
            K_prime = min(K, full_list[-1] + 2)
            K_set = np.arange(K_prime)
        k_new = np.random.randint(len(K_set))
        if K == 1:
            N_set = [1]
        else:
            for k in K_set:
                if k == k_new:
                    continue
                tot_nodes += 1
                tau_tmp = copy.deepcopy(tau)
                adj_tau_k = copy.deepcopy(tau_tmp[k])
                adj_tau_k.append(xi_new)
                tau_tmp[k] = adj_tau_k
                N_set.append(tau_tmp)       # Breadth first

        # save every 10 minutes
        if time.time() - start_time - prev_save_time > 10*60:
            prev_save_time = time.time() - start_time
            # also save inc_tot_nodes
            inc_tot_nodes[time.time() - start_time] = len(N_set)
            cum_tot_nodes[time.time() - start_time] = tot_nodes
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas": inc_thetas, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
            "runtime": time.time() - start_time, "tot_nodes": cum_tot_nodes, "num_nodes_curr": inc_tot_nodes, "mp_time": mp_time, "sp_time": sp_time}

            with open("Results/Decisions/tmp_results_sp_K{}_N{}_inst{}.pickle".format(K, env.N, env.inst_num), "wb") as handle:
                pickle.dump([env, tmp_results], handle)
        iteration += 1
    # termination results
    runtime = time.time() - start_time
    inc_thetas[runtime] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_tot_nodes[runtime] = len(N_set)
    cum_tot_nodes[runtime] = tot_nodes

    now = datetime.now().time()
    print("Instance {} completed at {}, solved in {} minutes".format(env.inst_num, now, runtime/60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas": inc_thetas, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
            "runtime": time.time() - start_time, "tot_nodes": cum_tot_nodes, "num_nodes_curr": inc_tot_nodes, "mp_time": mp_time, "sp_time": sp_time}

    with open("Results/Decisions/final_results_cp_K{}_N{}_inst{}.pickle".format(K, env.N, env.inst_num), "wb") as handle:
        pickle.dump([env, results], handle)
    return results

