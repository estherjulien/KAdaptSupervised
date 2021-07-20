import time
from datetime import datetime
import pickle
import numpy as np
import copy


def algorithm(K, env, scenario_fun_build, scenario_fun_update, separation_fun, time_limit=20*60, print_info=False, problem_type="test", k_adapt_centroid_tau=None):
    # Initialize
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
    inc_thetas_vor = dict()
    inc_tau_vor = dict()
    inc_x_vor = dict()
    inc_y_vor = dict()
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
    theta_i, x_i, y_i = (env.upper_bound, [], [])
    theta_vor, x_vor, y_vor = (env.upper_bound, [], [])
    tau_vor = copy.deepcopy(N_set[0])
    inc_lb = dict()
    inc_lb[0] = 0
    # K-branch and bound algorithm
    now = datetime.now().time()
    xi_new, k_new = None, None
    print("Instance {} started at {}".format(env.inst_num, now))
    while N_set and time.time() - start_time < time_limit:
        if xi_new is None:
            # take new node
            tau = N_set.pop(0)
            # new tau_all
            tau_all = np.array(env.init_uncertainty).reshape(1, env.xi_dim)
            for k in np.arange(K):
                for xi in tau[k]:
                    tau_all = np.vstack([tau_all, xi.reshape(1, env.xi_dim)])
            # master problem
            start_mp = time.time()
            theta, x, y, model = scenario_fun_build(K, tau, env, return_model=True)
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
            # theta, x, y, model = scenario_fun_build(K, tau, env, return_model=True)
            mp_time += time.time() - start_mp
        # prune if theta higher than current robust theta
        if theta - theta_i > -1e-8:
            prune_count += 1
            xi_new = None
            k_new = None
            continue
        elif all([len(t) for t in tau.values()]) and k_adapt_centroid_tau is not None:
            theta_tmp, x_tmp, y_tmp = k_adapt_centroid_tau(K, env, tau)
            if theta_tmp - theta_vor < -1e-2:
                theta_vor, x_vor, y_vor = (copy.deepcopy(theta_tmp), copy.deepcopy(x_tmp), copy.deepcopy(y_tmp))
                tau_vor = copy.deepcopy(tau)
                inc_thetas_vor[time.time() - start_time] = theta_vor
                inc_tau_vor[time.time() - start_time] = tau
                inc_x_vor[time.time() - start_time] = x_vor
                inc_y_vor[time.time() - start_time] = y_vor
                if abs(theta_i - env.upper_bound) < 1e-5:
                    print("Instance {}: VORONOI at iteration {} ({}) (time {})   :theta_vor = {}".format(
                        env.inst_num, iteration, np.round(time.time()-start_time, 3), now, np.round(theta_vor, 4)))
                else:
                    print("Instance {}: VORONOI at iteration {} ({}) (time {})   :ratio = {},   theta_vor = {}".format(
                        env.inst_num, iteration, np.round(time.time()-start_time, 3), now, np.round(theta_vor/theta_i, 3), np.round(theta_vor, 4)))
                try:
                    env.plot_graph_solutions(K, y, tau, x=x, tmp=True, it=iteration, vor_bound=True)
                except AttributeError:
                    pass
        # subproblem
        start_sp = time.time()
        zeta, xi, z = separation_fun(K, x, y, theta, env, tau)
        sp_time += time.time() - start_sp

        # check if robust
        if zeta <= 1e-04:
            if print_info:
                now = datetime.now().time()
                print("Instance {}: ROBUST at iteration {} ({}) (time {})   :theta = {},    Xi{},   prune count = {}".format(
                    env.inst_num, iteration, np.round(time.time()-start_time, 3), now, np.round(theta, 4), [len(t) for t in tau.values()], prune_count))
            try:
                env.plot_graph_solutions(K, y, tau, x=x, tmp=True, it=iteration)
            except AttributeError:
                pass
            theta_i, x_i, y_i = (copy.deepcopy(theta), copy.deepcopy(x), copy.deepcopy(y))
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
            # check if xi_new is already in one of the bags
            # for xi_check in tau_all:
            #     if all(xi_check == xi_new):
            #         print(f"Instance {env.inst_num}: it {iteration}, xi already in tau")
            # tau_all = np.vstack([tau_all, xi_new.reshape(1, env.xi_dim)])        # Create new branches
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
            tmp_results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas": inc_thetas, "inc_x": inc_x,
                            "inc_y": inc_y, "inc_tau": inc_tau, "runtime": time.time() - start_time,
                            "tot_nodes": cum_tot_nodes, "num_nodes_curr": inc_tot_nodes, "mp_time": mp_time, "sp_time": sp_time,
                            "vor": {"theta": theta_vor, "x": x_vor, "y": y_vor, "tau": tau_vor, "inc_thetas": inc_thetas_vor,
                                    "inc_x": inc_x_vor, "inc_y": inc_y_vor, "inc_tau": inc_tau_vor}}
            with open("Results/Decisions/tmp_results_{}_inst{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
                pickle.dump([env, tmp_results], handle)
        iteration += 1
    # termination results
    runtime = time.time() - start_time
    inc_thetas[runtime] = theta_i
    inc_tau[runtime] = tau_i
    inc_x[runtime] = x_i
    inc_y[runtime] = y_i
    inc_thetas_vor[time.time() - start_time] = theta_vor
    inc_tau_vor[time.time() - start_time] = tau_vor
    inc_x_vor[time.time() - start_time] = x_vor
    inc_y_vor[time.time() - start_time] = y_vor
    inc_tot_nodes[runtime] = len(N_set)
    cum_tot_nodes[runtime] = tot_nodes

    now = datetime.now().time()
    print("Instance {} completed at {}, solved in {} minutes".format(env.inst_num, now, runtime/60))
    results = {"theta": theta_i, "x": x_i, "y": y_i, "tau": tau_i, "inc_thetas": inc_thetas, "inc_x": inc_x, "inc_y": inc_y, "inc_tau": inc_tau,
                "runtime": time.time() - start_time, "tot_nodes": cum_tot_nodes, "num_nodes_curr": inc_tot_nodes, "mp_time": mp_time, "sp_time": sp_time,
                "vor": {"theta": theta_vor, "x": x_vor, "y": y_vor, "tau": tau_vor, "inc_thetas": inc_thetas_vor, "inc_x": inc_x_vor, "inc_y": inc_y_vor, "inc_tau": inc_tau_vor}}

    with open("Results/Decisions/final_results_{}_inst{}.pickle".format(problem_type, env.inst_num), "wb") as handle:
        pickle.dump([env, results], handle)

    try:
        env.plot_graph_solutions(K, y_i, tau_i, x=x_i)
    except:
        pass
    return results

