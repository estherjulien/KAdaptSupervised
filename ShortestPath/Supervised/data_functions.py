import pandas as pd
import numpy as np
import pickle


def train_data_prep(problem_type, two_stage=True):
    # open results
    env = dict()
    results = dict()
    i = 0
    while True:
        try:
            with open(f"./Results/Decisions/final_results_{problem_type}_inst{i}.pickle", "rb") as handle:
                env[i], results[i] = pickle.load(handle)
        except:
            break
        i += 1
    inst_num = i
    K = len(results[0]["y"])

    # store input data
    X = pd.DataFrame(columns=["dist_ratio", "dist_to_s", "dist_to_t", "in_x_range", "in_y_range"], dtype=np.float16)
    for i in np.arange(inst_num):
        avg_path_dist = np.mean(env[i].distances_array)
        for a in np.arange(env[i].num_arcs):
            # initialize row
            X.loc[f"arc{a}_inst{i}"] = [0, np.nan, np.nan, 0, 0]
            X.loc[f"arc{a}_inst{i}", "dist_ratio"] = env[i].distances_array[a]/avg_path_dist
        for a_in in env[i].inside_range:
            X.loc[f"arc{a_in}_inst{i}", "in_x_range"] = 1
        for a_out in env[i].outside_range:
            X.loc[f"arc{a_out}_inst{i}", "in_y_range"] = 1

    # store output data
    Y = pd.DataFrame(columns=np.arange(K))
    for i in np.arange(inst_num):
        # decide to use either vor or normal decisions
        if results[i]["vor"]["theta"] < results[i]["theta"]:
            use_normal = False
        else:
            use_normal = True
        # find K centroids
        if use_normal:
            centroid = np.array([np.mean(np.array(results[i]["tau"][k]), axis=0) for k in np.arange(K)])
            # order subsets based on objective value
            theta_subsets = dict()
            x = results[i]["x"]
            y = results[i]["y"]
            tau = results[i]["tau"]
        else:
            centroid = np.array([np.mean(np.array(results[i]["vor"]["tau"][k]), axis=0) for k in np.arange(K)])
            # order subsets based on objective value
            theta_subsets = dict()
            x = results[i]["vor"]["x"]
            y = results[i]["vor"]["y"]
            tau = results[i]["vor"]["tau"]

        for k in np.arange(K):
            if two_stage:
                theta_subsets[k] = max([sum((1 + xi[a] / 2) * env[i].distances_array[a] * (x[a] + y[k][a]) for a in np.arange(env[i].num_arcs)) for
                     xi in tau[k]])
            else:
                theta_subsets[k] = max([sum((1 + xi[a] / 2) * env[i].distances_array[a] * y[k][a] for a in np.arange(env[i].num_arcs)) for
                     xi in tau[k]])
        order_subset = [k for k, val in sorted(theta_subsets.items(), key=lambda item: item[1])]
        for a in np.arange(env[i].num_arcs):
            k_sort = 0
            for k in order_subset:
                Y.loc[f"arc{a}_inst{i}", k_sort] = centroid[k][a]
                k_sort += 1

    return X, Y

