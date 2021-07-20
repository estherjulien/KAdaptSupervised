import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy


class Graph:
    def __init__(self, N, first_stage_ratio=0.2, gamma_perc=0.5, inst_num=0, throw_away_perc=0.1, max_degree=10, min_degree=3):
        self.N = N
        self.gamma_perc = gamma_perc
        self.vertices, init_arcs = self.init_graph(self)
        self.distances = self.update_graph(self, init_arcs, throw_away_perc, max_degree, min_degree)
        self.arcs = self.distances > 1e-5
        self.num_arcs = int(self.arcs.sum())
        self.xi_dim = self.num_arcs
        self.gamma = self.gamma_perc * self.num_arcs
        self.arcs_array = np.array([[i, j] for i in np.arange(self.N) for j in np.arange(self.N) if self.arcs[i, j]])
        self.distances_array = np.array([self.distances[i, j] for i, j in self.arcs_array])
        self.arcs_in, self.arcs_out = self.in_out_arcs(self)
        self.bigM = sum(self.distances_array)*3
        self.upper_bound = sum(self.distances_array)*3
        self.inst_num = inst_num
        self.init_uncertainty = np.zeros(self.num_arcs)
        self.max_first_stage = self.shortest_path(self)*first_stage_ratio

        # list of nodes inside first stage range
        self.inside_range, self.outside_range = self.shortest_path(self, range_used=True)

        # plot graph
        self.plot_graph()

    def plot_graph(self):
        arcs = self.arcs
        sns.set()
        sns.set_style("whitegrid")
        # plot all arcs and vertices
        for i in np.arange(self.N):
            for j in np.arange(self.N):
                if arcs[i, j] < 1e-5:
                    continue
                plt.plot(self.vertices[[i, j], 0], self.vertices[[i, j], 1], "darkgrey")
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], "rx")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig("Results/Plots/init_graph_N{}_inst{}".format(self.N, self.inst_num))
        plt.close()
        return plt

    def plot_graph_solutions(self, K, y, tau, x=None, tmp=False, it=0, svm_bound=False, vor_bound=False, extra=False,
                             decomp=False, feas=False, obj=False, scen=False, att=False):
        arcs = self.arcs
        sns.set()
        sns.set_style("whitegrid")
        # plot all arcs and vertices
        for i in np.arange(self.N):
            for j in np.arange(self.N):
                if arcs[i, j] < 1e-5:
                    continue
                plt.plot(self.vertices[[i, j], 0], self.vertices[[i, j], 1], "darkgrey")
        plt.plot(self.vertices[:, 0], self.vertices[:, 1], "rx")

        # first stage
        if x is not None:
            for a in np.arange(self.num_arcs):
                if x[a] > 0.5:
                    i, j = self.arcs_array[a]
                    plt.plot(self.vertices[[i, j], 0], self.vertices[[i, j], 1], "k-", linewidth=5)

        # second stage
        cols = ["blue", "purple", "green", "red", "yellow", "orange", "grey", "cornflowerblue", "hotpink"]
        # find most expensive y
        max_thetas = np.zeros(K)
        for k in np.arange(K):
            if tau[k]:
                max_thetas[k] = max([sum((1 + xi[a] / 2) * self.distances_array[a] * y[k][a] for a in np.arange(self.num_arcs)) for xi in tau[k]])
            else:
                max_thetas[k] = 0
        k_worst = np.argmax(max_thetas)

        for k in np.arange(K):
            if k == k_worst:
                for a in np.arange(self.num_arcs):
                    if y[k][a] > 0.5:
                        i, j = self.arcs_array[a]
                        plt.plot(self.vertices[[i, j], 0], self.vertices[[i, j], 1], "k", linewidth=3)
            if not tau[k]:
                continue
            for a in np.arange(self.num_arcs):
                if y[k][a] > 0.5:
                    i, j = self.arcs_array[a]
                    plt.plot(self.vertices[[i, j], 0], self.vertices[[i, j], 1], cols[k])

        plt.xlim(0, 10)
        plt.ylim(0, 10)

        if tmp:
            if svm_bound:
                plt.savefig("Results/Plots/tmp_svm_graph_K{}_N{}_inst{}_it{}".format(K, self.N, self.inst_num, it))
            elif extra:
                plt.savefig("Results/Plots/tmp_extra_graph_K{}_N{}_inst{}_it{}".format(K, self.N, self.inst_num, it))
            elif vor_bound:
                plt.savefig("Results/Plots/tmp_vor_graph_K{}_N{}_inst{}_it{}".format(K, self.N, self.inst_num, it))
            elif decomp:
                plt.savefig("Results/Plots/tmp_decomp_graph_K{}_N{}_inst{}_it{}".format(K, self.N, self.inst_num, it))
            elif feas:
                plt.savefig("Results/Plots/tmp_feas_graph_K{}_N{}_inst{}_it{}".format(K, self.N, self.inst_num, it))
            elif obj:
                plt.savefig("Results/Plots/tmp_obj_graph_K{}_N{}_inst{}_it{}".format(K, self.N, self.inst_num, it))
            elif scen:
                plt.savefig("Results/Plots/tmp_scen_graph_K{}_N{}_inst{}_it{}".format(K, self.N, self.inst_num, it))
            elif att:
                plt.savefig("Results/Plots/tmp_att_graph_K{}_N{}_inst{}_it{}".format(K, self.N, self.inst_num, it))
            else:
                plt.savefig("Results/Plots/tmp_graph_K{}_N{}_inst{}_it{}".format(K, self.N, self.inst_num, it))
        else:
            plt.savefig("Results/Plots/final_graph_K{}_N{}_inst{}".format(K, self.N, self.inst_num))
            plt.savefig("Results/Plots/final_graph_K{}_N{}_inst{}.pdf".format(K, self.N, self.inst_num))

        plt.close()

    def adaptive_range(self, K, tau=None, xi=None):
        N = self.N
        # find tau for all scenarios
        if tau is not None:
            tau_matrix = np.array([tau[k] for k in np.arange(K) if tau[k]])[:, 0, :]
        else:
            tau_matrix = np.vstack([self.tau_matrix, xi])
        self.tau_matrix = tau_matrix
        tau_all = np.max(tau_matrix, axis=0)
        # change distance matrix
        new_distances_array = self.distances_array*(1 + 1/2*tau_all)
        new_distances = copy.deepcopy(self.distances)
        for a in np.arange(self.num_arcs):
            new_distances[self.arcs_array[a][0], self.arcs_array[a][1]] = new_distances_array[a]
        # dijkstra on arcs, with node s=0 and t=N
        # initialize stuff
        tmp_bigM = 10**2
        dist = np.array([0, *np.ones(N-1)*tmp_bigM], dtype=np.float)
        Q = list(np.arange(N))
        prev = dict()
        # algorithm
        connected = False
        while Q and not connected:
            i = Q[np.argmin(dist[Q])]
            Q.remove(i)

            for j in np.arange(N):
                if self.arcs[i, j] < 1e-5:
                    continue
                alt = dist[i] + new_distances[i, j]
                if alt < dist[j]:
                    if j == N - 1:
                        connected = True
                    dist[j] = alt
                    prev[j] = i
        inside_range = np.where(dist < self.max_first_stage)[0]
        # outside_range = [i for i in np.arange(self.N) if i not in inside_range]
        return inside_range

    @staticmethod
    def shortest_path(self, range_used=False):
        N = self.N
        # dijkstra on arcs, with node s=0 and t=N
        # initialize stuff
        tmp_bigM = 10**2
        dist = np.array([0, *np.ones(N-1)*tmp_bigM], dtype=np.float)
        Q = list(np.arange(N))
        prev = dict()
        # algorithm
        connected = False
        while Q and not connected:
            i = Q[np.argmin(dist[Q])]
            Q.remove(i)

            for j in np.arange(N):
                if self.arcs[i, j] < 1e-5:
                    continue
                alt = dist[i] + self.distances[i, j]
                if alt < dist[j]:
                    if j == N - 1:
                        connected = True
                    dist[j] = alt
                    prev[j] = i

        if range_used:
            inside_range = np.where(dist*1.5 < self.max_first_stage)[0]
            outside_range = np.where(dist > 1.1*self.max_first_stage)[0]
            return inside_range, outside_range
        else:
            return dist[N-1]

    @staticmethod
    def isconnected(self, arcs):
        N = self.N
        # dijkstra on arcs, with node s=0 and t=N
        # initialize stuff
        tmp_bigM = 10**2
        dist = np.array([0, *np.ones(N-1)*tmp_bigM])
        Q = list(np.arange(N))
        prev = dict()
        # algorithm
        connected = False
        while Q and not connected:
            i = Q[np.argmin(dist[Q])]
            Q.remove(i)

            for j in np.arange(N):
                if arcs[i, j] < 1e-5:
                    continue
                alt = dist[i] + arcs[i, j]
                if alt < dist[j]:
                    if j == N - 1:
                        connected = True
                        break
                    dist[j] = alt
                    prev[j] = i
        return connected

    @staticmethod
    def vertices_fun(self):
        vertices_set = np.zeros([self.N, 2], dtype=np.float)
        num_nodes = 1
        # start and terminal node are the first and last one, on 0,0 and 10,10, respectively
        vertices_set[0] = [0, 0]
        vertices_set[self.N - 1] = [10, 10]
        while num_nodes < self.N-1:
            x, y = np.random.uniform(0, 10, 2)
            if (2 < x < 4 and y < 8) or (6 < x < 8 and y > 2):      # (2 < x < 4 and y < 8) or (6 < x < 8 and y > 2)
                pass
            else:
                vertices_set[num_nodes] = [x, y]
                num_nodes += 1
        return vertices_set

    @staticmethod
    def init_graph(self):
        connected = False
        while not connected:
            vertices = self.vertices_fun(self)
            N = self.N
            # make initial arcs
            arcs = np.ones([N, N], dtype=np.float)
            for i in np.arange(N):
                if i == N-1:
                    arcs[i, :] = np.zeros(N)
                for j in np.arange(N):
                    if j == 0:
                        arcs[:, j] = np.zeros(N)
                    if i == j:
                        arcs[i, j] = 0
                        continue
                    x_i, y_i = vertices[i]
                    x_j, y_j = vertices[j]
                    x_mid, y_mid = [(x_i + x_j)/2, (y_i + y_j)/2]
                    x_qt, y_qt = [x_i*1/4 + x_j*3/4, y_i*1/4 + y_j*3/4]
                    x_tqt, y_tqt = [x_i*3/4 + x_j*1/4, y_i*3/4 + y_j*1/4]
                    if (2 < x_mid < 4 and y_mid < 8) or (6 < x_mid < 8 and y_mid > 2):
                        arcs[i, j] = 0
                    elif (2 < x_qt < 4 and y_qt < 8) or (6 < x_qt < 8 and y_qt > 2):
                        arcs[i, j] = 0
                    elif (2 < x_tqt < 4 and y_tqt < 8) or (6 < x_tqt < 8 and y_tqt > 2):
                        arcs[i, j] = 0
            if self.isconnected(self, arcs):
                connected = True
            else:
                print("again")

        return vertices, arcs

    @staticmethod
    def update_graph(self, arcs, throw_away_perc, max_degree, min_degree):
        vertices = self.vertices
        N = self.N
        # delete arcs with middle in no go zones
        arc_dict = dict()
        for i in np.arange(N):
            for j in np.arange(N):
                if arcs[i, j] < 1e-5:
                    continue
                x_i, y_i = vertices[i]
                x_j, y_j = vertices[j]
                distance = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                arcs[i, j] = distance
                arc_dict[(i, j)] = distance
        # first check
        if not self.isconnected(self, arcs):
            print("Not connected")
        # delete long arcs (first sort and then sth percent)
        arc_dict_order = {k: v for k, v in sorted(arc_dict.items(), key=lambda item: -item[1])}
        # delete "throw_away_perc" longest arcs
        throw_away_num = np.floor(len(arc_dict_order)*throw_away_perc)
        del_arc = 0
        while del_arc < throw_away_num:
            # check here if when you delete this, degree out and in will be >= 1 and total degree >= min_degree
            i, j = list(arc_dict_order.keys())[del_arc]
            # check in degree of j
            if sum([arcs[:, j] > 1e-5][0]) < 1:
                continue
            # check out degree of i
            if sum([arcs[i, :] > 1e-5][0]) < 1:
                continue
            # check each time if you can delete this arc based on connected graph
            distance = copy.copy(arcs[i, j])
            arcs[i, j] = 0
            if not self.isconnected(self, arcs):
                arcs[i, j] = distance
            del_arc += 1

        # delete arcs with too many neighbours
        for v in np.arange(N):
            # in degree
            rp_in = np.random.permutation(N)
            in_degree = sum([arcs[:, v] > 1e-5][0])
            if in_degree > max_degree:
                for i in rp_in:
                    if arcs[i, v] < 1e-5:
                        continue
                    i_out_degree = sum([arcs[i, :] > 1e-5][0])
                    if i_out_degree <= min_degree:
                        continue
                    distance = copy.copy(arcs[i, v])
                    arcs[i, v] = 0
                    if not self.isconnected(self, arcs):
                        arcs[i, v] = distance
                    else:
                        in_degree -= 1
                    if in_degree == max_degree:
                        break
            # out degree
            rp_out = np.random.permutation(N)
            out_degree = sum([arcs[v, :] > 1e-5][0])
            if out_degree > max_degree:
                for j in rp_out:
                    if arcs[v, j] < 1e-5:
                        continue
                    j_in_degree = sum([arcs[:, j] > 1e-5][0])
                    if j_in_degree <= min_degree:
                        continue
                    distance = copy.copy(arcs[v, j])
                    arcs[v, j] = 0
                    if not self.isconnected(self, arcs):
                        arcs[v, j] = distance
                    else:
                        out_degree -= 1
                    if out_degree == max_degree:
                        break

        return arcs

    @staticmethod
    def in_out_arcs(self):
        arcs_in = {i: [] for i in np.arange(self.N)}
        arcs_out = {i: [] for i in np.arange(self.N)}
        for a in np.arange(self.num_arcs):
            i, j = self.arcs_array[a]
            arcs_in[j].append(a)
            arcs_out[i].append(a)
        return arcs_in, arcs_out
