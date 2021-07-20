import pickle
import numpy as np

results_list = dict()
for i in np.arange(56):
    with open(f"ShortestPath/Results/Decisions/final_results_sp_2s_K4_N100_inst{i}.pickle", "rb") as handle:
        _, results_list[i] = pickle.load(handle)

