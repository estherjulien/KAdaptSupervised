import numpy as np


def voronoi_hyperlanes(K, centroids, env):
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
            coef[i].append(z - z_b)
            b[i].append(1/2*(np.sum([z_b[j]**2 for j in np.arange(env.xi_dim)]) - np.sum([z[j]**2 for j in np.arange(env.xi_dim)])))
            l += 1
        i += 1
    return coef, b
