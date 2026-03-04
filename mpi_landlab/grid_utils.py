"""
This includes the utility functions for global or sub grid
"""

import numpy as np
from scipy.spatial import cKDTree

def get_perimeter_nodes_and_links(points):
    """
    points: (N,2) array, nodes of a regular equilateral-triangle lattice subset (can be concave)
    returns:
      bidx: boundary node indices (np.ndarray)
      edges: perimeter edges as list of (i,j) with i<j (indices into points)
      s, r: estimated spacing and neighbor radius
    """
    pts = np.asarray(points, dtype=float)
    tree = cKDTree(pts)

    # 1) estimate spacing s
    dists, nn = tree.query(pts, k=2)
    s = float(np.median(dists[:, 1]))

    # 2) radius between 1st ring (s) and 2nd ring (sqrt(3)s)
    r = 0.5 * (1.0 + np.sqrt(3.0)) * s

    # 3) build first-ring neighbor lists and degrees
    neigh = tree.query_ball_point(pts, r)
    N = []
    deg = np.empty(len(pts), dtype=int)
    for i, lst in enumerate(neigh):
        lst = [j for j in lst if j != i]
        N.append(set(lst))
        deg[i] = len(lst)

    # 4) boundary nodes
    bmask = deg < 6
    bset = set(np.where(bmask)[0])

    # 5) candidate boundary edges: boundary-boundary and are neighbors
    cand = []
    for i in bset:
        for j in N[i]:
            if j in bset and j > i:
                cand.append((i, j))

    # 6) keep only edges that are not interior (shared neighbors < 2)
    edges = []
    for i, j in cand:
        shared = len(N[i].intersection(N[j]))
        if shared < 2:
            edges.append((i, j))

    return np.array(sorted(bset), dtype=int), edges