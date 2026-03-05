"""
This includes the utility functions for global or sub grid
"""

import numpy as np
from scipy.spatial import cKDTree


def get_perimeter_nodes_and_links(points):
    """
    points: (N,2) array, nodes of a regular equilateral-triangle lattice subset
    returns:
      bidx: boundary node indices (np.ndarray)
      edges: perimeter edges as list of (i,j) with i<j (indices into points)
      s, r: estimated spacing and neighbor radius
    """
    pts = np.asarray(points, dtype=float)
    tree = cKDTree(pts)

    # 1) estimate spacing s
    # find closest 2 neighbors (self and 1st ring), nn is index of that neighbor
    dists, nn = tree.query(pts, k=2)

    # identify grid spacing as median distance to 1st ring neighbors
    s = float(np.median(dists[:, 1]))

    # 2) radius between 1st ring (s) and 2nd ring (for equilateral-triangle)
    r = 0.5 * (1.0 + np.sqrt(3.0)) * s

    # 3) build first-ring neighbor lists and degrees
    neigh = tree.query_ball_point(pts, r) #idenfity the neighbors within radius r
    N = []
    deg = np.empty(len(pts), dtype=int)
    for i, lst in enumerate(neigh):
        lst = [j for j in lst if j != i]
        N.append(set(lst))  # index of neighbors within radius r
        deg[i] = len(lst)   # number of neighbors within radius r (interior nodes=6)

    # 4) boundary nodes
    bmask = deg < 6
    bset = set(np.where(bmask)[0]) # use set to speed up neighbor checks below

    # 5) candidate boundary edges: boundary-boundary and are neighbors
    cand = []
    for i in bset:
        for j in N[i]:
            if j in bset and j > i:
                cand.append((i, j))

    # 6) keep only edges that are not interior (shared neighbors < 2)
    edges = []
    for i, j in cand:
        # find out how many triangles [i,j] link share
        shared = len(N[i].intersection(N[j]))
        if shared < 2:  # if shared in 2 triangles, it is an interior edge
            edges.append((i, j))

    return np.array(sorted(bset), dtype=int), edges
