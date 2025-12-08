"""
This is to create a small voronoi graph to debug the links not connected error

Methods:
This needs changes of the landlab code in landlab/src/landlab/graph/object/ext/at_patch.pyx

Key findings:
- the error info comes from at_patch.pyx line 54
- _get_nodes_at_patch is at at_patch.pyx line 103
- use "with gil" to print out the link and node info
- cell 1 has issue with overlap corners 0,1,2
- link pair (2,1) and link pair (0,3) has issue for common nodes
- for overlap links: link 0 [0,1], link1 [0,2], link2 [1,2]
"""

import numpy as np
import os
import matplotlib.pyplot as plt

from landlab.plot import plot_graph
from landlab import HexModelGrid
from landlab.graph import DualVoronoiGraph

# create output dir
output_dir = os.path.join(os.getcwd(), "experiment", "debug_links_error")
os.makedirs(output_dir, exist_ok=True)

# define hex model grid
grid = HexModelGrid((8, 11), node_layout="rect")
plt.figure(figsize=(20, 20))
plot_graph(grid, at="node,link")
plt.savefig(os.path.join(output_dir,'hex_graph.png'))

# remove nodes and define perimeter_links
x = grid.x_of_node
y = grid.y_of_node

ranges = [(0,10), (11,20), (22,31), (33,41), (44,52), (55,61), (66,72)]
remove_ind = [i for start, end in ranges for i in range(start, end)]

perimeter_links = [
    [0,2],[2,4],[4,7],[7,10],[10,15],[15,20],[20,31],
    [21,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],[29,30],[30,31],
    [0,1],[1,3],[3,5],[5,8],[8,12],[11,12],[11,16],[16,26],
]

# define vor graph
vor_x = np.delete(x,remove_ind)
vor_y = np.delete(y,remove_ind)

graph = DualVoronoiGraph((vor_y, vor_x), sort=True,
                         perimeter_links=perimeter_links,
                        )
plt.figure(figsize=(16, 16))
plot_graph(graph, at="node,link")
plt.savefig(os.path.join(output_dir,'vor_graph.png'))

# plot cell graph
plt.figure(figsize=(16, 16))
plot_graph(graph, at="cell")

plt.figure(figsize=(20,20))
plot_graph(graph, at="cell,face,corner",)
plt.savefig(os.path.join(output_dir, 'vor_cell.png'))

plt.figure(figsize=(20,20))
plot_graph(graph, at="corner",)
plt.savefig(os.path.join(output_dir, 'vor_corner.png'))


# change of at_patch.pyx see # my changes section

# cimport cython
# from cython.parallel cimport prange
# from libc.stdint cimport int8_t
#
# # https://cython.readthedocs.io/en/stable/src/userguide/fusedtypes.html
# ctypedef fused float_or_int:
#     cython.floating
#     cython.integral
#     long long
#     int8_t
#
# ctypedef fused id_t:
#     cython.integral
#     long long
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def get_rightmost_edge_at_patch(
#     const id_t [:, :] links_at_patch,
#     const cython.floating [:, :] xy_of_link,
#     id_t [:] edge,
# ):
#     cdef int n_patches = links_at_patch.shape[0]
#     cdef int n_cols = links_at_patch.shape[1]
#     cdef int patch
#     cdef int link
#     cdef int n
#     cdef int max_n
#     cdef double max_x
#
#     for patch in prange(n_patches, nogil=True, schedule="static"):
#         link = links_at_patch[patch, 0]
#         max_x, max_n = xy_of_link[link][0], 0
#
#         for n in range(1, n_cols):
#             link = links_at_patch[patch, n]
#             if link == -1:
#                 break
#             if xy_of_link[link][0] > max_x:
#                 max_x, max_n = xy_of_link[link][0], n
#         edge[patch] = max_n
#
#
# cdef id_t find_common_node(
#     const id_t * link_a,
#     const id_t * link_b,
# ) noexcept nogil:
#     if link_a[0] == link_b[0] or link_a[0] == link_b[1]:
#         return link_a[0]
#     elif link_a[1] == link_b[0] or link_a[1] == link_b[1]:
#         return link_a[1]
#     else:
#         with gil:  # my changes
#             print(f"error: link_a=({link_a[0]}, {link_a[1]}), "
#                   f"link_b=({link_b[0]}, {link_b[1]})")
#             import sys
#             sys.stdout.flush()
#         raise ValueError("links are not connected")
#
#
# cdef long all_nodes_at_patch(
#     const id_t * links_at_patch,
#     long max_links,
#     const id_t * nodes_at_link,
#     long * out,
# ) noexcept nogil:
#     cdef long n_links = max_links
#     cdef long link
#     cdef long i
#     cdef long n_nodes = 0
#
#     while links_at_patch[n_links - 1] == -1:
#         n_links -= 1
#
#     for i in range(n_links):
#         link = links_at_patch[i]
#
#         out[n_nodes] = nodes_at_link[link * 2]
#         out[n_nodes + 1] = nodes_at_link[link * 2 + 1]
#
#         n_nodes += 2
#
#     return n_links
#
#
# cdef void order_nodes_at_patch(
#     const id_t * all_nodes_at_patch,
#     id_t * out,
#     const long n_vertices,
# ):
#     cdef long i
#     cdef long vertex
#
#     out[0] = all_nodes_at_patch[1]
#     for vertex in range(n_vertices - 1):
#         i = vertex * 2
#         while all_nodes_at_patch[i] != out[vertex]:
#             i += 1
#         if i % 2 == 0:
#             out[vertex + 1] = all_nodes_at_patch[i + 1]
#         else:
#             out[vertex + 1] = all_nodes_at_patch[i - 1]
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def get_nodes_at_patch(
#     const id_t [:, :] links_at_patch,
#     const id_t [:, :] nodes_at_link,
#     id_t [:, :] nodes_at_patch,
# ):
#     cdef int n_patches = links_at_patch.shape[0]
#     cdef int max_links_at_patch = links_at_patch.shape[1]
#     cdef int patch
#
#     for patch in prange(n_patches, nogil=True, schedule="static"):
#         with gil: # my changes
#             print(f"Start patch {patch}, links: {list(links_at_patch[patch])}")
#             import sys
#             sys.stdout.flush()   # force flush
#
#         _nodes_at_patch(
#             &links_at_patch[patch, 0],
#             max_links_at_patch,
#             &nodes_at_link[0, 0],
#             &nodes_at_patch[patch, 0],
#         )
#
#         with gil: # my changes
#             print(f"Finish Processing patch {patch}")
#             sys.stdout.flush()
#
#
# cdef long _nodes_at_patch(
#     const id_t * links_at_patch,
#     const long max_links,
#     const id_t * nodes_at_link,
#     id_t * out,
# ) noexcept nogil:
#     cdef long n_links = max_links
#     cdef long link, next_link
#     cdef long i
#
#     while links_at_patch[n_links - 1] == -1:
#         n_links -= 1
#
#     next_link = links_at_patch[0]
#     for i in range(0, n_links - 1):
#         link, next_link = next_link, links_at_patch[i + 1]
#
#         out[i] = find_common_node(
#             &nodes_at_link[link * 2], &nodes_at_link[next_link * 2]
#         )
#
#         with gil: # my changes
#             print(f"Link pair ({link}, {next_link}) -> common node: {out[i]}")
#             import sys
#             sys.stdout.flush()
#
#     link, next_link = links_at_patch[n_links - 1], links_at_patch[0]
#     out[n_links - 1] = find_common_node(
#         &nodes_at_link[link * 2], &nodes_at_link[next_link * 2]
#     )
#
#     return n_links
