"""
This is to test how to remove the extra links and cells for voronoi grid when
representing hex model grid (without partition)

method:
identify the tail and head nodes of each perimeter links of the hex grid,
pass these nodes list to voronoi graph as perimeter_link
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from landlab.grid import HexModelGrid, VoronoiDelaunayGrid
from landlab.graph import DualVoronoiGraph
from landlab.plot import plot_graph


# create output dir
output_dir = os.path.join(os.getcwd(), "experiment", "remove_extra_links_hex_new_test")
os.makedirs(output_dir, exist_ok=True)

# create 3 by 3 Hex Grid
hex_grid = HexModelGrid((10, 10), spacing=1, node_layout="rect")

for option in ["node", "link", "cell"]:
    fig, ax = plt.subplots(figsize=(16, 16))
    plot_graph(hex_grid, at=option, axes=ax)
    ax.set_title(f"hex_grid_{option}")
    fig.savefig(os.path.join(output_dir, f"hex_{option}.png"))
    plt.close(fig)

# identify perimeter links in Hex Grid
# boundary_node_types = classify_boundary_nodes(hex_grid)
boundary_node_types = {
    "left": hex_grid.nodes_at_left_edge,
    "right": hex_grid.nodes_at_right_edge,
    "bottom": hex_grid.nodes_at_bottom_edge,
    "top": hex_grid.nodes_at_top_edge,
}

boundary_links = []
boundary_links_tail_head_nodes = []
for _, node_list in boundary_node_types.items():
    for node in node_list:
        links = hex_grid.links_at_node[node]
        for link in links:
            if link != -1:
                tail_head_nodes = hex_grid.nodes_at_link[link]
                mask = np.isin(tail_head_nodes, node_list)
                if np.all(mask) and (link not in boundary_links):
                    boundary_links.append(link)
                    boundary_links_tail_head_nodes.append(tail_head_nodes)

# boundary_links_tail_head_nodes2 = []
# for link in sorted(boundary_links):
#     boundary_links_tail_head_nodes2.append(hex_grid.nodes_at_link[link])


# create voronoi grid without perimeter links
x = hex_grid.node_x
y = hex_grid.node_y
vor_grid = VoronoiDelaunayGrid(x, y)

for option in ["node", "link", "cell"]:
    fig, ax = plt.subplots(figsize=(16, 16))
    plot_graph(vor_grid, at=option, axes=ax)
    ax.set_title(f"vor_grid_{option}")
    fig.savefig(os.path.join(output_dir, f"vor_grid_{option}.png"))
    plt.close(fig)


# create voronoi grid perimeter links
x = hex_grid.node_x
y = hex_grid.node_y
vor_grid = VoronoiDelaunayGrid(x, y, perimeter_links=boundary_links_tail_head_nodes)

for option in ["node", "link", "cell"]:
    fig, ax = plt.subplots(figsize=(16, 16))
    plot_graph(vor_grid, at=option, axes=ax)
    ax.set_title(f"vor_grid_{option}")
    fig.savefig(os.path.join(output_dir, f"vor_grid_adj_{option}.png"))
    plt.close(fig)

# create voronoi graph with perimeter links
vor_graph = DualVoronoiGraph(
    (y, x), sort=True, perimeter_links=boundary_links_tail_head_nodes
)
for option in ["node", "link", "cell"]:
    fig, ax = plt.subplots(figsize=(16, 16))
    plot_graph(vor_graph, at=option, axes=ax)
    ax.set_title(f"vor_graph_{option}")
    fig.savefig(os.path.join(output_dir, f"vor_graph_{option}.png"))
    plt.close(fig)
