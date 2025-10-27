"""
This is the code used to try with pymetis to do grid partition for landlab

Method 2
- grid partition based on nodes

References
https://github.com/inducer/pymetis
https://landlab.readthedocs.io/en/latest/user_guide/grid.html

@Tian Mar 20, 2025
"""

import numpy as np
import pymetis

import matplotlib.pyplot as plt
from landlab import RasterModelGrid

# Create a RasterModelGrid
mg = RasterModelGrid((16, 16), xy_spacing=(1, 1))

# Number of partitions
num_partitions = 5

# Build adjacency list
adjacency_list = []

# useful properties
# mg.core_cells  # id of cells
# mg.number_of_core_cells
# mg.number_of_cell_rows
# mg.number_of_cell_columns
# mg.corners # id of corners
# mg.faces_at_corner
# mg.corner_at

# Method 2 partition based on nodes
# create adjacency list for corners
for node_id in mg.nodes.flat:
    adjacent_nodes = [n for n in mg.adjacent_nodes_at_node[node_id] if n != -1]
    adjacency_list.append(np.array(adjacent_nodes))
    print(node_id, mg.adjacent_nodes_at_node[node_id], adjacent_nodes)

# Partition the grid using pymetis
n_cuts, part_labels = pymetis.part_graph(num_partitions, adjacency=adjacency_list)

# Convert partition labels to a NumPy array
partition_array = np.array(part_labels)

# Assign partition labels to the grid (for visualization)
mg.add_field("grid_partition", partition_array, at="node")

# visualization
fig, ax = plt.subplots(figsize=(16, 14))
mg.at_node["default"] = np.zeros([16, 16])
mg.imshow("default")
ax.scatter(mg.node_x, mg.node_y, c=partition_array, cmap="viridis")
ax.set_title("grid partition based on nodes")
for node_id in mg.nodes.flat:
    ax.annotate(
        f"{node_id}/par{partition_array[node_id]}",
        (mg.node_x[node_id], mg.node_y[node_id]),
        color="black",
        fontsize=8,
        ha="center",
        va="top",
    )
fig.savefig("grid_partition_nodes.png")
