"""
This is the code used to try with pymetis to do grid partition for landlab

Method 1
- grid partition based on corners

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
mg = RasterModelGrid((4, 4), xy_spacing=(1, 1))

# Number of partitions
num_partitions = 4

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

# Method 1 partition based on corners
# create adjacency list for corners
for corner_id in mg.corners.flat:
    faces = mg.faces_at_corner[corner_id]
    adjacent_corners = [mg.corners_at_face[face] for face in faces if face != -1]
    flattened = np.unique(np.concatenate(adjacent_corners))
    flattened = flattened[flattened != corner_id]
    adjacency_list.append(flattened)

# Partition the grid using pymetis
n_cuts, part_labels = pymetis.part_graph(num_partitions, adjacency=adjacency_list)

# Convert partition labels to a NumPy array
partition_array = np.array(part_labels)

# Assign partition labels to the grid (for visualization)
mg.add_field("cell_partition", partition_array, at="corner")

# visualization
fig, ax = plt.subplots(figsize=(6, 8))
mg.at_node["default"] = np.zeros([4, 4])
mg.imshow("default")
ax.scatter(mg.corner_x, mg.corner_y, c=partition_array, cmap="viridis")
ax.set_title("grid partition based on corners")
for node_id in mg.nodes.flat:
    ax.annotate(
        node_id,
        (mg.node_x[node_id], mg.node_y[node_id]),
        color="black",
        fontsize=8,
        ha="center",
        va="center",
    )
for corner_id in mg.corners.flat:
    ax.annotate(
        f"<{corner_id}>/par{partition_array[corner_id]}",
        (mg.corner_x[corner_id], mg.corner_y[corner_id]),
        color="red",
        fontsize=8,
        ha="center",
        va="top",
    )
fig.savefig("grid_partition_corner.png")
