"""This is to experiment how to identify the hollow/artificial nodes of each partition

In this code, it idetified the artificial nodes and make them as send_to recv_from to
send their node value of the grid.
"""

import os
import numpy as np
import pymetis

import matplotlib.pyplot as plt

from collections import defaultdict

from landlab import HexModelGrid, VoronoiDelaunayGrid

# create output dir for global grid
output_dir = os.path.join(os.getcwd(), 'experiment/hollow_nodes_test')
os.makedirs(output_dir, exist_ok=True)

## step1: define global grid
mg = HexModelGrid((17, 17), spacing=1, node_layout='rect')
z = mg.add_zeros("topographic__elevation", at="node")
cum_depo = mg.add_zeros("total_deposit__thickness", at="node")

midpoint = 8
dx = np.abs(mg.x_of_node - midpoint)
dy = np.abs(mg.y_of_node - midpoint)
ds = np.sqrt(dx * dx + dy * dy)
z[:] = (midpoint - ds) - 3.0
z[z < -3.0] = -3.0
z0 = z.copy()

boundary_nodes = mg.boundary_nodes

## step2: grid partition
adjacency_list = []
num_partitions=4

# create adjacency list for corners
for node_id in mg.nodes.flat:
    adjacent_nodes = [n for n in mg.adjacent_nodes_at_node[node_id] if n != -1]
    adjacency_list.append(np.array(adjacent_nodes))
    # print(node_id, mg.adjacent_nodes_at_node[node_id], adjacent_nodes)

# Partition the grid using pymetis
n_cuts, part_labels = pymetis.part_graph(num_partitions, adjacency=adjacency_list)

# Convert partition labels to a NumPy array
partition_array = np.array(part_labels)

# visualization
fig, ax = plt.subplots(figsize=[16, 14])
ax.scatter(mg.node_x, mg.node_y, c=partition_array, cmap='viridis')
ax.set_title('grid partition based on nodes')
for node_id in mg.nodes.flat:
    ax.annotate(f"{node_id}/par{partition_array[node_id]}",
                (mg.node_x[node_id], mg.node_y[node_id]),
                color='black', fontsize=8, ha='center', va='top')
fig.savefig(os.path.join(output_dir, 'global_grid_partition.png'))
plt.close(fig)

## step3 Identify ghost and hollow nodes
results_recv = {}

for rank in range(0, num_partitions):

    # local nodes for each rank
    local_nodes = [node for node, part in enumerate(part_labels) if part == rank]

    # ghost nodes for each rank
    # get recv_from info
    # send_to = defaultdict(set)
    recv_from = defaultdict(set)

    for node in local_nodes:
        for neighbor in adjacency_list[node]:
            neighbor_part = part_labels[neighbor]
            if neighbor_part != rank:
                # print(neighbor_part,node)
                recv_from[neighbor_part].add(neighbor)
                # send_to[neighbor_part].add(node)
                # for test_node in adjacency_list[node]:
                #     test_node_part = part_labels[test_node]
                #     if test_node_part == rank:
                #         send_to[neighbor_part].add(test_node)


    ghost_nodes = [int(node) for pid, node_list in recv_from.items() for node in
                   node_list]


    # hollow nodes for each rank
    hollow_nodes = []
    for ghost_node in ghost_nodes:
        for ghost_neighbor in adjacency_list[ghost_node]:
            neighbor_part = part_labels[ghost_neighbor]
            if (neighbor_part != rank) and (ghost_neighbor not in ghost_nodes):
                recv_from[neighbor_part].add(ghost_neighbor)
                if ghost_neighbor not in hollow_nodes:
                    hollow_nodes.append(ghost_neighbor)

    results_recv[rank] = recv_from

    # print(f"rank: {rank}")
    # print(f"send_to: {send_to}")
    # print(f"recv_from: {recv_from}")

    # define local grid and make plot
    vmg_global_ind = sorted(local_nodes + ghost_nodes + hollow_nodes)
    x = mg.node_x[vmg_global_ind]
    y = mg.node_y[vmg_global_ind]
    elev = mg.at_node["topographic__elevation"][vmg_global_ind]

    local_ghost_nodes_ind = [vmg_global_ind.index(val) for val in ghost_nodes]
    local_hollow_nodes_ind = [vmg_global_ind.index(val) for val in hollow_nodes]
    local_boundary_nodes = [node for node in local_nodes+ghost_nodes if node in boundary_nodes]
    local_boundary_nodes_ind = [vmg_global_ind.index(val) for val in
                                sorted(hollow_nodes + local_boundary_nodes)]

    local_vmg = VoronoiDelaunayGrid(x.tolist(), y.tolist())
    local_vmg.add_field("topographic__elevation", elev, at="node")
    local_vmg.status_at_node[
        local_boundary_nodes_ind] = local_vmg.BC_NODE_IS_FIXED_VALUE

    # make a plot
    fig, ax = plt.subplots(figsize=[18, 14])
    sc = ax.scatter(local_vmg.node_x, local_vmg.node_y,
                    c=local_vmg.at_node["topographic__elevation"], cmap="coolwarm",
                    vmin=-3)
    ax.set_title(f'subgrid nodes rank={rank}')
    for node_id in local_boundary_nodes_ind:
        ax.annotate(f"B",
                    (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
                    color='blue', fontsize=10, ha='left', va='bottom')
    for node_id in local_ghost_nodes_ind:
        ax.annotate(f"G",
                    (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
                    color='red', fontsize=10, ha='right', va='bottom')
    for node_id in local_hollow_nodes_ind:
        ax.annotate(f"H",
                    (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
                    color='green', fontsize=10, ha='right', va='bottom')
    for node_id in vmg_global_ind:
        ax.annotate(f"{node_id}/par{partition_array[node_id]}",
                    (mg.node_x[node_id], mg.node_y[node_id]),
                    color='black', fontsize=8, ha='center', va='top')

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Elevation (m)')
    fig.savefig(os.path.join(output_dir, f'subgrid_for_rank{rank}.png'))
    plt.close(fig)


# create send_to nodes
results_send = {}

for rank in range(0, num_partitions):
    results_send[rank] = {}
    for key, values in results_recv.items():
        if key != rank:
            results_send[rank][key] = values[rank]


# validate the send_to and recv_from
for rank in range(0, num_partitions):
    for key in results_recv.keys():
        if key != rank:
            a = results_recv[rank][key]
            b = results_send[key][rank]
            if a != b:
                print(f"error! results[{rank}]['recv_from'][{key}]")
                print(f"recv_from:{a}")
                print(f"send_to:{b}")

print("Done!!")