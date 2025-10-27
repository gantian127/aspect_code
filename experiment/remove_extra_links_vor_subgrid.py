"""
This is to test how to remove the extra links and cells from the sub voronoi grid based
on the global hex model grid partition

method:
identify the tail and head nodes of each perimeter links of subgrid,
pass these nodes list to voronoi graph as perimeter_link
"""

import os
import json
import numpy as np
import pymetis

import matplotlib.pyplot as plt

from collections import defaultdict

from landlab import HexModelGrid, VoronoiDelaunayGrid
from landlab.graph import DualVoronoiGraph
from landlab.plot.graph import plot_graph


## user settings
num_partitions = 3
hex_grid_shape = (10,10)

create_vor_grid = False
create_data_file = True

# create output dir
output_dir = os.path.join(os.getcwd(),"experiment",
                          f"remove_extra_links_vor_subgrid_{hex_grid_shape}_{num_partitions}")
os.makedirs(output_dir, exist_ok=True)

## step1: define global grid
mg = HexModelGrid(hex_grid_shape, spacing=1, node_layout='rect')
z = mg.add_zeros("topographic__elevation", at="node")

midpoint = 8
dx = np.abs(mg.x_of_node - midpoint)
dy = np.abs(mg.y_of_node - midpoint)
ds = np.sqrt(dx * dx + dy * dy)
z[:] = (midpoint - ds) - 3.0
z[z < -3.0] = -3.0
z0 = z.copy()

boundary_nodes = mg.boundary_nodes

# make graph plots for hex model
for option in ['node', 'link', 'cell']:
    fig, ax = plt.subplots(figsize=(16, 16))
    plot_graph(mg, at=option, axes=ax,
               with_id=True,
               #fontsize=7
               )
    ax.set_title(f'hex_grid_{option}')
    fig.savefig(os.path.join(output_dir, f'hex_grid_{option}.png'))
    plt.close(fig)

## step2: grid partition
adjacency_list = []

# create adjacency list for corners
for node_id in mg.nodes.flat:
    adjacent_nodes = [n for n in mg.adjacent_nodes_at_node[node_id] if n != -1]
    adjacency_list.append(np.array(adjacent_nodes))
    # print(node_id, mg.adjacent_nodes_at_node[node_id], adjacent_nodes)

# Partition the grid using pymetis
n_cuts, part_labels = pymetis.part_graph(num_partitions, adjacency=adjacency_list)

# Convert partition labels to a NumPy array
partition_array = np.array(part_labels)

# plot partition results
fig, ax = plt.subplots(figsize=[16, 14])
ax.scatter(mg.node_x, mg.node_y, c=partition_array, cmap='viridis')
ax.set_title('grid partition based on nodes')
for node_id in mg.nodes.flat:
    ax.annotate(f"{node_id}/par{partition_array[node_id]}",
                (mg.node_x[node_id], mg.node_y[node_id]),
                color='black', fontsize=8, ha='center', va='top')
fig.savefig(os.path.join(output_dir, 'global_grid_partition.png'))
plt.close(fig)
print('grid partition done')

## step3 Identify ghost and hollow nodes
for rank in range(0, num_partitions):
    print(f"start rank {rank}")
    # get local nodes
    local_nodes = [node for node, part in enumerate(part_labels) if part == rank]

    # get send_to and recv_from info
    send_to = defaultdict(set)
    recv_from = defaultdict(set)

    for node in local_nodes:
        for neighbor in adjacency_list[node]:
            neighbor_part = part_labels[neighbor]
            if neighbor_part != rank:
                # print(neighbor_part,node)
                send_to[neighbor_part].add(node)
                recv_from[neighbor_part].add(neighbor)

    # get ghost and artificial nodes
    ghost_nodes = [int(node) for pid, node_list in recv_from.items() for node in
                   node_list]
    artificial_nodes = []
    for ghost_node in ghost_nodes:
        for ghost_neighbor in adjacency_list[ghost_node]:
            neighbor_part = part_labels[ghost_neighbor]
            if (neighbor_part != rank) and (ghost_neighbor not in ghost_nodes):
                recv_from[neighbor_part].add(ghost_neighbor)
                if ghost_neighbor not in artificial_nodes:
                    artificial_nodes.append(ghost_neighbor)

    # identify node index for different types
    vmg_global_ind = sorted(local_nodes + ghost_nodes + artificial_nodes)
    local_nodes_ind = [vmg_global_ind.index(val) for val in sorted(local_nodes)]
    local_ghost_nodes_ind = [vmg_global_ind.index(val) for val in sorted(ghost_nodes)]
    local_artificial_nodes_ind = [vmg_global_ind.index(val) for val in
                                  sorted(artificial_nodes)]

    local_boundary_nodes = [node for node in local_nodes + ghost_nodes if
                            node in boundary_nodes]
    local_boundary_nodes_ind = [vmg_global_ind.index(val) for val in
                                sorted(artificial_nodes + local_boundary_nodes)]

    # get x, y and elevation data
    x = mg.node_x[vmg_global_ind]
    y = mg.node_y[vmg_global_ind]
    elev = mg.at_node["topographic__elevation"][vmg_global_ind]

    ## create voronoi grid
    if create_vor_grid:
        local_vmg = VoronoiDelaunayGrid(x.tolist(), y.tolist())
        local_vmg.status_at_node[
            local_boundary_nodes_ind] = local_vmg.BC_NODE_IS_FIXED_VALUE

        # make graph plot for vor grid
        for option in ['node','link','cell']:
            fig, ax = plt.subplots(figsize=(16,16))
            plot_graph(local_vmg, at=option,axes=ax,
                        with_id=True,
                        #fontsize=7
            )
            fig.savefig(os.path.join(output_dir,f'vor_grid_{option}_{rank}.png'))
            plt.close(fig)

        # make partition plot for vor grid
        fig, ax = plt.subplots(figsize=[18, 14])
        sc = ax.scatter(local_vmg.node_x, local_vmg.node_y,
                        c=np.arange(len(local_vmg.node_x)), cmap="coolwarm",
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
        for node_id in local_artificial_nodes_ind:
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


    ## Create vor graph
    subgrid_boundary_nodes = sorted(artificial_nodes + local_boundary_nodes)
    global_boundary_node_types = {
        "left": mg.nodes_at_left_edge,
        "right": mg.nodes_at_right_edge,
        "bottom": mg.nodes_at_bottom_edge,
        "top": mg.nodes_at_top_edge,
    }

    # get subgrid boundary node types
    local_node_structure = [
        sorted([node for node in subgrid_boundary_nodes if left <= node <= right])
        for left, right in
        zip(global_boundary_node_types['left'], global_boundary_node_types['right'])
    ]

    local_node_structure = [row for row in local_node_structure if row] # remove empty list

    local_boundary_node_types = {
        'bottom': local_node_structure[0],
        'top': local_node_structure[-1],
        'left':[local_node_structure[0][0], local_node_structure[-1][0]],
        'right':[local_node_structure[0][-1], local_node_structure[-1][-1]],
    }

    for row in local_node_structure[1:-1]:
        if len(row)<=2:
            local_boundary_node_types['left'].append(row[0])
            local_boundary_node_types['right'].append(row[-1])
        else:
            for i in range(0, len(row)):
                if row[i+1] - row[i]!=1:
                    split_ind = i+1
                    break
            local_boundary_node_types['left'] += row[:split_ind]
            local_boundary_node_types['right'] += row[split_ind:]

    # get local grid perimeter links
    local_boundary_links = []
    local_boundary_links_tail_head_nodes_global_ind = []
    local_boundary_links_tail_head_nodes = []
    for _, node_list in local_boundary_node_types.items():
        for node in node_list:
            links = mg.links_at_node[node]
            for link in links:
                if link != -1:
                    tail_head_nodes = mg.nodes_at_link[link]
                    mask = np.isin(tail_head_nodes, node_list)
                    if np.all(mask) and (link not in local_boundary_links):
                        local_boundary_links.append(link)
                        local_boundary_links_tail_head_nodes_global_ind.append(tail_head_nodes)
                        tail_head_nodes_local_ind = [vmg_global_ind.index(node) for node in tail_head_nodes]
                        local_boundary_links_tail_head_nodes.append(tail_head_nodes_local_ind)

    # create voronoi graph with perimeter links
    vor_graph = DualVoronoiGraph((y, x), sort=True,
                                 perimeter_links=local_boundary_links_tail_head_nodes
    )
    for option in ['node', 'link', 'cell']:
        fig, ax = plt.subplots(figsize=(16, 16))
        plot_graph(vor_graph, at=option, axes=ax,
                   with_id=True,
                   #fontsize=7
                   )
        ax.set_title(f'vor_graph_{option}_{rank}')
        fig.savefig(os.path.join(output_dir, f'vor_graph_{option}_{rank}.png'))
        plt.close(fig)

    # create voronoi grid with perimeter links
    local_vmg_adj = VoronoiDelaunayGrid(x.tolist(), y.tolist(),
                                        perimeter_links=local_boundary_links_tail_head_nodes)
    local_vmg_adj.status_at_node[
        local_boundary_nodes_ind] = local_vmg_adj.BC_NODE_IS_FIXED_VALUE

    # make graph plot for vor grid
    for option in ['node', 'link', 'cell']:
        fig, ax = plt.subplots(figsize=(16, 16))
        plot_graph(local_vmg_adj, at=option, axes=ax,
                   with_id=True,
                   #fontsize=7
                   )
        fig.savefig(os.path.join(output_dir, f'vor_grid_adj_{option}_{rank}.png'))
        plt.close(fig)

    # save the grid info as json files
    if create_data_file:
        data = {
            'x': x.tolist(),
            'y': y.tolist(),
            'perimeter_links': local_boundary_links_tail_head_nodes
        }

        with open(os.path.join(output_dir, f"data_input_{rank}.json"), "w") as f:
            json.dump(data, f, indent=2)


    print(f"end rank {rank}")

print("Done!")