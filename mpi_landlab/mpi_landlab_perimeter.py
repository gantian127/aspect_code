"""
This is an experiment to use pymetis and mpi4py for landlab parallel

Model to run on
SimpleSubmarineDiffuser
https://landlab.csdms.io/tutorials/marine_sediment_transport/simple_submarine_diffuser_tutorial.html

Logic:
if rank =0,
    step1: define HexModelGrid and add data fields: topographic__elevation, total_deposit__depth
    step2: grid partition using pymetis.
    step3: send each partition information used for the local grid
step4: define local grid as Voronoi ModelGrid (this could be in a finer resolution if needed)
step5: define SimpleSubmarineDiffuser model with local grid
for each_step in time_steps:
    run SimpleSubmarineDiffuser model at one step
    step6: send and receive elevation data for ghost nodes, update ghost nodes values in local grid
    step7: graph output for each rank (e.g. solution_time_step_rank.vtu)

Notes:
    - this is based on original mpi_landlab.py
    - added artificial node code
    - added perimeter links code

To run the program:
mpiexec -np 5 python mpi_landlab.py

"""

import os
import shutil
import numpy as np
import pymetis

import matplotlib.pyplot as plt

from landlab import HexModelGrid, VoronoiDelaunayGrid
from landlab.components import SimpleSubmarineDiffuser
from landlab.plot.graph import plot_graph
from plot_utils import create_pvd
from landlab_parallel.io import vtu_dump, pvtu_dump


## step 0: set up parallel
from mpi4py import MPI
from collections import defaultdict

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# make sure number of partitions matches the MPI processes
num_partitions = size

if rank == 0:
    print(f"number of partitions: {num_partitions}")

    ## step 1: define hex model grid and assign z values

    # create output dir for global grid
    output_dir = os.path.join(os.getcwd(),f'perimeter_links_output_png_{num_partitions}')
    os.makedirs(output_dir, exist_ok=True)

    output_pvtu = os.path.join(os.getcwd(),f'perimeter_links_output_pvtu_{num_partitions}')
    os.makedirs(output_pvtu, exist_ok=True)

    # define global grid
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

    # identify boundary nodes
    boundary_nodes = mg.boundary_nodes  # status as fixed value 1

    # testing code!! global grid boundary node status as 1 (fixed value)
    # print(f"global grid boundary node status: {mg.status_at_node[boundary_nodes]}")

    # plot z 2D and 1D
    mg.imshow(z, cmap="coolwarm", vmin=-3)
    plt.title("Elevation on Global Grid")
    plt.savefig(os.path.join(output_dir, "dem_hex.png"))
    plt.close()

    plt.plot(mg.x_of_node, z, ".")
    plt.plot([0, 17], [0, 0], "b:")
    plt.grid(True)
    plt.xlabel("Distance (m)")
    plt.ylabel("Elevation (m)")
    plt.savefig(os.path.join(output_dir,"dem_hex_slice.png"))
    plt.close()

    # plot node, cell, link
    for option in ['link', 'node', 'cell']:
        fig, ax = plt.subplots(figsize=(16, 16))
        plot_graph(mg, at=option, axes=ax)
        ax.set_title(f'{option} graph for global grid')
        fig.savefig(os.path.join(output_dir, f'{option}_global_grid.png'))
        plt.close(fig)

    # plot total_deposit__thickness
    mg.imshow("total_deposit__thickness", cmap="coolwarm", vmin=-1,vmax=1)
    plt.title("Total deposit thickness initiation (m)")
    plt.savefig(os.path.join(output_dir,"total_deposit_init.png"))
    plt.close()


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

    # visualization
    fig, ax = plt.subplots(figsize=[16, 14])
    ax.scatter(mg.node_x, mg.node_y, c=partition_array, cmap='viridis')
    ax.set_title('grid partition based on nodes')
    for node_id in mg.nodes.flat:
        ax.annotate(f"{node_id}/par{partition_array[node_id]}",
                    (mg.node_x[node_id], mg.node_y[node_id]),
                    color='black', fontsize=8, ha='center', va='top')
    fig.savefig(os.path.join(output_dir,'global_grid_partition.png'))
    plt.close(fig)

    print(f"grid partition finished at rank {rank}")

    ## step3 send partition grid info to each process
    for rank in range(size-1, -1, -1):
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
        ghost_nodes = [int(node) for pid, node_list in recv_from.items() for node in node_list]
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
        local_artificial_nodes_ind = [vmg_global_ind.index(val) for val in sorted(artificial_nodes)]

        local_boundary_nodes = [node for node in local_nodes + ghost_nodes if node in boundary_nodes]
        local_boundary_nodes_ind = [vmg_global_ind.index(val) for val in
                                    sorted(artificial_nodes + local_boundary_nodes)]

        # get x, y and elevation data
        x = mg.node_x[vmg_global_ind]
        y = mg.node_y[vmg_global_ind]
        elev = mg.at_node["topographic__elevation"][vmg_global_ind]

        # get boundary node types for local grid
        subgrid_boundary_nodes = sorted(artificial_nodes + local_boundary_nodes)
        global_boundary_node_types = {
            "left": mg.nodes_at_left_edge,
            "right": mg.nodes_at_right_edge,
            "bottom": mg.nodes_at_bottom_edge,
            "top": mg.nodes_at_top_edge,
        }

        local_node_structure = [
            sorted([node for node in subgrid_boundary_nodes if left <= node <= right])
            for left, right in
            zip(global_boundary_node_types['left'], global_boundary_node_types['right'])
        ]

        local_node_structure = [row for row in local_node_structure if row]  # remove empty list

        local_boundary_node_types = {
            'bottom': local_node_structure[0],
            'top': local_node_structure[-1],
            'left': [local_node_structure[0][0], local_node_structure[-1][0]],
            'right': [local_node_structure[0][-1], local_node_structure[-1][-1]],
        }

        for row in local_node_structure[1:-1]:
            if len(row) <= 2:
                local_boundary_node_types['left'].append(row[0])
                local_boundary_node_types['right'].append(row[-1])
            else:
                for i in range(0, len(row)):
                    if row[i + 1] - row[i] != 1:
                        split_ind = i + 1
                        break
                local_boundary_node_types['left'] += row[:split_ind]
                local_boundary_node_types['right'] += row[split_ind:]

        # get local grid perimeter links
        perimeter_links_ind = []
        perimeter_links_tailhead_global_ind = []
        perimeter_links_tailhead_local_ind = []
        for _, node_list in local_boundary_node_types.items():
            for node in node_list:
                links = mg.links_at_node[node]
                for link in links:
                    if link != -1:
                        tail_head_nodes = mg.nodes_at_link[link]
                        mask = np.isin(tail_head_nodes, node_list)
                        if np.all(mask) and (link not in perimeter_links_ind):
                            perimeter_links_ind.append(link)
                            perimeter_links_tailhead_global_ind.append(
                                tail_head_nodes)
                            tail_head_nodes_local_ind = [vmg_global_ind.index(node) for
                                                         node in tail_head_nodes]
                            perimeter_links_tailhead_local_ind.append(
                                tail_head_nodes_local_ind)

        # !! Testing code to get each partition x, y boundary info
        # print(f"rank:{rank}")
        # print(f"x:{x}")
        # print(f"y:{y}")
        # print(f"lenx: {len(x)} leny: {len(y)}")
        # print(f"local_boundary:{local_boundary_nodes_ind}")
        # print(f"perimeter_link_global: {perimeter_links_tailhead_global_ind}")
        # print(f"perimeter_link_local: {perimeter_links_tailhead_local_ind}")


        if rank != 0:
            comm.send(
                (vmg_global_ind, x, y, elev,
                 perimeter_links_tailhead_local_ind,
                 local_boundary_nodes_ind,
                 local_nodes_ind,
                 local_ghost_nodes_ind,
                 local_artificial_nodes_ind),
                dest=rank,
                tag=0
            )
            comm.send((send_to, recv_from), dest=rank, tag=1)

else:
    (vmg_global_ind, x, y, elev,
     perimeter_links_tailhead_local_ind,
     local_boundary_nodes_ind,
     local_nodes_ind,
     local_ghost_nodes_ind,
     local_artificial_nodes_ind) = comm.recv(source=0, tag=0)

    send_to, recv_from = comm.recv(source=0, tag=1)

    output_dir = None
    output_pvtu = None

output_dir = comm.bcast(output_dir, root=0)
output_pvtu = comm.bcast(output_pvtu, root=0)


## step4: define local model grid
if rank ==0:
    print('define local grid')
local_vmg = VoronoiDelaunayGrid(x.tolist(), y.tolist(),
                                perimeter_links=perimeter_links_tailhead_local_ind)  # x, y needs to be list type

# !! testing code for boundary node status
# print(f"rank {rank} default boundary: {local_vmg.boundary_nodes}")
# print(f"rank {rank} default boundary status: {local_vmg.status_at_node[local_vmg.boundary_nodes]}")

# !! testing code for zero division error
# print(f'rank: {rank}')
# print(x.tolist())
# print(y.tolist())
# print(f'end: rank{rank}')

local_vmg.add_field("topographic__elevation", elev, at="node")
cum_depo = local_vmg.add_zeros("total_deposit__thickness", at="node")
local_vmg.status_at_node[local_boundary_nodes_ind] = local_vmg.BC_NODE_IS_FIXED_VALUE

# !! testing code for boundary node status
# print(f"rank {rank} supposed boundary: {local_boundary_nodes_ind}")
# print(f"rank {rank} supposed boundary status: {local_vmg.status_at_node[local_boundary_nodes_ind]}")

# !! test code findings:
# the default local grid boundary nodes are much less than the supposed boundary nodes.
# default boundary nodes status is 1 (fixed values)


# plot subgrid for each rank
fig, ax = plt.subplots(figsize=[18, 14])
sc = ax.scatter(local_vmg.node_x, local_vmg.node_y,
           c=local_vmg.at_node["topographic__elevation"], cmap="coolwarm", vmin=-3)
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
    ax.annotate(f"A",
                (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
                color='green', fontsize=10, ha='right', va='bottom')
for node_id in range(0,local_vmg.number_of_nodes):
    ax.annotate(f"{vmg_global_ind[node_id]}/{rank}",
                (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
                color='black', fontsize=8, ha='center', va='top')
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Elevation (m)')
fig.savefig(os.path.join(output_dir,f'subgrid_for_rank{rank}.png'))
plt.close(fig)

# plot subgrid of link, cell, node
for option in ['link','node','cell']:
    fig, ax = plt.subplots(figsize=(16, 16))
    plot_graph(local_vmg, at=option, axes=ax)
    ax.set_title(f'{option} graph for rank={rank}')
    fig.savefig(os.path.join(output_dir, f'{option}_subgrid_{rank}.png'))
    plt.close(fig)

# # !! testing compare local model grid without perimeter links
# local_vmg_default = VoronoiDelaunayGrid(x.tolist(), y.tolist())
# local_vmg_default.status_at_node[
#     local_boundary_nodes_ind] = local_vmg_default.BC_NODE_IS_FIXED_VALUE
# for option in ['node', 'link', 'cell']:
#     fig, ax = plt.subplots(figsize=(16, 16))
#     plot_graph(local_vmg_default, at=option, axes=ax)
#     fig.savefig(os.path.join(output_dir, f'vor_grid_default_{option}_{rank}.png'))
#     plt.close(fig)


## step 5: run simulation
if rank ==0:
    print('start model setup')

# define model
ssd = SimpleSubmarineDiffuser(
    local_vmg, sea_level=0.0, wave_base=1.0, shallow_water_diffusivity=1.0
)

time_steps = list(range(0,50))

# # assign artificial nodes value as no data
for field_name in local_vmg.at_node:
    local_vmg.at_node[field_name][local_artificial_nodes_ind] = np.nan
    # print(field_name)
    # print(local_vmg.at_node[field_name][local_artificial_nodes_ind])

# define visual grid for local nodes  #TODO: this also needs to identify perimeter links
vis_x = x[local_nodes_ind].tolist()
vis_y = y[local_nodes_ind].tolist()
vis_vmg = VoronoiDelaunayGrid(vis_x, vis_y)  # x, y needs to be list type

for field_name in local_vmg.at_node:
    data = local_vmg.at_node[field_name][local_nodes_ind]
    vis_vmg.add_field(field_name, data, at="node")

# plot visual grid
fig, ax = plt.subplots(figsize=[18, 14])
sc = ax.scatter(vis_vmg.node_x, vis_vmg.node_y,
           c=vis_vmg.at_node["topographic__elevation"], cmap="coolwarm", vmin=-3)
ax.set_title(f'subgrid local nodes only rank={rank}')
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Elevation (m)')
fig.savefig(os.path.join(output_dir,f'vis_subgrid_for_rank{rank}.png'))
plt.close(fig)

# loop for multiple time steps
if rank ==0:
    print('start model loops')

for time_step in time_steps:

    # run one step
    ssd.run_one_step(0.2)

    # assign value to local nodes grid for visualization
    for field_name in local_vmg.at_node:
        vis_elev = local_vmg.at_node[field_name][local_nodes_ind]
        vis_vmg.at_node[field_name] = vis_elev[:]

    ## step6: send and receive data for ghost nodes
    for pid, nodes_to_send in send_to.items():
        # Convert to sorted list
        nodes_to_send_local_id = [vmg_global_ind.index(val) for val in nodes_to_send]
        elev_to_send = local_vmg.at_node["topographic__elevation"][nodes_to_send_local_id]
        comm.send((nodes_to_send, elev_to_send), dest=pid, tag=rank)
        #print(f"Rank {rank} sent data to {pid} for nodes: {nodes_to_send}")

    local_vmg_ghost_nodes= []
    for pid in recv_from.keys():
        ghost_nodes, elev_values = comm.recv(source=pid, tag=pid)
        ghost_nodes_local_id = [vmg_global_ind.index(val) for val in ghost_nodes]
        local_vmg.at_node["topographic__elevation"][ghost_nodes_local_id] = elev_values
        local_vmg_ghost_nodes.extend(ghost_nodes)

    ## step7: make plots for each rank at each time as png file
    # fig, ax = plt.subplots(figsize=[18, 14])
    # sc = ax.scatter(local_vmg.node_x, local_vmg.node_y,
    #                 c=local_vmg.at_node["topographic__elevation"], cmap="coolwarm",
    #                 vmin=-3)
    # ax.set_title(f'subgrid nodes rank={rank}')
    # for node_id in local_boundary_nodes_ind:
    #     ax.annotate(f"B",
    #                 (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
    #                 color='blue', fontsize=12, ha='center', va='top')
    # cbar = fig.colorbar(sc, ax=ax)
    # cbar.set_label('Elevation (m)')
    # fig.savefig(os.path.join(output_sub_dir, f'subgrid_for_rank{rank}_loop_{time_step}.png'))
    # plt.close(fig)

    ## step7: make vtu file for each rank at each time step
    with open(os.path.join(output_pvtu, f"rank{rank}_{time_step}.vtu"), "w") as fp:
        fp.write(vtu_dump(vis_vmg))

# Testing!! check the artificial node values
# for field_name in local_vmg.at_node:
#     print(field_name)
#     print(local_vmg.at_node[field_name][local_artificial_nodes_ind])

# check sum values
local_sum = local_vmg.at_node["topographic__elevation"][local_nodes_ind].sum()
print(f"{rank}: {local_sum}")
global_sum = comm.allreduce(local_sum, op=MPI.SUM)

# pass local nodes values
local_updates = []
for node in local_nodes_ind:
    local_updates.append((vmg_global_ind[node],
                          local_vmg.at_node["topographic__elevation"][node]))

all_updates = comm.gather(local_updates, root=0)

if rank==0:
    # # Flatten list of updates from all ranks
    # flat_updates = [item for sublist in all_updates for item in sublist]
    # for node_id, elev, cum_depo in flat_updates:
    #     mg.at_node["topographic__elevation"][node_id] = elev
    #     mg.at_node["total_deposit__thickness"][node_id] = cum_depo

    # check global sum
    print(global_sum)

    # save final global elevation results
    flat_updates = [item for sublist in all_updates for item in sublist]
    for node_id, elev in flat_updates:
        mg.at_node["topographic__elevation"][node_id] = elev

    np.save(os.path.join(output_dir, f"elevation_result_{num_partitions}.npy"),
                         mg.at_node["topographic__elevation"])


    # create pvtu files for each time step
    pvtu_files = []
    for time_step in time_steps:
        # write pvtu file for each time step
        with open(os.path.join(output_pvtu, f"global_{time_step}.pvtu"), "w") as fp:
            fp.write(pvtu_dump(local_vmg,
                               [os.path.join(output_pvtu, f"rank{i}_{time_step}.vtu")
                               for i in range(0, num_partitions)]
                               )
                     )

        pvtu_files.append(f"global_{time_step}.pvtu")

    # create pvd files for all time steps
    pvd_file_path = os.path.join(output_pvtu, "simulation.pvd")
    create_pvd(pvtu_files, time_steps, pvd_file_path)


    print('Simulation is done!')
