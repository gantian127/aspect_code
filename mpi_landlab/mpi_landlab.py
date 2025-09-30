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
from plot_utils import vtu_dump, pvtu_dump, create_pvd


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
    output_dir = os.path.join(os.getcwd(),f'output_png_{num_partitions}')
    os.makedirs(output_dir, exist_ok=True)

    output_pvtu = os.path.join(os.getcwd(),f'output_pvtu_{num_partitions}')
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

        ghost_nodes = [int(node) for pid, node_list in recv_from.items() for node in node_list]

        vmg_global_ind = sorted(local_nodes + ghost_nodes)
        x = mg.node_x[vmg_global_ind]
        y = mg.node_y[vmg_global_ind]
        elev = mg.at_node["topographic__elevation"][vmg_global_ind]
        local_boundary_nodes = [node for node in local_nodes if node in boundary_nodes]
        local_boundary_nodes_ind = [vmg_global_ind.index(val) for val in
                                    sorted(ghost_nodes + local_boundary_nodes)]
        local_nodes_ind = [vmg_global_ind.index(val) for val in sorted(local_nodes)]

        # !! Testing code to get each partition x, y boundary info
        # print(f"rank:{rank}")
        # print(f"x:{x}")
        # print(f"y:{y}")
        # print(f"lenx: {len(x)} leny: {len(y)}")
        # print(f"local_boundary:{local_boundary_nodes_ind}")

        if rank != 0:
            comm.send(
                (vmg_global_ind, x, y, elev,local_boundary_nodes_ind, local_nodes_ind),
                dest=rank,
                tag=0
            )
            comm.send((send_to, recv_from), dest=rank, tag=1)



else:
    vmg_global_ind, x, y, elev,local_boundary_nodes_ind, local_nodes_ind = comm.recv(source=0, tag=0)
    send_to, recv_from = comm.recv(source=0, tag=1)
    output_dir = None
    output_pvtu = None

output_dir = comm.bcast(output_dir, root=0)
output_pvtu = comm.bcast(output_pvtu, root=0)


## step4: define local model grid
local_vmg = VoronoiDelaunayGrid(x.tolist(), y.tolist())  # x, y needs to be list type

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
                color='blue', fontsize=12, ha='center', va='top')
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Elevation (m)')
fig.savefig(os.path.join(output_dir,f'subgrid_for_rank{rank}.png'))
plt.close(fig)

## step 5: run simulation
# define model
ssd = SimpleSubmarineDiffuser(
    local_vmg, sea_level=0.0, wave_base=1.0, shallow_water_diffusivity=1.0
)

time_steps = list(range(0,50))

# define visual grid for local nodes
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

# check sum values
local_sum = local_vmg.at_node["topographic__elevation"][local_nodes_ind].sum()
print(f"{rank}: {local_sum}")
global_sum = comm.allreduce(local_sum, op=MPI.SUM)

if rank==0:
    # # Flatten list of updates from all ranks
    # flat_updates = [item for sublist in all_updates for item in sublist]
    # for node_id, elev, cum_depo in flat_updates:
    #     mg.at_node["topographic__elevation"][node_id] = elev
    #     mg.at_node["total_deposit__thickness"][node_id] = cum_depo

    # check global sum
    print(global_sum)

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
