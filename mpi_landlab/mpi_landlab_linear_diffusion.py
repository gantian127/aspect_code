"""
This is an experiment to use pymetis and mpi4py for landlab parallel workflow

Model to run on
simple 2D scarp diffusion model
https://github.com/landlab/landlab/blob/master/docs/source/tutorials/fault_scarp/
landlab-fault-scarp.ipynb

Logic:
if rank =0,
    step1: define HexModelGrid and add data fields: topographic__elevation,
            total_deposit__depth
    step2: grid partition using pymetis.
    step3: send each partition information used for the local grid
step4: define local grid as Voronoi ModelGrid (this could be in a finer resolution
        if needed)
step5: define SimpleSubmarineDiffuser model with local grid
for each_step in time_steps:
    run SimpleSubmarineDiffuser model at one step
    step6: send and receive elevation data for ghost nodes, update ghost nodes values
           in local grid
    step7: graph output for each rank (e.g. solution_time_step_rank.vtu)

Notes:
    - this is based on original mpi_landlab_perimeter.py
    - only include ghost nodes


To run the program:
mpiexec -np 5 python mpi_landlab.py

"""

import os
import numpy as np
from collections import defaultdict
import warnings

import matplotlib.pyplot as plt
from mpi4py import MPI
import pymetis

from landlab import HexModelGrid, VoronoiDelaunayGrid
from landlab.plot.graph import plot_graph
from landlab_parallel.io import vtu_dump, pvtu_dump

from plot_utils import create_pvd
from grid_utils import get_perimeter_nodes_and_links

warnings.simplefilter("always")

## step 0: set up parallel
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# make sure number of partitions matches the MPI processes
num_partitions = size


if rank == 0:
    print(f"number of partitions: {num_partitions}")

    ## step 1: define hex model grid and assign z values
    # grid info
    grid_shape = [25, 40]
    spacing = 10
    print(f"global grid shape: {grid_shape}, spacing: {spacing}m")

    # model parameter info
    time_steps = list(range(0, 100))
    D = 0.01  # m2/yr
    dt = 0.2 * spacing * spacing / D  # courant condition for stability of diffusion

    model_parameters = {
        "time_steps": time_steps,
        "D": D,
        "dt": dt,
    }
    print(f"total time_steps: {len(time_steps)}")
    print(f"D:{D} m2/yr, dt: {dt} yr")

    # create output dir for global grid
    output_dir = os.path.join(
        os.getcwd(), f"linear_diffusion_output_png_{num_partitions}"
    )
    os.makedirs(output_dir, exist_ok=True)

    output_pvtu = os.path.join(
        os.getcwd(), f"linear_diffusion_output_pvtu_{num_partitions}"
    )
    os.makedirs(output_pvtu, exist_ok=True)

    # define global grid
    mg = HexModelGrid(grid_shape, spacing=spacing, node_layout="rect")
    z = mg.add_zeros("topographic__elevation", at="node")

    fault_trace_y = 50.0 + 0.25 * mg.x_of_node
    z[mg.y_of_node > fault_trace_y] += (
        10.0 + 0.01 * mg.x_of_node[mg.y_of_node > fault_trace_y]
    )
    print(f"initial elevation sum: {sum(z)} ")

    qs = mg.add_zeros("sediment_flux", at="link")

    # identify boundary nodes
    boundary_nodes = mg.boundary_nodes  # status as fixed value 1

    # plot z 2D and 1D
    mg.imshow(z)
    plt.title("Elevation on Global Grid")
    plt.savefig(os.path.join(output_dir, "dem_hex.png"))
    plt.close()

    ## step2: grid partition
    adjacency_list = []

    # create adjacency list for corners
    for node_id in mg.nodes.flat:
        adjacent_nodes = [n for n in mg.adjacent_nodes_at_node[node_id] if n != -1]
        adjacency_list.append(np.array(adjacent_nodes))

    # Partition the grid using pymetis
    n_cuts, part_labels = pymetis.part_graph(num_partitions, adjacency=adjacency_list)

    # Convert partition labels to a NumPy array
    partition_array = np.array(part_labels)

    # visualization of partition results
    fig, ax = plt.subplots(figsize=[20, 16])
    ax.scatter(mg.node_x, mg.node_y, c=partition_array, cmap="viridis")
    ax.set_title("grid partition based on nodes")
    for node_id in mg.nodes.flat:
        ax.annotate(
            f"{node_id}/{partition_array[node_id]}",
            (mg.node_x[node_id], mg.node_y[node_id]),
            color="black",
            fontsize=8,
            ha="center",
            va="top",
        )
    for node_id in mg.nodes.flat:
        ax.annotate(
            f"{mg.at_node['topographic__elevation'][node_id]}",
            (mg.node_x[node_id], mg.node_y[node_id]),
            color="red",
            fontsize=8,
            ha="center",
            va="bottom",
        )
    fig.savefig(os.path.join(output_dir, "global_grid_partition.png"))
    plt.close(fig)

    print(f"grid partition finished at rank {rank}")

    ## step3 send partition grid info to each process
    for rank in range(size - 1, -1, -1):
        print(f"Create subgrid info for rank {rank}")

        # get local nodes
        local_nodes = [node for node, part in enumerate(part_labels) if part == rank]

        # get send_to and recv_from info
        send_to = defaultdict(set)
        recv_from = defaultdict(set)

        for node in local_nodes:
            for neighbor in adjacency_list[node]:
                neighbor_part = part_labels[neighbor]
                if neighbor_part != rank:
                    send_to[neighbor_part].add(node)
                    recv_from[neighbor_part].add(neighbor)

        # get ghost nodes
        ghost_nodes = [
            int(node) for pid, node_list in recv_from.items() for node in node_list
        ]

        # identify node index for different types
        vmg_global_ind = sorted(local_nodes + ghost_nodes)
        global2local = {g: i for i, g in enumerate(vmg_global_ind)}
        local_nodes_ind = [global2local[val] for val in sorted(local_nodes)]
        local_ghost_nodes_ind = [global2local[val] for val in sorted(ghost_nodes)]
        local_boundary_nodes = [val for val in local_nodes if val in boundary_nodes]

        # get x, y and elevation data
        x = mg.node_x[vmg_global_ind]
        y = mg.node_y[vmg_global_ind]
        elev = mg.at_node["topographic__elevation"][vmg_global_ind]

        # get local perimeter nodes and links
        points = np.column_stack((x, y))
        perimeter_nodes_ind, perimeter_links_ind = get_perimeter_nodes_and_links(points)

        # check if is a valid subgrid
        if len(vmg_global_ind) == len(perimeter_nodes_ind):
            raise ValueError(
                f"Rank {rank}: subgrid includes no core nodes. "
                "Try fewer number of partitions or a larger global grid."
            )

        if rank != 0:
            comm.send(
                (
                    vmg_global_ind,
                    x,
                    y,
                    elev,
                    perimeter_links_ind,
                    perimeter_nodes_ind,
                    local_nodes_ind,
                    local_ghost_nodes_ind,
                    global2local,
                ),
                dest=rank,
                tag=0,
            )
            comm.send((send_to, recv_from), dest=rank, tag=1)

else:
    (
        vmg_global_ind,
        x,
        y,
        elev,
        perimeter_links_ind,
        perimeter_nodes_ind,
        local_nodes_ind,
        local_ghost_nodes_ind,
        global2local,
    ) = comm.recv(source=0, tag=0)

    send_to, recv_from = comm.recv(source=0, tag=1)

    output_dir = None
    output_pvtu = None
    model_parameters = None

output_dir = comm.bcast(output_dir, root=0)
output_pvtu = comm.bcast(output_pvtu, root=0)
model_parameters = comm.bcast(model_parameters, root=0)


## step4: define local model grid
print(f"define local grid for rank {rank}")

with warnings.catch_warnings(record=True) as w:
    local_vmg = VoronoiDelaunayGrid(
        x.tolist(), y.tolist(), perimeter_links=perimeter_links_ind
    )  # x, y needs to be list type

    if w:
        print(f"Warning in Rank {rank}")
        print("Warning occurred:", w[0].message)
        print(perimeter_links_ind)


local_z = local_vmg.add_field("topographic__elevation", elev, at="node")
local_qs = local_vmg.add_zeros("sediment_flux", at="link")
local_vmg.status_at_node[perimeter_nodes_ind] = local_vmg.BC_NODE_IS_FIXED_VALUE

# plot subgrid for each rank
fig, ax = plt.subplots(figsize=[18, 14])
sc = ax.scatter(
    local_vmg.node_x,
    local_vmg.node_y,
    c=local_vmg.at_node["topographic__elevation"],
    vmin=-3,
)
ax.set_title(f"subgrid nodes rank={rank}")

for node_id in local_vmg.boundary_nodes:
    ax.annotate(
        "B",
        (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
        color="blue",
        fontsize=10,
        ha="left",
        va="bottom",
    )
for node_id in local_ghost_nodes_ind:
    ax.annotate(
        "G",
        (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
        color="red",
        fontsize=10,
        ha="right",
        va="bottom",
    )
for node_id in range(0, local_vmg.number_of_nodes):
    ax.annotate(
        f"{vmg_global_ind[node_id]}/{rank}",
        (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
        color="black",
        fontsize=8,
        ha="center",
        va="top",
    )
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Elevation (m)")
fig.savefig(os.path.join(output_dir, f"subgrid_for_rank{rank}.png"))
plt.close(fig)

# plot subgrid of link, cell, node
for option in ["link", "node", "cell"]:
    fig, ax = plt.subplots(figsize=(16, 16))
    plot_graph(local_vmg, at=option, axes=ax)
    ax.set_title(f"{option} graph for rank={rank}")
    fig.savefig(os.path.join(output_dir, f"{option}_subgrid_{rank}.png"))
    plt.close(fig)


## step 5: run simulation
if rank == 0:
    print("start model setup")

time_steps = model_parameters["time_steps"]
D = model_parameters["D"]
dt = model_parameters["dt"]

for time_step in time_steps:
    # run one step
    g = local_vmg.calc_grad_at_link(local_z)
    local_qs[local_vmg.active_links] = -D * g[local_vmg.active_links]
    dzdt = -local_vmg.calc_flux_div_at_node(local_qs)
    local_z[local_vmg.core_nodes] += dzdt[local_vmg.core_nodes] * dt

    # step 6: send and receive data for ghost nodes (new methods)
    # make sure all ranks have finished the model run before communication
    comm.Barrier()

    # set non-blocking receive
    recv_reqs = {}
    for pid in recv_from.keys():
        recv_reqs[pid] = comm.irecv(source=pid, tag=pid)

    # set non-blocking send
    send_reqs = []
    for pid, nodes_to_send in send_to.items():
        nodes_to_send = sorted(nodes_to_send)
        nodes_to_send_local_id = [global2local[val] for val in nodes_to_send]
        elev_to_send = local_vmg.at_node["topographic__elevation"][
            nodes_to_send_local_id
        ].copy()
        send_reqs.append(comm.isend((nodes_to_send, elev_to_send), dest=pid, tag=rank))

    # wait for all recv to finish and update ghost nodes values (non-blocking receive)
    for pid, req in recv_reqs.items():
        ghost_nodes, elev_values = (
            req.wait()
        )  # wait for finishing and then get the data
        ghost_nodes_local_id = np.array(
            [global2local[g] for g in ghost_nodes], dtype=int
        )
        local_vmg.at_node["topographic__elevation"][ghost_nodes_local_id] = elev_values

    # make sure all send finished before next step
    for req in send_reqs:
        req.wait()

    # step7: make vtu file for each rank at each time step
    with open(os.path.join(output_pvtu, f"rank{rank}_{time_step}.vtu"), "w") as fp:
        fp.write(vtu_dump(local_vmg))

    # # testing code!! make plots for each rank at each time as png file for debugging
    # fig, ax = plt.subplots(figsize=[18, 14])
    # sc = ax.scatter(local_vmg.node_x, local_vmg.node_y,
    #                 c=local_vmg.at_node["topographic__elevation"], cmap="coolwarm",
    #                 vmin=-3)
    # ax.set_title(f'subgrid nodes rank={rank}')
    # for node_id in local_vmg.boundary_nodes:
    #     ax.annotate(f"B",
    #                 (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
    #                 color='blue', fontsize=12, ha='center', va='top')
    # for node_id in range(0, local_vmg.number_of_nodes):
    #     ax.annotate(f'{node_id}/{local_vmg.at_node["topographic__elevation"][node_id]}',
    #                 (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
    #                 color='black',fontsize=12, ha='center', va='bottom')
    # cbar = fig.colorbar(sc, ax=ax)
    # cbar.set_label('Elevation (m)')
    # fig.savefig(os.path.join(output_dir,
    #             f'subgrid_for_rank{rank}_loop_{time_step}.png'))
    # plt.close(fig)


# check sum values
local_sum = local_vmg.at_node["topographic__elevation"][local_nodes_ind].sum()
print(f"final rank {rank}: {local_sum}")
global_sum = comm.allreduce(local_sum, op=MPI.SUM)

# pass local nodes values
local_updates = []
for node in local_nodes_ind:
    local_updates.append(
        (vmg_global_ind[node], local_vmg.at_node["topographic__elevation"][node])
    )

all_updates = comm.gather(local_updates, root=0)

if rank == 0:
    # check global sum
    print(f"final global sum: {global_sum}")

    # save final global elevation results
    flat_updates = [item for sublist in all_updates for item in sublist]
    for node_id, elev in flat_updates:
        mg.at_node["topographic__elevation"][node_id] = elev

    np.save(
        os.path.join(output_dir, f"elevation_result_{num_partitions}.npy"),
        mg.at_node["topographic__elevation"],
    )

    # create pvtu files for each time step
    pvtu_files = []
    for time_step in time_steps:
        # write pvtu file for each time step
        with open(os.path.join(output_pvtu, f"global_{time_step}.pvtu"), "w") as fp:
            fp.write(
                pvtu_dump(
                    local_vmg,
                    [
                        os.path.join(output_pvtu, f"rank{i}_{time_step}.vtu")
                        for i in range(0, num_partitions)
                    ],
                )
            )

        pvtu_files.append(f"global_{time_step}.pvtu")

    # create pvd files for all time steps
    pvd_file_path = os.path.join(output_pvtu, "simulation.pvd")
    create_pvd(pvtu_files, time_steps, pvd_file_path)

    print("Simulation is done!")
