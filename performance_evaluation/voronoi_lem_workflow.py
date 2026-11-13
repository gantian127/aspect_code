"""
Testing LEM workflow for performance evaluation

This is based on mpi_landlab/mpi_landlab_lem_workflow.py
with adjustments to track the time spent on different sections and disable output files.

mpiexec -n 4 python voronoi_lem_workflow.py --grid-shape 300,300
"""

import time

script_started = time.perf_counter()

import argparse
import csv
import os
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pymetis
from grid_utils import Uplift, get_perimeter_nodes_and_links
from landlab import HexModelGrid, VoronoiDelaunayGrid
from landlab.components import FlowAccumulator, LinearDiffuser, StreamPowerEroder
from landlab.plot.graph import plot_graph
from landlab_parallel.io import pvtu_dump, vtu_dump
from mpi4py import MPI
from plot_utils import create_pvd

warnings.simplefilter("always")


TIMING_FIELDS = (
    "mpi_ranks",
    "grid_shape",
    "time_steps",
    "global_grid_setup_s",
    "local_grid_setup_s",
    "model_loop_s",
    "total_workflow_run_s",
)


def format_seconds(value):
    """Format a timing value consistently for CSV output."""
    return f"{value:.9f}"


def write_timing_csv(path, rows):
    """Write one maximum-across-ranks summary row for a workflow run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_grid_shape(value):
    """Parse a positive two-dimensional global grid shape."""
    parts = value.split(",")
    if len(parts) != 2 or any(not part or part != part.strip() for part in parts):
        raise argparse.ArgumentTypeError("grid shape must be ROWS,COLUMNS")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "grid shape dimensions must be integers"
        ) from error
    if any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("grid shape dimensions must be positive")
    return shape


parser = argparse.ArgumentParser()
parser.add_argument(
    "--grid-shape",
    type=parse_grid_shape,
    default=(500, 500),
    metavar="ROWS,COLUMNS",
    help="global grid shape (default: 500,500)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="random seed for initial elevations (default: 1)",
)
parser.add_argument(
    "--timing-output-dir",
    type=Path,
    default=Path(__file__).resolve().parent / "voronoi_timing_results",
    metavar="DIR",
    help=(
        "directory for the per-run timing CSV "
        "(default: performance_evaluation/voronoi_timing_results)"
    ),
)
cli_args = parser.parse_args()

## step 0: set up parallel
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# make sure number of partitions matches the MPI processes
num_partitions = size
debug_plot = False
write_result_files = False

run_id = None
if rank == 0:
    run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_%f")

# Global setup includes global-grid creation, partitioning, construction of the
# per-rank metadata, distribution, and the broadcasts that complete step 3.
comm.Barrier()
global_grid_setup_started = MPI.Wtime()

if rank == 0:
    print(f"number of partitions: {num_partitions}")

    ## step 1: define hex model grid and assign z values
    # grid info
    grid_shape = list(cli_args.grid_shape)
    spacing = 100
    print(f"global grid shape: {grid_shape}, spacing: {spacing} m")

    # model parameter info
    time_steps = list(range(0, 200))  # time step original 2000
    D = 0.01  # m2/yr
    dt = 250.0  # yr
    uplift_rate = 0.004  # m/yr

    model_parameters = {
        "time_steps": time_steps,
        "D": D,
        "uplift_rate": uplift_rate,
        "dt": dt,
    }
    print(f"total time_steps: {len(time_steps)}")
    print(f"D:{D} m2/yr, uplift_rate m/yr: {uplift_rate}, dt:{dt} yr")

    # Diagnostic plots and result files are disabled for pure scaling tests.
    output_dir = None
    if debug_plot or write_result_files:
        output_dir = os.path.join(
            os.getcwd(), f"lem_workflow_output_png_{num_partitions}_{len(time_steps)}"
        )
        os.makedirs(output_dir, exist_ok=True)

    output_pvtu = None
    if write_result_files:
        output_pvtu = os.path.join(
            os.getcwd(), f"lem_workflow_output_pvtu_{num_partitions}_{len(time_steps)}"
        )
        os.makedirs(output_pvtu, exist_ok=True)

    # define global grid
    mg = HexModelGrid(grid_shape, spacing=spacing, node_layout="rect")
    z = mg.add_zeros("topographic__elevation", at="node")

    # Use the same deterministic elevation input as raster_lem_example.py.
    rng = np.random.default_rng(seed=cli_args.seed)
    z[:] = rng.uniform(size=grid_shape).flatten()

    print(f"initial elevation sum: {sum(z)} ")

    # identify boundary nodes
    boundary_nodes = mg.boundary_nodes  # status as fixed value 1

    # plot z 2D and 1D
    if debug_plot:
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

    if debug_plot:
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
        local_boundary_nodes_ind = [
            global2local[val] for val in local_nodes if val in boundary_nodes
        ]

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
                    local_boundary_nodes_ind,
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
        local_boundary_nodes_ind,
    ) = comm.recv(source=0, tag=0)

    send_to, recv_from = comm.recv(source=0, tag=1)

    output_dir = None
    output_pvtu = None
    model_parameters = None

output_dir = comm.bcast(output_dir, root=0)
output_pvtu = comm.bcast(output_pvtu, root=0)
model_parameters = comm.bcast(model_parameters, root=0)

comm.Barrier()
global_grid_setup_time = MPI.Wtime() - global_grid_setup_started

## step4: define local model grid
local_grid_setup_started = MPI.Wtime()
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
# local_qs = local_vmg.add_zeros("sediment_flux", at="link")
local_uplift_rate = local_vmg.add_full(
    "uplift_rate", model_parameters["uplift_rate"], at="node"
)
local_vmg.status_at_node[perimeter_nodes_ind] = local_vmg.BC_NODE_IS_FIXED_VALUE

comm.Barrier()
local_grid_setup_time = MPI.Wtime() - local_grid_setup_started

if debug_plot:
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
            f"{node_id}/{local_vmg.at_node['topographic__elevation'][node_id]}",
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

# model parameters
time_steps = model_parameters["time_steps"]
D = model_parameters["D"]
dt = model_parameters["dt"]

# initiate model components
uplift = Uplift(local_vmg)
ld = LinearDiffuser(local_vmg, linear_diffusivity=D)
fa = FlowAccumulator(local_vmg)
sp = StreamPowerEroder(local_vmg, K_sp=0.0001)

# organize components and parameters
components = [
    (uplift, (dt,)),
    (ld, (dt,)),
    (fa, ()),
    (sp, (dt,)),
]

# Synchronize once at the measurement boundary. This complete loop wall time,
# including component computation and required halo exchange, is the primary
# strong/weak-scaling metric.
comm.Barrier()
model_loop_started = MPI.Wtime()

for time_step in time_steps:
    # run onestep

    # # !! testing code: linear diffusion with math equations
    # g = local_vmg.calc_grad_at_link(local_z)
    # local_qs[local_vmg.active_links] = -D * g[local_vmg.active_links]
    # dzdt = -local_vmg.calc_flux_div_at_node(local_qs)
    # local_z[local_vmg.core_nodes] += dzdt[local_vmg.core_nodes] * dt

    # loop to run each component
    for component, component_args in components:
        component.run_one_step(*component_args)

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
            drainage_area_to_send = local_vmg.at_node["drainage_area"][
                nodes_to_send_local_id
            ].copy()
            send_reqs.append(
                comm.isend(
                    (nodes_to_send, elev_to_send, drainage_area_to_send),
                    dest=pid,
                    tag=rank,
                )
            )

        # wait for all recv to finish and update ghost nodes values (non-blocking receive)
        for pid, req in recv_reqs.items():
            ghost_nodes, elev_values, drainage_area_values = (
                req.wait()
            )  # wait for finishing and then get the data
            ghost_nodes_local_id = np.array(
                [global2local[g] for g in ghost_nodes], dtype=int
            )
            local_vmg.at_node["topographic__elevation"][ghost_nodes_local_id] = (
                elev_values
            )
            local_vmg.at_node["drainage_area"][ghost_nodes_local_id] = (
                drainage_area_values
            )

            # make sure all send finished before next step
            for req in send_reqs:
                req.wait()

    # step7: make a VTU file for each rank at each time step
    if write_result_files:
        with open(
            os.path.join(output_pvtu, f"rank{rank}_{time_step}.vtu"), "w"
        ) as fp:
            fp.write(vtu_dump(local_vmg))

    # testing code!! make plots for each rank at each time as png file for debugging
    # if time_step == time_steps[-1]:
    #     fig, ax = plt.subplots(figsize=[18, 14])
    #     sc = ax.scatter(
    #         local_vmg.node_x,
    #         local_vmg.node_y,
    #         c=local_vmg.at_node["topographic__elevation"],
    #         cmap="coolwarm",
    #         vmin=-3,
    #     )
    #     ax.set_title(f"subgrid nodes rank={rank}")
    #     for node_id in local_vmg.boundary_nodes:
    #         ax.annotate(
    #             "B",
    #             (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
    #             color="blue",
    #             fontsize=12,
    #             ha="center",
    #             va="top",
    #         )
    #     for node_id in range(0, local_vmg.number_of_nodes):
    #         ax.annotate(
    #             f"{node_id}/{local_vmg.at_node['topographic__elevation'][node_id]}",
    #             (local_vmg.node_x[node_id], local_vmg.node_y[node_id]),
    #             color="black",
    #             fontsize=12,
    #             ha="center",
    #             va="bottom",
    #         )
    #     cbar = fig.colorbar(sc, ax=ax)
    #     cbar.set_label("Elevation (m)")
    #     fig.savefig(
    #         os.path.join(output_dir, f"subgrid_for_rank{rank}_loop_{time_step}.png")
    #     )
    #     plt.close(fig)


comm.Barrier()
model_loop_time = MPI.Wtime() - model_loop_started

# Final checksum, reduction, gather, and global-grid reconstruction are kept
# separate from the model loop because the distributed numerical solution is
# already complete at this point.
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

if rank == 0:
    if write_result_files:
        np.save(
            os.path.join(output_dir, f"elevation_result_{num_partitions}.npy"),
            mg.at_node["topographic__elevation"],
        )

    # create png file for final step elevation
    if debug_plot:
        mg.imshow("topographic__elevation")
        print(
            "final elev min and max: ",
            mg.at_node["topographic__elevation"][mg.core_nodes].min(),
            mg.at_node["topographic__elevation"][mg.core_nodes].max(),
        )
        plt.title("Elevation on Global Grid Final Step ")
        plt.savefig(os.path.join(output_dir, "dem_hex_final.png"))

    if write_result_files:
        # Create PVTU/PVD metadata and include it in the output timing category.
        pvtu_files = []
        for time_step in time_steps:
            # write pvtu file for each time step
            with open(
                os.path.join(output_pvtu, f"global_{time_step}.pvtu"), "w"
            ) as fp:
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

# Measure from the first executable statement in this file through imports,
# setup, the model loop, finalization, and optional result output. Timing-CSV
# creation and Python/MPI process shutdown necessarily occur afterward.
comm.Barrier()
total_workflow_run_time = time.perf_counter() - script_started
max_global_grid_setup_time = comm.reduce(
    global_grid_setup_time, op=MPI.MAX, root=0
)
max_local_grid_setup_time = comm.reduce(local_grid_setup_time, op=MPI.MAX, root=0)
max_model_loop_time = comm.reduce(model_loop_time, op=MPI.MAX, root=0)
max_total_workflow_run_time = comm.reduce(
    total_workflow_run_time, op=MPI.MAX, root=0
)

if rank == 0:
    print("\nWorkflow timing summary (maximum across MPI ranks)")
    print(f"  Global-grid setup:    {max_global_grid_setup_time:.6f} s")
    print(f"  Local-grid setup:     {max_local_grid_setup_time:.6f} s")
    print(f"  Model loop:           {max_model_loop_time:.6f} s")
    print(
        f"  Total workflow run:   {max_total_workflow_run_time:.6f} s",
        flush=True,
    )
    timing_row = {
        "mpi_ranks": num_partitions,
        "grid_shape": f"{cli_args.grid_shape[0]}x{cli_args.grid_shape[1]}",
        "time_steps": len(time_steps),
        "global_grid_setup_s": format_seconds(max_global_grid_setup_time),
        "local_grid_setup_s": format_seconds(max_local_grid_setup_time),
        "model_loop_s": format_seconds(max_model_loop_time),
        "total_workflow_run_s": format_seconds(max_total_workflow_run_time),
    }
    timing_filename = (
        f"voronoi_timing_{run_id}_ranks_{num_partitions}_"
        f"shape_{cli_args.grid_shape[0]}x{cli_args.grid_shape[1]}.csv"
    )
    timing_path = cli_args.timing_output_dir.resolve() / timing_filename
    write_timing_csv(timing_path, [timing_row])
    print(f"  Timing CSV:           {timing_path}", flush=True)
