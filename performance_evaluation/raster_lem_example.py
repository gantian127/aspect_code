"""
This is based on Eric's run_lem_example.py in landlab-parallel package.

mpiexec -n 4 python raster_lem_example.py --grid-shape 300,300 --mode d4 --seed 1
mpiexec -n 4 python raster_lem_example.py --grid-shape 300,300 --mode odd-r --seed 1
"""

import time

script_started = time.perf_counter()

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from landlab.components import FlowAccumulator
from landlab.components import LinearDiffuser
from landlab.components import StreamPowerEroder
from landlab.grid.nodestatus import NodeStatus
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from landlab_parallel.grid import create_landlab_grid
from landlab_parallel.io import pvtu_dump
from landlab_parallel.io import vtu_dump
from landlab_parallel.tiler import D4Tiler
from landlab_parallel.tiler import OddRTiler
from landlab_parallel.tiler import Tile

debug_plot = False
write_result_files = False
timing_output_dir = Path(__file__).resolve().parent / "raster_timing_results"

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


def run(shape, mode="odd-r", seed=None):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    RANK = comm.Get_rank()
    n_partitions = comm.Get_size()

    run_id = None
    if RANK == 0:
        run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S_%f")

    # Global setup includes global-array creation, partitioning, scattering,
    # and distribution of each rank's tile data.
    comm.Barrier()
    global_grid_setup_started = MPI.Wtime()

    if RANK == 0:
        rng = np.random.default_rng(seed=seed)
        elevation = rng.uniform(size=shape)
        uplift_rate = np.zeros_like(elevation)
        uplift_rate[1:-1, 1:-1] = 0.004
        status_at_node = np.full_like(elevation, NodeStatus.CORE, dtype=np.uint8)
        status_at_node[:, (0, -1)] = NodeStatus.FIXED_VALUE
        status_at_node[(0, -1), :] = NodeStatus.FIXED_VALUE

        if mode == "odd-r":
            tiler = OddRTiler.from_pymetis(shape, n_partitions, halo=2)
        else:
            tiler = D4Tiler.from_pymetis(shape, n_partitions, halo=2)

        for _rank in range(1, n_partitions):
            ij_of_lower_left = np.asarray([s.start for s in tiler[_rank]], dtype="i")
            tile = tiler.getvalue(_rank)

            comm.Send(np.array(tile.shape, dtype="i"), dest=_rank, tag=0)
            comm.Send(ij_of_lower_left, dest=_rank, tag=1)
            comm.Send(tile.flatten(), dest=_rank, tag=2)
            comm.Send(tiler.scatter(elevation)[_rank].flatten(), dest=_rank, tag=3)
            comm.Send(tiler.scatter(uplift_rate)[_rank].flatten(), dest=_rank, tag=4)
            comm.Send(tiler.scatter(status_at_node)[_rank].flatten(), dest=_rank, tag=5)

        tile = tiler.getvalue(RANK)
        tile_shape = np.array(tile.shape, dtype="i")
        ij_of_lower_left = np.asarray([s.start for s in tiler[RANK]], dtype="i")
        partition = np.asarray(tile, dtype=int)
        elevation = tiler.scatter(elevation)[RANK]
        uplift_rate = tiler.scatter(uplift_rate)[RANK]
        status_at_node = tiler.scatter(status_at_node)[RANK]
    else:
        tile_shape = np.empty(2, dtype="i")
        ij_of_lower_left = np.empty(2, dtype="i")
        comm.Recv(tile_shape, source=0, tag=0)
        comm.Recv(ij_of_lower_left, source=0, tag=1)

        partition = np.empty(tile_shape, dtype=int)
        elevation = np.empty(tile_shape, dtype=float)
        uplift_rate = np.empty(tile_shape, dtype=float)
        status_at_node = np.empty(tile_shape, dtype=np.uint8)
        comm.Recv(partition.reshape(-1), source=0, tag=2)
        comm.Recv(elevation.reshape(-1), source=0, tag=3)
        comm.Recv(uplift_rate.reshape(-1), source=0, tag=4)
        comm.Recv(status_at_node.reshape(-1), source=0, tag=5)

    comm.Barrier()
    global_grid_setup_time = MPI.Wtime() - global_grid_setup_started

    # Local setup includes tile metadata, ghost ownership, local-grid
    # construction, boundary status, and local fields.
    local_grid_setup_started = MPI.Wtime()

    my_tile = Tile(ij_of_lower_left, shape, partition, id_=RANK, mode=mode)

    my_ghosts = transform_values(
        dict(my_tile.get_ghost_nodes_by_owner()), my_tile.local_to_global
    )
    their_ghosts = send_receive_ghost_ids(comm, my_ghosts)

    my_ghosts = transform_values(my_ghosts, my_tile.global_to_local)
    their_ghosts = transform_values(their_ghosts, my_tile.global_to_local)

    grid = create_landlab_grid(
        partition,
        spacing=100.0,
        ij_of_lower_left=ij_of_lower_left,
        id_=RANK,
        mode=mode,
    )
    my_nodes = (partition == RANK).reshape(-1)
    status_at_node.shape = (-1,)
    grid.status_at_node[my_nodes] = status_at_node[my_nodes]

    grid.add_field("topographic__elevation", elevation, at="node")
    grid.add_field("uplift_rate", uplift_rate, at="node")

    comm.Barrier()
    local_grid_setup_time = MPI.Wtime() - local_grid_setup_started

    uplift = Uplift(grid)
    fa = FlowAccumulator(grid)
    sp = StreamPowerEroder(grid, K_sp=0.0001)
    ld = LinearDiffuser(grid, linear_diffusivity=0.01)

    components = [(uplift, (250.0,)), (ld, (250.0,)), (fa, ()), (sp, (250.0,))]

    # The complete component loop, including required ghost-data exchange, is
    # the primary strong/weak-scaling metric.
    comm.Barrier()
    model_loop_started = MPI.Wtime()

    num_time_steps = 200
    for _ in range(num_time_steps):
        for component, args in components:
            component.run_one_step(*args)
            send_receive_ghost_data(
                comm,
                my_ghosts,
                their_ghosts,
                (grid.at_node["topographic__elevation"], grid.at_node["drainage_area"]),
            )

    comm.Barrier()
    model_loop_time = MPI.Wtime() - model_loop_started

    if write_result_files:
        with open(f"{RANK}.vtu", "w") as fp:
            fp.write(vtu_dump(grid, z_coord="topographic__elevation"))

    if RANK == 0:
        tile_data = {0: grid.at_node["topographic__elevation"]}
        for rank in range(1, n_partitions):
            tile_data[rank] = np.empty(tiler.get_tile_size(rank), dtype=float)
            comm.Recv(tile_data[rank], source=rank, tag=0)

        if debug_plot:
            print_output(tiler.gather(tile_data))

        if write_result_files:
            with open("full.pvtu", "w") as fp:
                fp.write(
                    pvtu_dump(grid, [f"{rank}.vtu" for rank in range(n_partitions)])
                )
    else:
        comm.Send(grid.at_node["topographic__elevation"], dest=0, tag=0)

    # Measure from the first executable statement in this file through setup,
    # model execution, MPI communication, and result-file output. The slowest
    # rank determines the parallel wall-clock runtime.
    comm.Barrier()
    script_time = time.perf_counter() - script_started
    max_script_time = comm.reduce(script_time, op=MPI.MAX, root=0)
    max_global_grid_setup_time = comm.reduce(
        global_grid_setup_time, op=MPI.MAX, root=0
    )
    max_local_grid_setup_time = comm.reduce(
        local_grid_setup_time, op=MPI.MAX, root=0
    )
    max_model_loop_time = comm.reduce(model_loop_time, op=MPI.MAX, root=0)

    if RANK == 0:
        print(f"Complete Python script: {max_script_time:.6f} s", flush=True)
        print("\nWorkflow timing summary (maximum across MPI ranks)")
        print(f"  Global-grid setup:    {max_global_grid_setup_time:.6f} s")
        print(f"  Local-grid setup:     {max_local_grid_setup_time:.6f} s")
        print(f"  Model loop:           {max_model_loop_time:.6f} s")
        print(f"  Total workflow run:   {max_script_time:.6f} s", flush=True)

        timing_row = {
            "mpi_ranks": n_partitions,
            "grid_shape": f"{shape[0]}x{shape[1]}",
            "time_steps": num_time_steps,
            "global_grid_setup_s": format_seconds(max_global_grid_setup_time),
            "local_grid_setup_s": format_seconds(max_local_grid_setup_time),
            "model_loop_s": format_seconds(max_model_loop_time),
            "total_workflow_run_s": format_seconds(max_script_time),
        }
        timing_filename = (
            f"raster_timing_{run_id}_ranks_{n_partitions}_"
            f"shape_{shape[0]}x{shape[1]}.csv"
        )
        timing_path = timing_output_dir.resolve() / timing_filename
        write_timing_csv(timing_path, [timing_row])
        print(f"  Timing CSV:           {timing_path}", flush=True)

    return 0


class Uplift:
    def __init__(self, grid, uplift_rate=1.0):
        self.grid = grid

    def run_one_step(self, dt):
        z = self.grid.at_node["topographic__elevation"]
        dz_dt = self.grid.at_node["uplift_rate"]

        z[self.grid.core_nodes] += dz_dt[self.grid.core_nodes] * dt


def main():
    global timing_output_dir

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "size",
        type=int,
        nargs="?",
        help="legacy grid size; creates a SIZE by 2*SIZE global grid",
    )
    parser.add_argument(
        "--grid-shape",
        type=parse_grid_shape,
        metavar="ROWS,COLUMNS",
        help="explicit global grid shape",
    )
    parser.add_argument(
        "--mode", choices=("d4", "odd-r"), default="odd-r", help="Grid type"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for initial elevations (default: 1)",
    )
    parser.add_argument(
        "--timing-output-dir",
        type=Path,
        default=timing_output_dir,
        metavar="DIR",
        help=(
            "directory for the per-run timing CSV "
            "(default: performance_evaluation/raster_timing_results)"
        ),
    )
    args = parser.parse_args()
    timing_output_dir = args.timing_output_dir

    if args.grid_shape is not None and args.size is not None:
        parser.error("use either size or --grid-shape, not both")
    if args.grid_shape is None and args.size is None:
        parser.error("provide size or --grid-shape")

    shape = args.grid_shape or (args.size, 2 * args.size)

    return run(shape, args.mode, args.seed)


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


def print_output(array):
    if os.getenv("CI") == "true":
        import tabulate

        print(tabulate.tabulate(array))
    else:
        import matplotlib.pyplot as plt

        plt.imshow(array)
        plt.show()


def transform_values(d: dict[int, NDArray], xform, inplace=False):
    if inplace:
        for array in d.values():
            array[:] = xform(array)
        return d
    else:
        return {key: xform(array).astype("i") for key, array in d.items()}


def send_receive_ghost_ids(comm, my_ghosts: dict[int, ArrayLike]):
    my_rank = comm.Get_rank()

    my_count = np.empty(1, dtype="i")
    their_count = np.empty(1, dtype="i")

    their_ghosts = {}
    for rank in my_ghosts:
        my_count[0] = len(my_ghosts[rank])
        comm.Sendrecv(
            my_count,
            rank,
            sendtag=my_rank,
            recvbuf=their_count,
            source=rank,
            recvtag=rank,
        )
        their_ghosts[rank] = np.empty(their_count[0], dtype="i")

    for rank in my_ghosts:
        comm.Sendrecv(
            my_ghosts[rank],
            rank,
            sendtag=my_rank,
            recvbuf=their_ghosts[rank],
            source=rank,
            recvtag=rank,
        )

    return their_ghosts


def send_receive_ghost_data(comm, my_ghosts, their_ghosts, data: tuple[NDArray, ...]):
    my_rank = comm.Get_rank()

    for rank in my_ghosts:
        for values in data:
            data_to_send = values[their_ghosts[rank]]
            data_to_receive = np.empty(len(my_ghosts[rank]), dtype=float)

            comm.Sendrecv(
                data_to_send,
                rank,
                sendtag=my_rank,
                recvbuf=data_to_receive,
                source=rank,
                recvtag=rank,
            )

            values[my_ghosts[rank]] = data_to_receive


if __name__ == "__main__":
    SystemExit(main())
