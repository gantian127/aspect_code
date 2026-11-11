"""
This script is used for performance evaluation of strong scaling.

Goal:
    For a fixed Landlab workflow, measure how much faster the simulation runs
    when processors are added (fixed problem size, increasing processor count).

Metrics:
    1. Model-loop runtime reported by the workflow.
    2. Speedup: how much faster the parallel run is, S(n) = T(1) / T(n).
    3. Parallel efficiency: how effectively n processors are used,
       E(n) = S(n) / n, with ideal E(n) = 1.
    4. Plot comparing ideal and measured scaling:
       - x-axis: number of processors (MPI ranks);
       - y-axis: speedup;
       - ideal line: S(n) = n;
       - measured line: speedup calculated from model-loop runtimes.

Example：
python performance_strong_scaling.py \
    --workflow raster_lem_example.py \
    --grid-shape 300,300 \
    --cores 1 2 4 \
    --repetitions 3 \
    --log-x \
    --results-dir strong_scaling_results_raster \
    --workflow-args --mode d4 --seed 1

python performance_strong_scaling.py \
    --workflow voronoi_lem_workflow.py \
    --grid-shape 300,300 \
    --cores 1 2 4 \
    --repetitions 3 \
    --log-x \
    --results-dir strong_scaling_results_voronoi

Results are saved under performance_evaluation/strong_scaling_results/.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_PARENT = BASE_DIR / "strong_scaling_results"

RAW_FIELDS = (
    "workflow",
    "cores",
    "grid_shape",
    "time_steps",
    "repetition",
    "global_grid_setup_s",
    "local_grid_setup_s",
    "model_loop_s",
    "total_workflow_run_s",
    "command_runtime_s",
    "status",
)
SUMMARY_FIELDS = (
    "cores",
    "successful_runs",
    "excluded_outliers",
    "average_model_loop_s",
    "std_model_loop_s",
    "speedup",
    "efficiency",
)
WORKFLOW_TIMING_FIELDS = (
    "mpi_ranks",
    "grid_shape",
    "time_steps",
    "global_grid_setup_s",
    "local_grid_setup_s",
    "model_loop_s",
    "total_workflow_run_s",
)
BASELINE_OUTLIER_Z_THRESHOLD = 3.5
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

GridShape = tuple[int, int]


class BenchmarkError(RuntimeError):
    """Raised when a benchmark campaign cannot produce valid results."""


@dataclass(frozen=True)
class WorkflowTiming:
    """Validated one-row timing summary produced by an MPI workflow."""

    mpi_ranks: int
    grid_shape: str
    time_steps: int
    global_grid_setup_s: float
    local_grid_setup_s: float
    model_loop_s: float
    total_workflow_run_s: float
    csv_path: Path


@dataclass(frozen=True)
class RunResult:
    """Result of one complete MPI workflow launch."""

    workflow: str
    cores: int
    global_shape: GridShape
    repetition: int
    time_steps: int | None
    global_grid_setup_s: float | None
    local_grid_setup_s: float | None
    model_loop_s: float | None
    total_workflow_run_s: float | None
    command_runtime_s: float
    timing_csv_path: Path | None
    status: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def as_csv_row(self) -> dict[str, str | int | float]:
        return {
            "workflow": self.workflow,
            "cores": self.cores,
            "grid_shape": format_timing_grid_shape(self.global_shape),
            "time_steps": "" if self.time_steps is None else self.time_steps,
            "repetition": self.repetition,
            "global_grid_setup_s": format_optional_seconds(
                self.global_grid_setup_s
            ),
            "local_grid_setup_s": format_optional_seconds(self.local_grid_setup_s),
            "model_loop_s": format_optional_seconds(self.model_loop_s),
            "total_workflow_run_s": format_optional_seconds(
                self.total_workflow_run_s
            ),
            "command_runtime_s": f"{self.command_runtime_s:.9f}",
            "status": self.status,
        }


def positive_int(value: str) -> int:
    """Return a positive integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_grid_shape(value: str) -> GridShape:
    """Parse a positive two-dimensional global grid shape."""
    parts = value.split(",")
    if len(parts) != 2 or any(not part or part != part.strip() for part in parts):
        raise argparse.ArgumentTypeError("grid shape must be ROWS,COLUMNS")

    shape = tuple(positive_int(part) for part in parts)
    return shape[0], shape[1]


def format_grid_shape(shape: GridShape) -> str:
    """Format a grid shape for commands, logs, and tabular output."""
    return f"{shape[0]},{shape[1]}"


def format_timing_grid_shape(shape: GridShape) -> str:
    """Format a grid shape as written by workflow timing CSV files."""
    return f"{shape[0]}x{shape[1]}"


def format_optional_seconds(value: float | None) -> str:
    """Format an optional timing value for raw CSV output."""
    return "" if value is None else f"{value:.9f}"


def _parse_timing_int(row: dict[str, str], field: str, path: Path) -> int:
    try:
        value = int(row[field])
    except (TypeError, ValueError) as error:
        raise BenchmarkError(
            f"invalid {field!r} in workflow timing CSV {path}: {row[field]!r}"
        ) from error
    if value <= 0:
        raise BenchmarkError(
            f"{field!r} must be positive in workflow timing CSV {path}"
        )
    return value


def _parse_timing_float(
    row: dict[str, str], field: str, path: Path, *, positive: bool
) -> float:
    try:
        value = float(row[field])
    except (TypeError, ValueError) as error:
        raise BenchmarkError(
            f"invalid {field!r} in workflow timing CSV {path}: {row[field]!r}"
        ) from error
    if not math.isfinite(value) or (value <= 0.0 if positive else value < 0.0):
        requirement = "finite and positive" if positive else "finite and non-negative"
        raise BenchmarkError(
            f"{field!r} must be {requirement} in workflow timing CSV {path}"
        )
    return value


def read_workflow_timing(
    timing_output_dir: Path,
    expected_cores: int,
    expected_shape: GridShape,
) -> WorkflowTiming:
    """Read and validate exactly one workflow-generated timing CSV."""
    timing_files = sorted(timing_output_dir.glob("*.csv"))
    if len(timing_files) != 1:
        raise BenchmarkError(
            f"expected exactly one timing CSV in {timing_output_dir}, "
            f"found {len(timing_files)}"
        )

    path = timing_files[0]
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fieldnames = tuple(reader.fieldnames or ())
            missing = [field for field in WORKFLOW_TIMING_FIELDS if field not in fieldnames]
            if missing:
                raise BenchmarkError(
                    f"workflow timing CSV {path} is missing columns: "
                    f"{', '.join(missing)}"
                )
            rows = list(reader)
    except OSError as error:
        raise BenchmarkError(f"could not read workflow timing CSV {path}: {error}") from error

    if len(rows) != 1:
        raise BenchmarkError(
            f"workflow timing CSV {path} must contain exactly one data row; "
            f"found {len(rows)}"
        )
    row = rows[0]

    mpi_ranks = _parse_timing_int(row, "mpi_ranks", path)
    if mpi_ranks != expected_cores:
        raise BenchmarkError(
            f"workflow timing CSV {path} reports {mpi_ranks} MPI ranks; "
            f"expected {expected_cores}"
        )

    grid_shape = row["grid_shape"].strip()
    expected_grid_shape = format_timing_grid_shape(expected_shape)
    if grid_shape != expected_grid_shape:
        raise BenchmarkError(
            f"workflow timing CSV {path} reports grid shape {grid_shape!r}; "
            f"expected {expected_grid_shape!r}"
        )

    return WorkflowTiming(
        mpi_ranks=mpi_ranks,
        grid_shape=grid_shape,
        time_steps=_parse_timing_int(row, "time_steps", path),
        global_grid_setup_s=_parse_timing_float(
            row, "global_grid_setup_s", path, positive=False
        ),
        local_grid_setup_s=_parse_timing_float(
            row, "local_grid_setup_s", path, positive=False
        ),
        model_loop_s=_parse_timing_float(row, "model_loop_s", path, positive=True),
        total_workflow_run_s=_parse_timing_float(
            row, "total_workflow_run_s", path, positive=True
        ),
        csv_path=path.resolve(),
    )


def _path_inside_base(path: Path, label: str) -> Path:
    """Resolve a path and require it to remain under performance_evaluation."""
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(BASE_DIR)
    except ValueError as error:
        raise ValueError(f"{label} must be inside {BASE_DIR}: {resolved}") from error
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential MPI strong-scaling experiments.",
        epilog=(
            "Put --workflow-args last; for example: "
            "--workflow-args --mode odd-r --seed 1"
        ),
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        required=True,
        help="required MPI Python workflow inside performance_evaluation",
    )
    parser.add_argument(
        "--grid-shape",
        type=parse_grid_shape,
        required=True,
        metavar="ROWS,COLUMNS",
        help="fixed global grid shape used for every MPI rank count",
    )
    parser.add_argument(
        "--cores",
        nargs="+",
        type=positive_int,
        default=[1, 2, 4],
        metavar="N",
        help="ordered MPI rank counts (default: 1 2 4)",
    )
    parser.add_argument(
        "--repetitions",
        type=positive_int,
        default=3,
        metavar="N",
        help="runs per MPI rank count (default: 3)",
    )
    parser.add_argument(
        "--mpi-launcher",
        default="mpiexec",
        metavar="PATH",
        help="MPI launcher name or path (default: mpiexec)",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="also create speedup_log_x.png with a logarithmic x-axis",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="results parent inside performance_evaluation",
    )
    parser.add_argument(
        "--workflow-args",
        nargs=argparse.REMAINDER,
        default=(),
        metavar="ARG",
        help="remaining arguments passed unchanged to the workflow; must be last",
    )
    args = parser.parse_args(argv)
    args.workflow_args = tuple(args.workflow_args)

    if any(
        value == "--timing-output-dir"
        or value.startswith("--timing-output-dir=")
        for value in args.workflow_args
    ):
        parser.error(
            "do not pass --timing-output-dir through --workflow-args; "
            "the benchmark manages it"
        )

    if len(set(args.cores)) != len(args.cores):
        parser.error("--cores must not contain duplicate values")
    if 1 not in args.cores:
        parser.error("--cores must include 1 to provide the speedup baseline")

    try:
        args.workflow = _path_inside_base(args.workflow, "--workflow")
    except ValueError as error:
        parser.error(str(error))
    if not args.workflow.is_file():
        parser.error(f"workflow does not exist: {args.workflow}")
    if args.workflow.suffix.lower() != ".py":
        parser.error(f"workflow must be a Python file: {args.workflow}")

    results_parent = args.results_dir or DEFAULT_RESULTS_PARENT
    try:
        args.results_dir = _path_inside_base(results_parent, "--results-dir")
    except ValueError as error:
        parser.error(str(error))
    if args.results_dir.exists() and not args.results_dir.is_dir():
        parser.error(f"--results-dir is not a directory: {args.results_dir}")

    return args


def resolve_mpi_launcher(value: str) -> str:
    """Resolve an MPI launcher without invoking a shell."""
    launcher = shutil.which(value)
    if launcher is None:
        raise BenchmarkError(
            f"MPI launcher {value!r} was not found. Install MPI or pass "
            "--mpi-launcher with a valid executable."
        )
    return launcher


def create_results_directory(parent: Path, now: datetime | None = None) -> Path:
    """Create a timestamped campaign directory without overwriting old data."""
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    candidate = parent / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = parent / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def run_workflow(
    workflow: Path,
    cores: int,
    global_shape: GridShape,
    repetition: int,
    mpi_launcher: str,
    timing_output_dir: Path,
    workflow_args: Sequence[str] = (),
) -> RunResult:
    """Launch one MPI run and read its workflow-owned model-loop timing."""
    if timing_output_dir.exists():
        if not timing_output_dir.is_dir() or any(timing_output_dir.iterdir()):
            raise BenchmarkError(
                f"workflow timing directory must be a new empty directory: "
                f"{timing_output_dir}"
            )
    else:
        timing_output_dir.mkdir(parents=True)

    command = (
        mpi_launcher,
        "-n",
        str(cores),
        sys.executable,
        str(workflow),
        "--grid-shape",
        format_grid_shape(global_shape),
        "--timing-output-dir",
        str(timing_output_dir.resolve()),
        *workflow_args,
    )
    environment = os.environ.copy()
    environment.update(SINGLE_THREAD_ENV)

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=workflow.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except OSError as error:
        returncode = -1
        stdout = ""
        stderr = f"{type(error).__name__}: {error}\n"
    command_runtime_s = time.perf_counter() - started

    timing: WorkflowTiming | None = None
    status = "failed"
    if returncode == 0:
        try:
            timing = read_workflow_timing(
                timing_output_dir,
                expected_cores=cores,
                expected_shape=global_shape,
            )
        except BenchmarkError as error:
            if stderr and not stderr.endswith("\n"):
                stderr += "\n"
            stderr += f"timing error: {error}\n"
        else:
            status = "success"

    return RunResult(
        workflow=workflow.name,
        cores=cores,
        global_shape=global_shape,
        repetition=repetition,
        time_steps=None if timing is None else timing.time_steps,
        global_grid_setup_s=(
            None if timing is None else timing.global_grid_setup_s
        ),
        local_grid_setup_s=None if timing is None else timing.local_grid_setup_s,
        model_loop_s=None if timing is None else timing.model_loop_s,
        total_workflow_run_s=(
            None if timing is None else timing.total_workflow_run_s
        ),
        command_runtime_s=command_runtime_s,
        timing_csv_path=None if timing is None else timing.csv_path,
        status=status,
        command=command,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def write_run_log(path: Path, result: RunResult) -> None:
    """Write complete diagnostics for one launch."""
    content = (
        f"command: {shlex.join(result.command)}\n"
        f"cores: {result.cores}\n"
        f"global_shape: {format_grid_shape(result.global_shape)}\n"
        f"repetition: {result.repetition}\n"
        f"exit_code: {result.returncode}\n"
        f"model_loop_s: {format_optional_seconds(result.model_loop_s)}\n"
        f"global_grid_setup_s: "
        f"{format_optional_seconds(result.global_grid_setup_s)}\n"
        f"local_grid_setup_s: "
        f"{format_optional_seconds(result.local_grid_setup_s)}\n"
        f"total_workflow_run_s: "
        f"{format_optional_seconds(result.total_workflow_run_s)}\n"
        f"command_runtime_s: {result.command_runtime_s:.9f}\n"
        f"timing_csv: "
        f"{'' if result.timing_csv_path is None else result.timing_csv_path}\n"
        f"status: {result.status}\n\n"
        "--- stdout ---\n"
        f"{result.stdout}"
        "\n--- stderr ---\n"
        f"{result.stderr}"
    )
    path.write_text(content, encoding="utf-8")


def initialize_raw_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writeheader()


def append_raw_result(path: Path, result: RunResult) -> None:
    """Persist one measurement immediately after its launch."""
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writerow(result.as_csv_row())


def filter_runtime_outliers(
    results: Sequence[RunResult],
) -> tuple[list[RunResult], list[RunResult]]:
    """Separate clear model-loop outliers using modified z-scores.

    At least three successful measurements are required. When the median
    absolute deviation is zero, no automatic exclusion is made because the
    modified z-score is undefined.
    """
    included = list(results)
    if len(included) < 3:
        return included, []

    if any(result.model_loop_s is None for result in included):
        raise BenchmarkError("successful run is missing model_loop_s")
    runtimes = [float(result.model_loop_s) for result in included]
    median = statistics.median(runtimes)
    deviations = [abs(runtime - median) for runtime in runtimes]
    median_absolute_deviation = statistics.median(deviations)
    if median_absolute_deviation == 0.0:
        return included, []

    excluded = [
        result
        for result, deviation in zip(included, deviations, strict=True)
        if 0.67448975 * deviation / median_absolute_deviation
        > BASELINE_OUTLIER_Z_THRESHOLD
    ]
    if len(included) - len(excluded) < 2:
        return included, []

    excluded_repetitions = {result.repetition for result in excluded}
    included = [
        result for result in included if result.repetition not in excluded_repetitions
    ]
    return included, excluded


def summarize_results(
    results: Sequence[RunResult], cores_order: Sequence[int]
) -> list[dict[str, int | float | str]]:
    """Calculate model-loop averages, speedup, and efficiency."""
    successful: dict[int, list[RunResult]] = {cores: [] for cores in cores_order}
    for result in results:
        if result.status == "success" and result.cores in successful:
            successful[result.cores].append(result)

    missing = [cores for cores, values in successful.items() if not values]
    if missing:
        values = ", ".join(str(cores) for cores in missing)
        raise BenchmarkError(f"no successful measurements for MPI ranks: {values}")

    shapes = {
        result.global_shape for values in successful.values() for result in values
    }
    if len(shapes) != 1:
        raise BenchmarkError("strong-scaling runs must use one fixed global shape")

    time_steps = {
        result.time_steps for values in successful.values() for result in values
    }
    if None in time_steps:
        raise BenchmarkError("successful run is missing time_steps")
    if len(time_steps) != 1:
        raise BenchmarkError("strong-scaling runs must use one fixed time-step count")

    excluded_by_cores: dict[int, list[RunResult]] = {}
    for cores in cores_order:
        included, excluded = filter_runtime_outliers(successful[cores])
        successful[cores] = included
        excluded_by_cores[cores] = excluded

    averages = {
        cores: statistics.fmean(float(result.model_loop_s) for result in values)
        for cores, values in successful.items()
    }
    baseline = averages[1]
    summary: list[dict[str, int | float | str]] = []
    for cores in cores_order:
        core_results = successful[cores]
        values = [float(result.model_loop_s) for result in core_results]
        average = averages[cores]
        speedup = baseline / average
        summary.append(
            {
                "cores": cores,
                "successful_runs": len(values),
                "excluded_outliers": len(excluded_by_cores[cores]),
                "average_model_loop_s": average,
                "std_model_loop_s": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
                "speedup": speedup,
                "efficiency": speedup / cores,
            }
        )
    return summary


def write_summary_csv(
    path: Path, summary: Sequence[dict[str, int | float | str]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    "cores": row["cores"],
                    "successful_runs": row["successful_runs"],
                    "excluded_outliers": row["excluded_outliers"],
                    "average_model_loop_s": (
                        f"{float(row['average_model_loop_s']):.9f}"
                    ),
                    "std_model_loop_s": f"{float(row['std_model_loop_s']):.9f}",
                    "speedup": f"{float(row['speedup']):.9f}",
                    "efficiency": f"{float(row['efficiency']):.9f}",
                }
            )


def plot_speedup(
    summary: Sequence[dict[str, int | float | str]],
    path: Path,
    log_x: bool = False,
) -> None:
    """Plot measured speedup against ideal strong scaling."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError as error:
        raise BenchmarkError("matplotlib is required to create plots") from error

    cores = [int(row["cores"]) for row in summary]
    speedups = [float(row["speedup"]) for row in summary]

    figure, axes = plt.subplots(figsize=(7, 5))
    axes.plot(cores, cores, "--", color="0.35", label="Ideal")
    axes.plot(cores, speedups, "o-", linewidth=2, label="Measured")
    if log_x:
        axes.set_xscale("log", base=2)
        axes.xaxis.set_major_formatter(ScalarFormatter())
    axes.set_xticks(cores)
    axes.set_xlabel("MPI ranks")
    axes.set_ylabel("Speedup")
    axes.set_title("Strong-Scaling Model-Loop Speedup")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def print_summary(summary: Sequence[dict[str, int | float | str]]) -> None:
    print("\nStrong-scaling summary")
    print(
        f"{'ranks':>7} {'used':>6} {'excluded':>10} "
        f"{'avg_model_s':>14} {'speedup':>11} {'efficiency':>13}"
    )
    for row in summary:
        print(
            f"{int(row['cores']):>7d} "
            f"{int(row['successful_runs']):>6d} "
            f"{int(row['excluded_outliers']):>10d} "
            f"{float(row['average_model_loop_s']):>14.6f} "
            f"{float(row['speedup']):>11.4f} "
            f"{float(row['efficiency']):>13.4f}"
        )


def run_benchmark(
    args: argparse.Namespace,
) -> tuple[Path, list[dict[str, int | float | str]]]:
    """Run a complete sequential benchmark campaign."""
    mpi_launcher = resolve_mpi_launcher(args.mpi_launcher)
    results_dir = create_results_directory(args.results_dir)
    logs_dir = results_dir / "run_logs"
    logs_dir.mkdir()
    workflow_timings_dir = results_dir / "workflow_timings"
    workflow_timings_dir.mkdir()
    raw_path = results_dir / "raw_runtime.csv"
    initialize_raw_csv(raw_path)

    results: list[RunResult] = []
    for cores in args.cores:
        for repetition in range(1, args.repetitions + 1):
            print(
                f"Running {cores} MPI ranks, "
                f"global shape {format_grid_shape(args.grid_shape)}, "
                f"repetition {repetition}/{args.repetitions}...",
                flush=True,
            )
            result = run_workflow(
                args.workflow,
                cores,
                args.grid_shape,
                repetition,
                mpi_launcher,
                workflow_timings_dir / f"cores_{cores}_repeat_{repetition}",
                args.workflow_args,
            )
            results.append(result)
            log_path = logs_dir / f"cores_{cores}_repeat_{repetition}.log"
            write_run_log(log_path, result)
            append_raw_result(raw_path, result)

            if result.status == "success":
                print(
                    f"Completed; model loop {float(result.model_loop_s):.2f} "
                    "seconds.",
                    flush=True,
                )
            else:
                print(
                    f"Failed after {result.command_runtime_s:.2f} seconds; "
                    f"see {log_path}.",
                    file=sys.stderr,
                    flush=True,
                )

        if not any(
            result.status == "success" and result.cores == cores for result in results
        ):
            raise BenchmarkError(
                f"all runs failed for {cores} MPI ranks; diagnostics: {logs_dir}"
            )

    summary = summarize_results(results, args.cores)
    write_summary_csv(results_dir / "strong_scaling_summary.csv", summary)
    plot_speedup(summary, results_dir / "speedup.png")
    if args.log_x:
        plot_speedup(summary, results_dir / "speedup_log_x.png", log_x=True)
    return results_dir, summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results_dir, summary = run_benchmark(args)
    except BenchmarkError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.", file=sys.stderr)
        return 130

    print_summary(summary)
    print(f"\nResults: {results_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
