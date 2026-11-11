"""This script is used for performance evaluation of weak scaling.

Goal:
    For a Landlab workflow, measure how well model-loop runtime remains
    constant when the global problem size and processor count increase
    proportionally, keeping the average workload per processor approximately
    constant.

Metrics:
    1. Model-loop runtime reported by the workflow.
    2. Weak-scaling efficiency: how well model-loop runtime remains constant as
       the global problem size and processor count increase proportionally,
       E(n) = T_model(reference) / T_model(n), with ideal E(n) = 1.
    3. Plot comparing ideal and measured scaling:
       - x-axis: number of processors (MPI ranks);
       - y-axis: model-loop runtime;
       - ideal line: constant reference model-loop runtime (horizontal line);
       - observed line: measured model-loop runtime for each processor count.


Example
-------
python performance_weak_scaling.py \
  --workflow raster_lem_example.py \
  --ranks 1 4 \
  --grid-shapes 300,300 600,600 \
  --repeats 1 \
  --results-dir weak_scaling_results_raster \
  --workflow-args --mode d4 --seed 1

python performance_weak_scaling.py \
    --workflow voronoi_lem_workflow.py \
    --ranks 1 4 \
    --grid-shapes 300,300 600,600 \
    --repeats 1 \
    --results-dir weak_scaling_results_voronoi

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
DEFAULT_RESULTS_PARENT = BASE_DIR / "weak_scaling_results"
DEFAULT_RANKS = [1, 4, 16, 64]
DEFAULT_GRID_SHAPES = [(250, 250), (500, 500), (1000, 1000), (2000, 2000)]
DEFAULT_REPEATS = 3
OUTLIER_Z_THRESHOLD = 3.5
WORKLOAD_RELATIVE_TOLERANCE = 0.05

RAW_FIELDS = (
    "workflow",
    "ranks",
    "global_shape",
    "time_steps",
    "repetition",
    "global_grid_setup_s",
    "local_grid_setup_s",
    "model_loop_s",
    "total_workflow_run_s",
    "command_runtime_s",
    "status",
    "excluded",
)
SUMMARY_FIELDS = (
    "ranks",
    "global_shape",
    "successful_runs",
    "excluded_outliers",
    "average_model_loop_s",
    "std_model_loop_s",
    "weak_scaling_efficiency",
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
    grid_shape: GridShape
    time_steps: int
    global_grid_setup_s: float
    local_grid_setup_s: float
    model_loop_s: float
    total_workflow_run_s: float
    csv_path: Path


@dataclass(frozen=True)
class BenchmarkCase:
    """One explicitly paired MPI-rank and global-grid-shape experiment."""

    ranks: int
    global_shape: GridShape


@dataclass
class RunResult:
    """Result of one complete MPI workflow launch."""

    workflow: str
    ranks: int
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
    excluded: bool = False

    def as_csv_row(self) -> dict[str, str | int]:
        return {
            "workflow": self.workflow,
            "ranks": self.ranks,
            "global_shape": format_grid_shape(self.global_shape),
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
            "excluded": str(self.excluded).lower(),
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
    """Parse one global grid shape written as ROWS,COLUMNS."""
    parts = value.split(",")
    if len(parts) != 2 or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            f"grid shape must be ROWS,COLUMNS, got {value!r}"
        )
    if any(part != part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            f"grid shape must not contain whitespace, got {value!r}"
        )

    try:
        rows, columns = (int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"grid shape dimensions must be integers, got {value!r}"
        ) from error

    if rows <= 0 or columns <= 0:
        raise argparse.ArgumentTypeError("grid shape dimensions must be positive")
    return rows, columns


def format_grid_shape(shape: GridShape) -> str:
    """Format a grid shape using the documented command-line syntax."""
    return f"{shape[0]},{shape[1]}"


def format_optional_seconds(value: float | None) -> str:
    """Format an optional timing value for raw CSV output."""
    return "" if value is None else f"{value:.9f}"


def parse_timing_grid_shape(value: str, path: Path) -> GridShape:
    """Normalize ROWSxCOLUMNS or ROWS,COLUMNS from a workflow timing CSV."""
    stripped = value.strip()
    if "x" in stripped.lower():
        parts = stripped.lower().split("x")
    else:
        parts = stripped.split(",")
    if len(parts) != 2 or any(not part or part != part.strip() for part in parts):
        raise BenchmarkError(
            f"invalid 'grid_shape' in workflow timing CSV {path}: {value!r}"
        )
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as error:
        raise BenchmarkError(
            f"invalid 'grid_shape' in workflow timing CSV {path}: {value!r}"
        ) from error
    if any(dimension <= 0 for dimension in shape):
        raise BenchmarkError(
            f"'grid_shape' must contain positive dimensions in workflow "
            f"timing CSV {path}"
        )
    return shape[0], shape[1]


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
    expected_ranks: int,
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
            missing = [
                field for field in WORKFLOW_TIMING_FIELDS if field not in fieldnames
            ]
            if missing:
                raise BenchmarkError(
                    f"workflow timing CSV {path} is missing columns: "
                    f"{', '.join(missing)}"
                )
            rows = list(reader)
    except OSError as error:
        raise BenchmarkError(
            f"could not read workflow timing CSV {path}: {error}"
        ) from error

    if len(rows) != 1:
        raise BenchmarkError(
            f"workflow timing CSV {path} must contain exactly one data row; "
            f"found {len(rows)}"
        )
    row = rows[0]

    mpi_ranks = _parse_timing_int(row, "mpi_ranks", path)
    if mpi_ranks != expected_ranks:
        raise BenchmarkError(
            f"workflow timing CSV {path} reports {mpi_ranks} MPI ranks; "
            f"expected {expected_ranks}"
        )

    grid_shape = parse_timing_grid_shape(row["grid_shape"], path)
    if grid_shape != expected_shape:
        raise BenchmarkError(
            f"workflow timing CSV {path} reports grid shape "
            f"{format_grid_shape(grid_shape)!r}; expected "
            f"{format_grid_shape(expected_shape)!r}"
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


def _results_path_inside_base(path: Path) -> Path:
    """Resolve a result folder name or path inside performance_evaluation."""
    if path.is_absolute():
        return _path_inside_base(path, "--results-dir")

    cwd_candidate = path.resolve()
    try:
        cwd_candidate.relative_to(BASE_DIR)
    except ValueError:
        return _path_inside_base(BASE_DIR / path, "--results-dir")
    return cwd_candidate


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sequential MPI weak-scaling experiments.",
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
        "--ranks",
        nargs="+",
        type=positive_int,
        default=list(DEFAULT_RANKS),
        metavar="N",
        help="ordered MPI rank counts (default: 1 4 16 64)",
    )
    parser.add_argument(
        "--grid-shapes",
        nargs="+",
        type=parse_grid_shape,
        default=list(DEFAULT_GRID_SHAPES),
        metavar="ROWS,COLUMNS",
        help=(
            "global shapes paired with --ranks "
            "(default: 250,250 500,500 1000,1000 2000,2000)"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=DEFAULT_REPEATS,
        metavar="N",
        help="runs per rank-shape pair, minimum 1 (default: 3)",
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
        help="also create runtime_log_x.png with a logarithmic x-axis",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        metavar="FOLDER",
        help=(
            "result parent folder inside performance_evaluation "
            "(default: weak_scaling_results)"
        ),
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

    if len(set(args.ranks)) != len(args.ranks):
        parser.error("--ranks must not contain duplicate values")
    if any(
        current >= following for current, following in zip(args.ranks, args.ranks[1:])
    ):
        parser.error("--ranks must be in strictly increasing order")
    if len(args.ranks) != len(args.grid_shapes):
        parser.error("--ranks and --grid-shapes must contain the same number of values")
    if any(
        value == "--grid-shape" or value.startswith("--grid-shape=")
        for value in args.workflow_args
    ):
        parser.error(
            "do not pass --grid-shape through --workflow-args; "
            "the benchmark manages it"
        )
    if any(
        value == "--timing-output-dir"
        or value.startswith("--timing-output-dir=")
        for value in args.workflow_args
    ):
        parser.error(
            "do not pass --timing-output-dir through --workflow-args; "
            "the benchmark manages it"
        )

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
        args.results_dir = _results_path_inside_base(results_parent)
    except ValueError as error:
        parser.error(str(error))
    if args.results_dir.exists() and not args.results_dir.is_dir():
        parser.error(f"--results-dir is not a directory: {args.results_dir}")

    args.cases = tuple(
        BenchmarkCase(ranks=ranks, global_shape=shape)
        for ranks, shape in zip(args.ranks, args.grid_shapes, strict=True)
    )
    return args


def workload_warnings(
    cases: Sequence[BenchmarkCase],
    relative_tolerance: float = WORKLOAD_RELATIVE_TOLERANCE,
) -> list[str]:
    """Return warnings for cases whose nominal work per rank differs materially."""
    if not cases:
        return []

    reference = math.prod(cases[0].global_shape) / cases[0].ranks
    warnings = []
    for case in cases[1:]:
        workload = math.prod(case.global_shape) / case.ranks
        if not math.isclose(workload, reference, rel_tol=relative_tolerance):
            relative_difference = abs(workload - reference) / reference
            warnings.append(
                f"{case.ranks} ranks with shape "
                f"{format_grid_shape(case.global_shape)} differs from the "
                f"reference workload per rank by {relative_difference:.1%}"
            )
    return warnings


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
    case: BenchmarkCase,
    repetition: int,
    mpi_launcher: str,
    timing_output_dir: Path,
    workflow_args: Sequence[str] = (),
) -> RunResult:
    """Launch one MPI run and read its workflow-owned model-loop timing."""
    if timing_output_dir.exists():
        if not timing_output_dir.is_dir() or any(timing_output_dir.iterdir()):
            raise BenchmarkError(
                "workflow timing directory must be a new empty directory: "
                f"{timing_output_dir}"
            )
    else:
        timing_output_dir.mkdir(parents=True)

    command = (
        mpi_launcher,
        "-n",
        str(case.ranks),
        sys.executable,
        str(workflow),
        "--grid-shape",
        format_grid_shape(case.global_shape),
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
                expected_ranks=case.ranks,
                expected_shape=case.global_shape,
            )
        except BenchmarkError as error:
            if stderr and not stderr.endswith("\n"):
                stderr += "\n"
            stderr += f"timing error: {error}\n"
        else:
            status = "success"

    return RunResult(
        workflow=workflow.name,
        ranks=case.ranks,
        global_shape=case.global_shape,
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
        f"ranks: {result.ranks}\n"
        f"global_shape: {format_grid_shape(result.global_shape)}\n"
        f"time_steps: {'' if result.time_steps is None else result.time_steps}\n"
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


def write_raw_csv(path: Path, results: Sequence[RunResult]) -> None:
    """Rewrite raw results after analysis so excluded flags are recorded."""
    initialize_raw_csv(path)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RAW_FIELDS)
        writer.writerows(result.as_csv_row() for result in results)


def filter_runtime_outliers(
    results: Sequence[RunResult],
) -> tuple[list[RunResult], list[RunResult]]:
    """Separate only clear model-loop outliers using modified z-scores."""
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
        if 0.67448975 * deviation / median_absolute_deviation > OUTLIER_Z_THRESHOLD
    ]
    if len(included) - len(excluded) < 2:
        return included, []

    excluded_ids = {id(result) for result in excluded}
    included = [result for result in included if id(result) not in excluded_ids]
    return included, excluded


def summarize_results(
    results: Sequence[RunResult], cases: Sequence[BenchmarkCase]
) -> list[dict[str, int | float | str]]:
    """Calculate model-loop averages and weak-scaling efficiency."""
    successful: dict[BenchmarkCase, list[RunResult]] = {case: [] for case in cases}
    for result in results:
        case = BenchmarkCase(result.ranks, result.global_shape)
        if result.status == "success" and case in successful:
            successful[case].append(result)

    missing = [case for case, values in successful.items() if not values]
    if missing:
        labels = ", ".join(
            f"{case.ranks} ranks/{format_grid_shape(case.global_shape)}"
            for case in missing
        )
        raise BenchmarkError("no successful measurements for: " + labels)

    time_steps = {
        result.time_steps for values in successful.values() for result in values
    }
    if None in time_steps:
        raise BenchmarkError("successful run is missing time_steps")
    if len(time_steps) != 1:
        raise BenchmarkError(
            "weak-scaling runs must use one fixed time-step count"
        )
    if any(
        result.model_loop_s is None
        for values in successful.values()
        for result in values
    ):
        raise BenchmarkError("successful run is missing model_loop_s")

    excluded_by_case: dict[BenchmarkCase, list[RunResult]] = {}
    for case in cases:
        included, excluded = filter_runtime_outliers(successful[case])
        successful[case] = included
        excluded_by_case[case] = excluded
        for result in excluded:
            result.excluded = True

    averages = {
        case: statistics.fmean(float(result.model_loop_s) for result in values)
        for case, values in successful.items()
    }
    baseline = averages[cases[0]]

    summary: list[dict[str, int | float | str]] = []
    for case in cases:
        retained = successful[case]
        runtimes = [float(result.model_loop_s) for result in retained]
        average = averages[case]
        summary.append(
            {
                "ranks": case.ranks,
                "global_shape": format_grid_shape(case.global_shape),
                "successful_runs": len(retained),
                "excluded_outliers": len(excluded_by_case[case]),
                "average_model_loop_s": average,
                "std_model_loop_s": (
                    statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
                ),
                "weak_scaling_efficiency": baseline / average,
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
                    "ranks": row["ranks"],
                    "global_shape": row["global_shape"],
                    "successful_runs": row["successful_runs"],
                    "excluded_outliers": row["excluded_outliers"],
                    "average_model_loop_s": (
                        f"{float(row['average_model_loop_s']):.9f}"
                    ),
                    "std_model_loop_s": (
                        f"{float(row['std_model_loop_s']):.9f}"
                    ),
                    "weak_scaling_efficiency": (
                        f"{float(row['weak_scaling_efficiency']):.9f}"
                    ),
                }
            )


def plot_runtime(
    summary: Sequence[dict[str, int | float | str]],
    path: Path,
    log_x: bool = False,
) -> None:
    """Plot observed model-loop runtime against ideal weak scaling."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except ImportError as error:
        raise BenchmarkError("matplotlib is required to create plots") from error

    ranks = [int(row["ranks"]) for row in summary]
    observed = [float(row["average_model_loop_s"]) for row in summary]
    ideal = [observed[0]] * len(observed)

    figure, axes = plt.subplots(figsize=(7, 5))
    axes.plot(ranks, ideal, "--", color="0.35", linewidth=2, label="Ideal")
    axes.plot(ranks, observed, "o-", linewidth=2, label="Observed")
    if log_x:
        axes.set_xscale("log", base=2)
        axes.xaxis.set_major_formatter(ScalarFormatter())
    axes.set_xticks(ranks)
    axes.set_xlabel("MPI ranks")
    axes.set_ylabel("Model-loop runtime (s)")
    axes.set_title("Weak-Scaling Model-Loop Runtime")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def print_summary(summary: Sequence[dict[str, int | float | str]]) -> None:
    print("\nWeak-scaling summary")
    print(
        f"{'ranks':>7} {'global_shape':>14} {'used':>6} {'excluded':>10} "
        f"{'avg_model_s':>14} {'efficiency':>12}"
    )
    for row in summary:
        print(
            f"{int(row['ranks']):>7d} "
            f"{str(row['global_shape']):>14} "
            f"{int(row['successful_runs']):>6d} "
            f"{int(row['excluded_outliers']):>10d} "
            f"{float(row['average_model_loop_s']):>14.6f} "
            f"{float(row['weak_scaling_efficiency']):>12.4f}"
        )


def run_benchmark(
    args: argparse.Namespace,
) -> tuple[Path, list[dict[str, int | float | str]]]:
    """Run a complete sequential weak-scaling benchmark campaign."""
    for warning in workload_warnings(args.cases):
        print(f"warning: {warning}", file=sys.stderr, flush=True)

    mpi_launcher = resolve_mpi_launcher(args.mpi_launcher)
    results_dir = create_results_directory(args.results_dir)
    logs_dir = results_dir / "run_logs"
    logs_dir.mkdir()
    workflow_timings_dir = results_dir / "workflow_timings"
    workflow_timings_dir.mkdir()
    raw_path = results_dir / "raw_runtime.csv"
    initialize_raw_csv(raw_path)

    results: list[RunResult] = []
    for case in args.cases:
        shape_text = format_grid_shape(case.global_shape)
        for repetition in range(1, args.repeats + 1):
            print(
                f"Running {case.ranks} MPI ranks, global shape {shape_text}, "
                f"repetition {repetition}/{args.repeats}...",
                flush=True,
            )
            result = run_workflow(
                args.workflow,
                case,
                repetition,
                mpi_launcher,
                workflow_timings_dir
                / f"ranks_{case.ranks}_repeat_{repetition}",
                args.workflow_args,
            )
            results.append(result)
            log_path = logs_dir / f"ranks_{case.ranks}_repeat_{repetition}.log"
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

        successful_count = sum(
            result.status == "success"
            and result.ranks == case.ranks
            and result.global_shape == case.global_shape
            for result in results
        )
        if successful_count == 0:
            raise BenchmarkError(
                "no successful runs for "
                f"{case.ranks} ranks/{shape_text}; diagnostics: {logs_dir}"
            )

    summary = summarize_results(results, args.cases)
    write_raw_csv(raw_path, results)
    write_summary_csv(results_dir / "weak_scaling_summary.csv", summary)
    plot_runtime(summary, results_dir / "runtime.png")
    if args.log_x:
        plot_runtime(summary, results_dir / "runtime_log_x.png", log_x=True)
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
