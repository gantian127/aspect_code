"""
This script is used for performance evaluation of strong scaling.

Goal:
    For a fixed Landlab workflow, measure how much faster the simulation runs
    when processors are added (fixed problem size, increasing processor count).

Metrics:
    1. Runtime.
    2. Speedup: how much faster the parallel run is, S(n) = T(1) / T(n).
    3. Parallel efficiency: how effectively n processors are used,
       E(n) = S(n) / n, with ideal E(n) = 1.
    4. Plot comparing ideal and measured scaling:
       - x-axis: number of processors (MPI ranks);
       - y-axis: speedup;
       - ideal line: S(n) = n;
       - measured line: speedup calculated from simulation runtimes.

Example：
python performance_strong_scaling.py \
    --workflow voronoi_lem_workflow.py \
    --cores 1 2 4 8 16 \
    --repetitions 3 \
    --mpi-launcher mpiexec \
    --log-x \
    --results-dir strong_scaling_results

Results are saved under performance_evaluation/strong_scaling_results/.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Sequence


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_PARENT = BASE_DIR / "strong_scaling_results"

RAW_FIELDS = ("workflow", "cores", "repetition", "runtime_s", "status")
SUMMARY_FIELDS = (
    "cores",
    "successful_runs",
    "average_runtime_s",
    "std_runtime_s",
    "speedup",
    "efficiency",
)
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


class BenchmarkError(RuntimeError):
    """Raised when a benchmark campaign cannot produce valid results."""


@dataclass(frozen=True)
class RunResult:
    """Result of one complete MPI workflow launch."""

    workflow: str
    cores: int
    repetition: int
    runtime_s: float
    status: str
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    def as_csv_row(self) -> dict[str, str | int | float]:
        return {
            "workflow": self.workflow,
            "cores": self.cores,
            "repetition": self.repetition,
            "runtime_s": f"{self.runtime_s:.9f}",
            "status": self.status,
        }


def positive_int(value: str) -> int:
    """Return a positive integer for argparse."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


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
        description="Run sequential MPI strong-scaling experiments."
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        required=True,
        help="required MPI Python workflow inside performance_evaluation",
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
    args = parser.parse_args(argv)

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
    repetition: int,
    mpi_launcher: str,
) -> RunResult:
    """Launch one MPI run and measure its end-to-end wall-clock runtime."""
    command = (
        mpi_launcher,
        "-n",
        str(cores),
        sys.executable,
        str(workflow),
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
    runtime_s = time.perf_counter() - started
    status = "success" if returncode == 0 else "failed"

    return RunResult(
        workflow=workflow.name,
        cores=cores,
        repetition=repetition,
        runtime_s=runtime_s,
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
        f"repetition: {result.repetition}\n"
        f"exit_code: {result.returncode}\n"
        f"runtime_s: {result.runtime_s:.9f}\n"
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


def summarize_results(
    results: Sequence[RunResult], cores_order: Sequence[int]
) -> list[dict[str, int | float]]:
    """Calculate average runtime, speedup, and efficiency."""
    successful: dict[int, list[float]] = {cores: [] for cores in cores_order}
    for result in results:
        if result.status == "success" and result.cores in successful:
            successful[result.cores].append(result.runtime_s)

    missing = [cores for cores, values in successful.items() if not values]
    if missing:
        values = ", ".join(str(cores) for cores in missing)
        raise BenchmarkError(f"no successful measurements for MPI ranks: {values}")

    averages = {
        cores: statistics.fmean(values) for cores, values in successful.items()
    }
    baseline = averages[1]
    summary: list[dict[str, int | float]] = []
    for cores in cores_order:
        values = successful[cores]
        average = averages[cores]
        speedup = baseline / average
        summary.append(
            {
                "cores": cores,
                "successful_runs": len(values),
                "average_runtime_s": average,
                "std_runtime_s": statistics.stdev(values) if len(values) > 1 else 0.0,
                "speedup": speedup,
                "efficiency": speedup / cores,
            }
        )
    return summary


def write_summary_csv(
    path: Path, summary: Sequence[dict[str, int | float]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    "cores": row["cores"],
                    "successful_runs": row["successful_runs"],
                    "average_runtime_s": f"{float(row['average_runtime_s']):.9f}",
                    "std_runtime_s": f"{float(row['std_runtime_s']):.9f}",
                    "speedup": f"{float(row['speedup']):.9f}",
                    "efficiency": f"{float(row['efficiency']):.9f}",
                }
            )


def plot_speedup(
    summary: Sequence[dict[str, int | float]], path: Path, log_x: bool = False
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
    axes.set_title("Strong-Scaling Speedup")
    axes.grid(True, alpha=0.3)
    axes.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def print_summary(summary: Sequence[dict[str, int | float]]) -> None:
    print("\nStrong-scaling summary")
    print(f"{'ranks':>7} {'average_s':>14} {'speedup':>11} {'efficiency':>13}")
    for row in summary:
        print(
            f"{int(row['cores']):>7d} "
            f"{float(row['average_runtime_s']):>14.6f} "
            f"{float(row['speedup']):>11.4f} "
            f"{float(row['efficiency']):>13.4f}"
        )


def run_benchmark(args: argparse.Namespace) -> tuple[Path, list[dict[str, int | float]]]:
    """Run a complete sequential benchmark campaign."""
    mpi_launcher = resolve_mpi_launcher(args.mpi_launcher)
    results_dir = create_results_directory(args.results_dir)
    logs_dir = results_dir / "run_logs"
    logs_dir.mkdir()
    raw_path = results_dir / "raw_runtime.csv"
    initialize_raw_csv(raw_path)

    results: list[RunResult] = []
    for cores in args.cores:
        for repetition in range(1, args.repetitions + 1):
            print(
                f"Running {cores} MPI ranks, "
                f"repetition {repetition}/{args.repetitions}...",
                flush=True,
            )
            result = run_workflow(
                args.workflow,
                cores,
                repetition,
                mpi_launcher,
            )
            results.append(result)
            log_path = logs_dir / f"cores_{cores}_repeat_{repetition}.log"
            write_run_log(log_path, result)
            append_raw_result(raw_path, result)

            if result.status == "success":
                print(f"Completed in {result.runtime_s:.2f} seconds.", flush=True)
            else:
                print(
                    f"Failed after {result.runtime_s:.2f} seconds; see {log_path}.",
                    file=sys.stderr,
                    flush=True,
                )

        if not any(
            result.status == "success" and result.cores == cores
            for result in results
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
