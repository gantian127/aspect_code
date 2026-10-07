# Codex Instruction: Strong-Scaling Evaluation

## Objective

Create a reusable Python strong-scaling benchmark driver for MPI-based Python
workflows. Use `performance_evaluation/voronoi_lem_workflow.py` as the first real
validation case, but do not hard-code the driver to that workflow. The user must
select a compatible workflow with the required `--workflow` option.

For a fixed scientific problem, measure how runtime changes when the number of
MPI ranks increases. Calculate:

1. Average runtime for each MPI rank count.
2. Speedup: `S(n) = T(1) / T(n)`.
3. Parallel efficiency: `E(n) = S(n) / n`.
4. A speedup plot comparing measured scaling with ideal scaling.

`T(n)` is the average runtime measured for `n` MPI ranks. `T(1)` must come from
the same MPI workflow executed with one rank.

## Scope

All work for this task must remain inside `performance_evaluation/`.

- Inspect `performance_evaluation/voronoi_lem_workflow.py` as the initial
  workflow being measured.
- Implement the benchmark in
  `performance_evaluation/performance_strong_scaling.py`.
- Save every generated CSV file and plot under
  `performance_evaluation/strong_scaling_results/`.
- Add tests or supporting files only inside `performance_evaluation/`.
- Do not inspect or modify source code outside `performance_evaluation/`.

Alternative workflow files supplied through `--workflow` must also be located
inside `performance_evaluation/`. Treat the selected workflow as a black-box MPI
executable. Do not hard-code its imports, functions, model type, result-file
names, or scientific parameters into the benchmark driver.

Preserve the scientific behavior of the selected workflow. For
`voronoi_lem_workflow.py`, do not change its grid, model parameters, active
components, time steps, or numerical calculation between MPI rank counts. For
any future workflow, use its unchanged default problem for every launch. The
global problem size must remain fixed throughout each strong-scaling experiment.

## Experiment execution

The benchmark script must run different MPI rank-count experiments sequentially,
not concurrently. For the default configuration, use this order:

```text
1 rank -> all repetitions
2 ranks -> all repetitions
4 ranks -> all repetitions
8 ranks -> all repetitions
```

Each individual MPI launch uses its requested ranks concurrently, but two
different rank-count experiments must never overlap. Concurrent experiments
would compete for CPU, memory bandwidth, and I/O and invalidate the timing.

For the default workflow, launch commands equivalent to:

```bash
mpiexec -n 1 python performance_evaluation/voronoi_lem_workflow.py
mpiexec -n 2 python performance_evaluation/voronoi_lem_workflow.py
mpiexec -n 4 python performance_evaluation/voronoi_lem_workflow.py
mpiexec -n 8 python performance_evaluation/voronoi_lem_workflow.py
```

Do not use `--oversubscribe` for performance measurements.

## Command-line interface

A user must be able to run the full benchmark, calculate the metrics, and create
the plots with one command:

```bash
python performance_evaluation/performance_strong_scaling.py \
  --workflow performance_evaluation/voronoi_lem_workflow.py
```

Default settings:

- MPI ranks: `1 2 4 8`;
- repetitions per rank count: `3`;
- MPI launcher: `mpiexec`;
- linear plot axes.

Users must be able to change MPI ranks without editing Python source:

```bash
python performance_evaluation/performance_strong_scaling.py \
  --cores 1 2 4 8 16 \
  --repetitions 5
```

Users must also be able to select another MPI Python workflow without editing
the benchmark source:

```bash
python performance_evaluation/performance_strong_scaling.py \
  --workflow performance_evaluation/another_workflow.py \
  --cores 1 2 4 8 \
  --repetitions 3
```

Support at least these options:

- `--workflow PATH`: required MPI Python script to benchmark; do not provide a
  default workflow;
- `--cores N [N ...]`: ordered positive MPI rank counts;
- `--repetitions N`: positive number of runs per rank count;
- `--mpi-launcher PATH`: launcher name or path, default `mpiexec`;
- `--log-x`: additionally create a speedup plot with a logarithmic x-axis;
- `--results-dir PATH`: optional parent output directory.

Validate all arguments. Reject duplicate, zero, or negative rank counts and
non-positive repetition counts with clear error messages.

## Runtime collection

For every rank count, run the workflow the requested number of times and retain
every individual runtime. Do not keep only the average.

Measure the elapsed wall-clock time of the complete MPI command with a monotonic
high-resolution timer such as `time.perf_counter()`. Start immediately before
launching `mpiexec` and stop only after the MPI command exits. This measures the
end-to-end time observed by the user; completion naturally waits for all ranks.

For each launch:

- wait for it to finish before starting the next launch;
- record rank count, repetition number, and runtime in seconds;
- check the process exit status;
- do not include failed runs in the average;
- stop with a clear error if a rank count has no successful measurements.

Set common threaded numerical libraries to one thread for every launch so the
experiment measures MPI scaling rather than uncontrolled hybrid parallelism:

```text
OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

## Calculations

After all experiments finish, calculate the metrics from successful raw
measurements.

For `r` successful repetitions at rank count `n`:

```text
T(n) = sum(runtime[n, i] for i = 1..r) / r
S(n) = T(1) / T(n)
E(n) = S(n) / n
```

The rank list must contain `1`, because it supplies the baseline. Calculate all
metrics only after the complete sequence of experiments has finished.

Also calculate the runtime standard deviation for each rank count so the user
can see run-to-run variability. The required speedup and efficiency values must
still use the average runtime, as defined above.

## Result storage

Create one timestamped directory per complete benchmark run:

```text
performance_evaluation/strong_scaling_results/run_YYYY-MM-DD_HHMMSS/
```

Do not overwrite an earlier experiment. Save only the following required
artifacts:

### `raw_runtime.csv`

One row per individual workflow launch:

```text
workflow,cores,repetition,runtime_s,status
voronoi_lem_workflow.py,1,1,20.3,success
voronoi_lem_workflow.py,1,2,20.1,success
voronoi_lem_workflow.py,2,1,11.2,success
```

### `strong_scaling_summary.csv`

One row per MPI rank count:

```text
cores,successful_runs,average_runtime_s,std_runtime_s,speedup,efficiency
1,5,20.27,0.12,1.00,1.00
2,5,11.10,0.09,1.83,0.91
4,5,6.40,0.08,3.17,0.79
```

Write efficiency as a fraction, where ideal efficiency is `1.0`.

### `speedup.png`

Create the primary speedup plot using linear axes:

- x-axis: number of MPI ranks;
- y-axis: speedup;
- ideal line: `S(n) = n`, labeled `Ideal`;
- measured line: calculated speedup, labeled `Measured`;
- markers at every tested rank count;
- readable title, legend, and grid.

### `speedup_log_x.png`

Create this additional plot only when the user supplies `--log-x`. It must show
the same ideal and measured data as `speedup.png`, but use a logarithmic x-axis.
Keep the y-axis linear. Always keep the normal linear `speedup.png` as the main
result.

### `run_logs/`

Create a `run_logs/` directory inside the timestamped results directory. Save
one log file for every MPI launch, including successful and failed runs. Use
clear names such as:

```text
run_logs/cores_1_repeat_1.log
run_logs/cores_2_repeat_1.log
run_logs/cores_4_repeat_3.log
```

Each log must contain:

- the exact MPI command;
- MPI rank count and repetition number;
- captured standard output;
- captured standard error;
- process exit code;
- measured runtime in seconds;
- final success or failure status.

Write the log immediately after each launch so diagnostics remain available if
the full benchmark is interrupted.

## Console output

While running, print concise progress such as:

```text
Running 2 MPI ranks, repetition 3/5...
Completed in 11.18 seconds.
```

At the end, print a readable summary table containing:

- MPI ranks;
- average runtime;
- speedup;
- parallel efficiency.

Also print the absolute path of the timestamped results directory.

## Validation

Create all visible validation code in
`performance_evaluation/unit tests/test_performance_strong_scaling.py`. Do not
place test logic only in temporary files. Add focused tests for:

- average-runtime calculation;
- speedup calculation;
- efficiency calculation;
- CSV contents and column order;
- rejection of invalid `--cores` and `--repetitions` values;
- rejection of a missing `--workflow` and acceptance of a valid workflow;
- creation of both linear and optional log-x plots from synthetic data.

Run a lightweight smoke test with `--cores 1 2 --repetitions 1` if the local
machine has MPI and the workflow dependencies. Do not run `1, 2, 4, 8` with five
repetitions unless resources and time are clearly sufficient. Never claim a
test passed unless it was actually executed.

## Acceptance criteria

The task is complete when:

- one command automatically runs the full sequence, calculates metrics, and
  creates the result files;
- `--workflow` can select another compatible MPI Python workflow without
  modifying the benchmark source;
- rank counts and repetitions are configurable without source edits;
- different rank-count experiments run one at a time;
- every individual runtime is preserved in `raw_runtime.csv`;
- every MPI launch has a corresponding file in `run_logs/`;
- summary metrics use average runtime;
- `speedup.png` contains both ideal and measured lines;
- `--log-x` optionally creates `speedup_log_x.png`;
- all generated work remains inside `performance_evaluation/`;
- available tests pass and unavailable MPI validation is reported honestly.
