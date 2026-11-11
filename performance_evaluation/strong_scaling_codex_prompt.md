# Codex Instruction: Strong-Scaling Evaluation

## Objective

Create a reusable Python strong-scaling benchmark driver for MPI-based Python
workflows. Use `performance_evaluation/voronoi_lem_workflow.py` as the first real
validation case, but do not hard-code the driver to that workflow. The user must
select a compatible workflow with the required `--workflow` option.

For a fixed scientific problem, measure how the model-loop wall-clock time
changes when the number of MPI ranks increases. Calculate:

1. Average model-loop runtime for each MPI rank count.
2. Speedup: `S(n) = T(1) / T(n)`.
3. Parallel efficiency: `E(n) = S(n) / n`.
4. A speedup plot comparing measured scaling with ideal scaling.

`T(n)` is the average `model_loop_s` measured for `n` MPI ranks. `T(1)` must
come from the same MPI workflow executed with one rank. Before calculating any
`T(n)`, apply the per-rank-count outlier check described below. Do not use the
complete workflow runtime or the outer `mpiexec` command runtime to calculate
`T(n)`, speedup, efficiency, or the measured plot line.

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
executable except for the required timing-output contract described below. Do
not hard-code its imports, functions, model type, result-file names, scientific
parameters, timing directory name, or timing CSV filename into the benchmark
driver.

Some workflows require their own positional or optional command-line arguments.
Accept these through `--workflow-args` and append them unchanged after the
workflow path in every MPI command. Use exactly the same workflow arguments for
every rank count and repetition so the scientific problem remains fixed.

Preserve the scientific behavior of the selected workflow. For
`voronoi_lem_workflow.py`, do not change its grid, model parameters, active
components, time steps, or numerical calculation between MPI rank counts. For
any future workflow, use its unchanged default problem for every launch. The
global problem size must remain fixed throughout each strong-scaling experiment.

The workflow, not the outer benchmark process, owns the model-loop timing
boundary. A compatible workflow must:

- accept `--timing-output-dir DIR`;
- create exactly one CSV in that launch-specific directory after a successful
  run;
- write exactly one summary row containing the maximum time across MPI ranks;
- provide these columns:

```text
mpi_ranks,grid_shape,time_steps,global_grid_setup_s,local_grid_setup_s,model_loop_s,total_workflow_run_s
```

The benchmark must fail that individual launch with a clear diagnostic if this
contract is not satisfied. It must never silently substitute the complete
workflow time or the outer command time for a missing `model_loop_s` value.

## Experiment execution

The benchmark script must run different MPI rank-count experiments sequentially,
not concurrently. For the default configuration, use this order:

```text
1 rank -> all repetitions
2 ranks -> all repetitions
4 ranks -> all repetitions
```

Each individual MPI launch uses its requested ranks concurrently, but two
different rank-count experiments must never overlap. Concurrent experiments
would compete for CPU, memory bandwidth, and I/O and invalidate the timing.

For the default workflow, launch commands equivalent to:

```bash
mpiexec -n 1 python performance_evaluation/voronoi_lem_workflow.py --timing-output-dir <RUN_DIR>/workflow_timings/cores_1_repeat_1
mpiexec -n 2 python performance_evaluation/voronoi_lem_workflow.py --timing-output-dir <RUN_DIR>/workflow_timings/cores_2_repeat_1
mpiexec -n 4 python performance_evaluation/voronoi_lem_workflow.py --timing-output-dir <RUN_DIR>/workflow_timings/cores_4_repeat_1
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

- MPI ranks: `1 2 4`;
- repetitions per rank count: `3`;
- MPI launcher: `mpiexec`;
- linear plot axes.

Users must be able to change MPI ranks without editing Python source:

```bash
python performance_evaluation/performance_strong_scaling.py \
  --workflow performance_evaluation/voronoi_lem_workflow.py \
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

For a workflow with its own arguments, put `--workflow-args` last. Everything
after it belongs to the workflow, even when an argument begins with `--`:

```bash
python performance_evaluation/performance_strong_scaling.py \
  --workflow performance_evaluation/raster_lem_example.py \
  --cores 1 2 4 \
  --repetitions 3 \
  --mpi-launcher mpiexec \
  --log-x \
  --results-dir performance_evaluation/strong_scaling_results_raster \
  --workflow-args --mode d4 --seed 1
```

This must launch commands equivalent to:

```bash
mpiexec -n 1 python performance_evaluation/raster_lem_example.py --grid-shape 300,300 --timing-output-dir <RUN_DIR>/workflow_timings/cores_1_repeat_1 --mode d4 --seed 1
mpiexec -n 2 python performance_evaluation/raster_lem_example.py --grid-shape 300,300 --timing-output-dir <RUN_DIR>/workflow_timings/cores_2_repeat_1 --mode d4 --seed 1
mpiexec -n 4 python performance_evaluation/raster_lem_example.py --grid-shape 300,300 --timing-output-dir <RUN_DIR>/workflow_timings/cores_4_repeat_1 --mode d4 --seed 1
```

Support at least these options:

- `--workflow PATH`: required MPI Python script to benchmark; do not provide a
  default workflow;
- `--cores N [N ...]`: ordered positive MPI rank counts;
- `--repetitions N`: positive number of runs per rank count;
- `--mpi-launcher PATH`: launcher name or path, default `mpiexec`;
- `--log-x`: additionally create a speedup plot with a logarithmic x-axis;
- `--results-dir PATH`: optional parent output directory inside
  `performance_evaluation/`;
- `--workflow-args ARG ...`: optional remaining arguments passed unchanged to
  the selected workflow; this option must be the final benchmark option.

Validate all arguments. Reject duplicate, zero, or negative rank counts and
non-positive repetition counts with clear error messages.

## Runtime collection

For every rank count, run the workflow the requested number of times and retain
every individual model-loop runtime. Do not keep only the average.

Before each MPI launch, create a new empty launch-specific directory under the
timestamped benchmark directory, for example:

```text
workflow_timings/cores_4_repeat_2/
```

Append the following driver-managed arguments to the selected workflow command:

```text
--timing-output-dir <absolute launch-specific directory>
```

Reject `--timing-output-dir` inside user-supplied `--workflow-args`; the driver
must own this argument so one launch cannot overwrite or reuse another launch's
timing file.

The benchmark may also measure the elapsed wall-clock time of the complete MPI
command with `time.perf_counter()`, but this is only the diagnostic
`command_runtime_s`. It must not be used for outlier filtering, averages,
speedup, efficiency, or scaling plots.

After a command exits successfully, require exactly one CSV in its
launch-specific timing directory. Parse the single row and validate that:

- all required timing columns are present;
- `mpi_ranks` equals the rank count requested by the benchmark;
- `grid_shape` and `time_steps` remain fixed across the complete strong-scaling
  experiment;
- `model_loop_s` and `total_workflow_run_s` are finite positive numbers;
- setup times are finite non-negative numbers.

Use `model_loop_s` as the runtime measurement for that launch. Preserve
`global_grid_setup_s`, `local_grid_setup_s`, `total_workflow_run_s`, and
`command_runtime_s` as diagnostic values only.

For each launch:

- wait for it to finish before starting the next launch;
- record rank count, repetition number, model-loop runtime, and diagnostic
  times in seconds;
- check the process exit status;
- treat a missing, malformed, empty, or ambiguous timing CSV as a failed run;
- do not include failed runs in the model-loop average;
- stop with a clear error if a rank count has no successful measurements.

Keep every raw model-loop runtime even when a measurement is later excluded
from its rank count's average. Outlier exclusion affects summary calculations
only and must never delete or overwrite the original measurement, workflow
timing CSV, or run log.

Build the command without invoking a shell. Its order must be:

```text
MPI launcher, -n, rank count, current Python executable, workflow path,
driver-managed timing-output arguments, workflow arguments
```

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
`model_loop_s` measurements.

### Per-rank-count outlier check

Apply automatic outlier filtering independently to the successful
`model_loop_s` measurements for every rank count. At least three successful
repetitions must be available for that rank count. Use the modified z-score
based on the median absolute deviation (MAD):

```text
median(n) = median(model_loop_s(n, i))
MAD(n) = median(abs(model_loop_s(n, i) - median(n)))
modified_z(n, i) = 0.67448975 * abs(model_loop_s(n, i) - median(n)) / MAD(n)
```

Exclude a measurement from its rank count's average when
`modified_z(n, i) > 3.5`. Do not filter automatically when MAD is zero or when
the filter would leave fewer than two measurements for that rank count.

Preserve excluded measurements in `raw_runtime.csv` with their original process
status. Record how many were excluded in `strong_scaling_summary.csv` so the
filter is visible and auditable.

For `r` successful repetitions at rank count `n`:

```text
T(n) = sum(model_loop_s[n, i] for i = 1..r) / r
S(n) = T(1) / T(n)
E(n) = S(n) / n
```

The rank list must contain `1`, because it supplies the baseline. Calculate all
metrics only after the complete sequence of experiments has finished. For every
summary row, `successful_runs` is the number of successful measurements used
after outlier filtering.

Also calculate the `model_loop_s` standard deviation for each rank count so the
user can see run-to-run variability. The required speedup and efficiency values
must still use the average model-loop runtime, as defined above. Setup,
complete-workflow, and command runtimes must remain diagnostic and must not be
combined with `model_loop_s`.

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
workflow,cores,grid_shape,time_steps,repetition,global_grid_setup_s,local_grid_setup_s,model_loop_s,total_workflow_run_s,command_runtime_s,status
voronoi_lem_workflow.py,1,500x500,2000,1,2.30,0.62,20.30,28.10,29.02,success
voronoi_lem_workflow.py,1,500x500,2000,2,2.28,0.61,20.10,27.88,28.81,success
voronoi_lem_workflow.py,2,500x500,2000,1,2.42,0.39,11.20,18.31,19.20,success
```

The field used as the scaling runtime is explicitly `model_loop_s`.
`total_workflow_run_s` and `command_runtime_s` must not be aliases for it.

### `strong_scaling_summary.csv`

One row per MPI rank count:

```text
cores,successful_runs,excluded_outliers,average_model_loop_s,std_model_loop_s,speedup,efficiency
1,4,1,20.27,0.12,1.00,1.00
2,5,0,11.10,0.09,1.83,0.91
4,5,0,6.40,0.08,3.17,0.79
```

Write efficiency as a fraction, where ideal efficiency is `1.0`.

### `workflow_timings/`

Preserve the one-row timing CSV produced by every successful workflow launch
inside its launch-specific directory. Do not merge, rename ambiguously, or
delete these source timing records after copying their values into
`raw_runtime.csv`.

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
- all forwarded workflow arguments as part of that command;
- MPI rank count and repetition number;
- captured standard output;
- captured standard error;
- process exit code;
- parsed `model_loop_s` in seconds, when available;
- diagnostic `global_grid_setup_s`, `local_grid_setup_s`,
  `total_workflow_run_s`, and `command_runtime_s`, when available;
- the absolute path of the workflow timing CSV, when available;
- final success or failure status.

Write the log immediately after each launch so diagnostics remain available if
the full benchmark is interrupted.

## Console output

While running, print concise progress such as:

```text
Running 2 MPI ranks, repetition 3/5...
Completed; model loop 11.18 seconds.
```

At the end, print a readable summary table containing:

- MPI ranks;
- number of successful measurements used;
- number of excluded model-loop outliers;
- average model-loop runtime;
- speedup;
- parallel efficiency.

Also print the absolute path of the timestamped results directory.

## Validation

Create all visible validation code in
`performance_evaluation/unit tests/test_performance_strong_scaling.py`. Do not
place test logic only in temporary files. Add focused tests for:

- average-model-loop-runtime calculation;
- parsing a valid one-row workflow timing CSV;
- rejection of missing, malformed, empty, multi-row, and multiple timing CSVs;
- rejection of a timing row whose `mpi_ranks` does not match the launch;
- rejection of changed `grid_shape` or `time_steps` during one strong-scaling
  experiment;
- confirmation that scaling calculations use `model_loop_s`, not
  `total_workflow_run_s` or `command_runtime_s`;
- per-rank-count MAD outlier detection on `model_loop_s`, including stable data
  that must not be filtered;
- confirmation that outliers can be filtered for both the one-rank baseline and
  higher rank counts;
- speedup calculation;
- efficiency calculation;
- CSV contents and column order;
- rejection of invalid `--cores` and `--repetitions` values;
- rejection of a missing `--workflow` and acceptance of a valid workflow;
- parsing named `--workflow-args` and preserving their order;
- appending workflow arguments to the generated MPI command;
- appending the driver-managed `--timing-output-dir` arguments and rejecting a
  conflicting user-supplied timing directory;
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
- `--workflow-args` can pass a fixed workflow configuration without modifying
  either Python file;
- rank counts and repetitions are configurable without source edits;
- different rank-count experiments run one at a time;
- every individual `model_loop_s` measurement is preserved in
  `raw_runtime.csv`;
- every successful workflow launch preserves its original one-row timing CSV;
- clearly anomalous measurements at every rank count are excluded only from
  summary calculations and counted in `excluded_outliers`;
- every MPI launch has a corresponding file in `run_logs/`;
- summary metrics use average `model_loop_s` and never the complete workflow or
  outer command runtime;
- `speedup.png` contains both ideal and measured lines;
- `--log-x` optionally creates `speedup_log_x.png`;
- all generated work remains inside `performance_evaluation/`;
- available tests pass and unavailable MPI validation is reported honestly.
