# Codex Instruction: Weak-Scaling Evaluation

## Objective

Create a reusable Python weak-scaling benchmark driver for MPI-based Python
workflows. Use the following workflows as the first real validation cases:

- `performance_evaluation/raster_lem_example.py`
- `performance_evaluation/voronoi_lem_workflow.py`

Do not hard-code the benchmark driver to either workflow. The user must select a
compatible workflow with the required `--workflow` option.

Implement the driver in:

```text
performance_evaluation/performance_weak_scaling.py
```

Use `performance_evaluation/performance_strong_scaling.py` as the design
reference for argument validation, sequential MPI launches, timing, result
storage, run logs, console progress, plotting style, and error handling.

In a weak-scaling experiment, increase the global problem size approximately in
proportion to the MPI rank count so that the average workload per rank remains
approximately constant:

```text
global_node_count / mpi_ranks ~= constant
```

Measure and report:

1. Average model-loop runtime for each MPI-rank/global-grid-shape pair.
2. Weak-scaling efficiency: `E(n) = T(reference) / T(n)`.
3. A runtime plot comparing measured model-loop runtime with ideal constant
   model-loop runtime.

`T(n)` is the post-filter average `model_loop_s` for the rank-shape pair using
`n` MPI ranks. Ideal weak scaling has constant model-loop runtime and efficiency
`1.0`. Do not use the complete workflow runtime or the outer `mpiexec` command
runtime to calculate `T(n)`, weak-scaling efficiency, or the observed plot line.

## Scope

All implementation work for this task must remain inside
`performance_evaluation/`.

- Add `performance_evaluation/performance_weak_scaling.py`.
- Make only the minimal changes required for the two example workflows to
  receive a global grid shape at runtime. Do not alter their existing
  scientific calculations.
- Treat the timing-output contract as a compatibility requirement for a
  selected workflow. Do not rewrite workflow timing instrumentation as part of
  the benchmark-driver implementation unless that work is separately requested.
- Preserve each workflow's existing behavior when it is run without the new
  grid-shape argument.
- Add tests and supporting files only inside `performance_evaluation/`.
- Save generated benchmark artifacts under
  `performance_evaluation/weak_scaling_results/` by default.
- Do not implement an automatic grid-size calibration utility as part of this
  task.

Treat a selected workflow as a black-box MPI executable except for the common
global-grid-shape interface and timing-output contract described below. Do not
hard-code workflow imports, model types, scientific components, result-file
names, timing CSV filenames, or workflow-specific branches into the benchmark
driver. Keep both Raster and Voronoi as supported examples; do not remove either
example merely because one cannot be exercised in the current environment.

## Default weak-scaling configuration

Use these defaults when the user does not override ranks, shapes, or repeats:

```text
MPI ranks:   1 4 16 64
Grid shapes: 250,250 500,500 1000,1000 2000,2000
Repeats:     3
```

The rank and shape lists are paired by position:

```text
1 rank   -> 250,250
4 ranks  -> 500,500
16 ranks -> 1000,1000
64 ranks -> 2000,2000
```

These defaults keep the nominal global node count proportional to the rank
count. They are defaults, not restrictions, and must not be embedded in logic
that prevents other values.

## Command-line interface

The benchmark must run the complete sequence, calculate metrics, and generate
results with one command. The workflow is always required:

```bash
python performance_evaluation/performance_weak_scaling.py \
  --workflow performance_evaluation/raster_lem_example.py
```

The preceding command uses the default ranks, grid shapes, and repeat count.

A user must be able to override them without editing source code:

```bash
python performance_evaluation/performance_weak_scaling.py \
  --workflow performance_evaluation/voronoi_lem_workflow.py \
  --ranks 1 4 16 64 \
  --grid-shapes 250,250 500,500 1000,1000 2000,2000 \
  --repeats 3
```

Support at least these options:

- `--workflow PATH`: required MPI Python workflow; do not provide a default;
- `--ranks N [N ...]`: ordered MPI rank counts, default `1 4 16 64`;
- `--grid-shapes SHAPE [SHAPE ...]`: ordered global grid shapes, default
  `250,250 500,500 1000,1000 2000,2000`;
- `--repeats N`: number of independent runs for every rank-shape pair, default
  `3` and minimum `1`;
- `--mpi-launcher PATH`: MPI launcher name or path, default `mpiexec`;
- `--results-dir PATH`: optional parent output directory inside
  `performance_evaluation/`;
- `--workflow-args ARG ...`: optional remaining arguments forwarded unchanged
  to every workflow launch; this must be the final benchmark option.

If the existing strong-scaling implementation provides another generally useful
non-scientific option, it may be reused when doing so does not conflict with
this instruction.

Running the benchmark without `--workflow` must fail through the argument
parser with a clear nonzero exit status. Validate that the workflow exists, is a
Python file, and is inside `performance_evaluation/`, consistently with the
strong-scaling driver.

### Rank and shape parsing

The two dimensions of one shape use an ASCII comma with no internal whitespace:

```text
250,250
```

Separate multiple shapes with spaces:

```text
--grid-shapes 250,250 500,500 1000,1000
```

Do not use `250x250` as the documented input format. Reject malformed shapes,
empty dimensions, and zero or negative dimensions with clear messages.

The number of values passed to `--ranks` must equal the number passed to
`--grid-shapes`. Pair the lists by position; do not run their Cartesian product.
Reject duplicate, zero, or negative rank counts. Require ranks to be supplied in
strictly increasing order so the first pair is an unambiguous reference case.

Accept rectangular grids. Square grids or a fixed aspect ratio are recommended
for controlled comparisons, but the program must not require them.

For every pair, calculate the nominal workload as the product of its global
shape dimensions divided by its rank count. If the values are not approximately
constant, print a warning but allow the explicitly requested experiment to run.

## Common workflow grid-shape interface

Every tested shape is the **global grid shape**, not a local subgrid shape.

Define a common single-run workflow interface conceptually equivalent to:

```text
--grid-shape ROWS,COLUMNS
```

The benchmark's plural `--grid-shapes` option describes the entire campaign.
For one MPI launch, pass only that pair's single global shape to the workflow.
Append any fixed `--workflow-args` without changing their content or order.

Make the smallest practical changes to the example workflows:

- `raster_lem_example.py` must be able to use an explicitly supplied two-value
  global shape rather than deriving the only supported shape from one scalar;
- `voronoi_lem_workflow.py` must be able to replace its hard-coded global
  `grid_shape` with the supplied shape;
- when the new shape input is absent, preserve each workflow's pre-existing
  default or standalone invocation behavior;
- do not change time steps, physical parameters, active components, random
  behavior, or numerical calculations merely to support weak scaling.

If inspection of the existing scripts shows that a slightly different common
interface integrates more cleanly, it may be used, but it must remain documented,
workflow-agnostic, and capable of forwarding an explicit global two-dimensional
shape to both example workflows.

Do not add a time-step option to `performance_weak_scaling.py`. Each workflow
continues to use its own existing time-step configuration, unchanged across all
rank-shape pairs and repeats.

## Common workflow timing-output contract

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

Thus one complete MPI workflow launch produces one timing row, not one row per
rank. Taking the maximum across ranks represents the elapsed time of each
synchronized parallel phase and prevents a faster rank from understating the
launch runtime.

The benchmark must fail that individual launch with a clear diagnostic if this
contract is not satisfied. It must never silently substitute the complete
workflow time or the outer command time for a missing `model_loop_s` value.

## Experiment execution

Run rank-shape experiments sequentially, never concurrently. Complete all
repeats for one pair before moving to the next pair. With the defaults, use this
order:

```text
1 rank,   shape 250,250   -> all repeats
4 ranks,  shape 500,500   -> all repeats
16 ranks, shape 1000,1000 -> all repeats
64 ranks, shape 2000,2000 -> all repeats
```

Each individual MPI launch uses its requested ranks concurrently, but separate
benchmark launches must not overlap and compete for resources.

Use the same workflow arguments and scientific settings for every launch except
for the explicitly paired MPI rank count and global grid shape. Do not use
`--oversubscribe` for performance measurements.

For Raster, launch commands equivalent to:

```bash
mpiexec -n 1 python performance_evaluation/raster_lem_example.py --grid-shape 250,250 --timing-output-dir <RUN_DIR>/workflow_timings/ranks_1_repeat_1
mpiexec -n 4 python performance_evaluation/raster_lem_example.py --grid-shape 500,500 --timing-output-dir <RUN_DIR>/workflow_timings/ranks_4_repeat_1
mpiexec -n 16 python performance_evaluation/raster_lem_example.py --grid-shape 1000,1000 --timing-output-dir <RUN_DIR>/workflow_timings/ranks_16_repeat_1
mpiexec -n 64 python performance_evaluation/raster_lem_example.py --grid-shape 2000,2000 --timing-output-dir <RUN_DIR>/workflow_timings/ranks_64_repeat_1
```

For Voronoi, launch commands equivalent to:

```bash
mpiexec -n 1 python performance_evaluation/voronoi_lem_workflow.py --grid-shape 250,250 --timing-output-dir <RUN_DIR>/workflow_timings/ranks_1_repeat_1
mpiexec -n 4 python performance_evaluation/voronoi_lem_workflow.py --grid-shape 500,500 --timing-output-dir <RUN_DIR>/workflow_timings/ranks_4_repeat_1
mpiexec -n 16 python performance_evaluation/voronoi_lem_workflow.py --grid-shape 1000,1000 --timing-output-dir <RUN_DIR>/workflow_timings/ranks_16_repeat_1
mpiexec -n 64 python performance_evaluation/voronoi_lem_workflow.py --grid-shape 2000,2000 --timing-output-dir <RUN_DIR>/workflow_timings/ranks_64_repeat_1
```

These are example commands, not workflow-specific driver branches. Fixed
arguments supplied through `--workflow-args` must appear after the
driver-managed grid-shape and timing-output arguments.

## Runtime collection

Before every MPI launch, create a new empty launch-specific directory under the
timestamped benchmark directory, for example:

```text
workflow_timings/ranks_4_repeat_2/
```

Append these driver-managed arguments to the selected workflow command:

```text
--grid-shape ROWS,COLUMNS
--timing-output-dir <absolute launch-specific directory>
```

Reject `--grid-shape` or `--timing-output-dir` inside user-supplied
`--workflow-args`; the driver must own these values so the current weak-scaling
pair is unambiguous and one launch cannot overwrite another launch's timing
file.

The benchmark may also measure the elapsed wall-clock time of the complete MPI
command with `time.perf_counter()`, but this is only the diagnostic
`command_runtime_s`. It must not be used for outlier filtering, averages,
weak-scaling efficiency, or scaling plots.

After a command exits successfully, require exactly one CSV in its
launch-specific timing directory. Parse its single row and validate that:

- all required timing columns are present;
- `mpi_ranks` equals the rank count requested by the benchmark;
- `grid_shape` equals the global shape requested for that launch after format
  normalization;
- `time_steps` remains fixed across the complete weak-scaling campaign;
- `model_loop_s` and `total_workflow_run_s` are finite positive numbers;
- setup times are finite non-negative numbers.

Use `model_loop_s` as the runtime measurement for that launch. Preserve
`global_grid_setup_s`, `local_grid_setup_s`, `total_workflow_run_s`, and
`command_runtime_s` as diagnostic values only.

For every launch:

- wait for it to finish before starting the next launch;
- record its rank count, global shape, repetition number, model-loop runtime,
  and diagnostic times in seconds;
- check the process exit status;
- treat a missing, malformed, empty, or ambiguous timing CSV as a failed run;
- do not include failed runs in the model-loop average;
- stop with a clear error if a rank-shape pair has no successful measurements.

Keep every raw model-loop runtime even when a measurement is later excluded
from its rank-shape pair's average. Outlier exclusion affects summary
calculations only and must never delete or overwrite the original measurement,
workflow timing CSV, or run log.

## Repeat filtering and average model-loop runtime

Retain the `model_loop_s` value of every individual workflow launch. For every
rank-shape pair, independently check for clearly anomalous model-loop
measurements. Do not automatically discard one run merely because it is
farthest from the median. If the three measurements are reasonably consistent,
retain all three and average all three.

Use the same robust modified-z-score check as the strong-scaling benchmark. For
the successful `model_loop_s` measurements of one rank-shape pair:

```text
median = median(model_loop_s)
MAD = median(abs(model_loop_s - median))
modified_z = 0.67448975 * abs(model_loop_s - median) / MAD
```

Exclude a model-loop measurement only when `modified_z > 3.5`. Do not filter
automatically when MAD is zero or when filtering would leave fewer than two
measurements. This means a rank-shape pair may have zero excluded outliers,
which is expected when its measurements are stable.

Calculate the arithmetic mean of every successful `model_loop_s` retained after
this conditional check. Use that average for weak-scaling efficiency and
plotting. Never delete an excluded raw measurement; preserve it in the raw data
and mark it as excluded. Also calculate the standard deviation of retained
`model_loop_s` values so run-to-run variability remains visible.

Require at least one successful run for every rank-shape pair. Stop with a
clear diagnostic only when a rank-shape pair has no successful runs. Outlier
filtering is applied only when at least three successful measurements are
available; with one or two measurements, retain all successful model-loop
measurements.

## Weak-scaling calculations

The first user-supplied rank-shape pair is the reference case. It does not have
to use one MPI rank, although the default reference does.

Let `T_model(reference)` be the post-filter average `model_loop_s` for the first
pair and `T_model(n)` be the post-filter average `model_loop_s` for another pair:

```text
weak_scaling_efficiency(n) = T_model(reference) / T_model(n)
ideal model-loop runtime(n) = T_model(reference)
```

The reference efficiency must be `1.0`. Ideal weak-scaling efficiency is `1.0`
for every rank count. Calculate metrics only after the complete campaign has
finished. Setup, complete-workflow, and command runtimes are diagnostics and
must not be combined with `model_loop_s`.

## PyMetis partitioning

PyMetis controls graph partitioning, so do not assume that every subgrid has the
same node count. For this task, do not:

- control the number of nodes assigned to each subgrid;
- collect per-rank local-node counts;
- calculate a load-imbalance metric;
- calculate edge cut;
- correct runtime or efficiency for partition imbalance.

Treat partition imbalance as part of the observed workflow behavior.

## Result storage

The default result parent is:

```text
performance_evaluation/weak_scaling_results/
```

Mirror the strong-scaling result organization by creating one timestamped
directory per complete campaign:

```text
performance_evaluation/weak_scaling_results/run_YYYY-MM-DD_HHMMSS/
```

Never overwrite an earlier campaign. Store at least the following artifacts in
the timestamped directory.

### `raw_runtime.csv`

Write one row per individual workflow launch. Include at least:

```text
workflow,ranks,global_shape,time_steps,repetition,global_grid_setup_s,local_grid_setup_s,model_loop_s,total_workflow_run_s,command_runtime_s,status,excluded
```

The field used as the scaling runtime is explicitly `model_loop_s`.
`total_workflow_run_s` and `command_runtime_s` must not be aliases for it. Keep
successful, failed, retained, and excluded launches auditable; unavailable
timing fields for a failed launch may be blank.

### `weak_scaling_summary.csv`

Write one row per rank-shape pair. Include at least:

```text
ranks,global_shape,successful_runs,excluded_outliers,average_model_loop_s,std_model_loop_s,weak_scaling_efficiency
```

The `average_model_loop_s` value is the mean of all successful `model_loop_s`
measurements retained after conditional outlier filtering. Write efficiency as
a fraction, with ideal efficiency `1.0`. Do not store a repeated
`ideal_model_loop_s` column: the ideal plot line is derived directly from the
reference row's `average_model_loop_s`.

### `workflow_timings/`

Preserve the one-row timing CSV produced by every successful workflow launch
inside its launch-specific directory. Do not merge, ambiguously rename, or
delete these source timing records after copying their values into
`raw_runtime.csv`.

### `runtime.png`

Create the primary runtime plot with:

- x-axis: number of processors / MPI ranks;
- y-axis: model-loop runtime in seconds;
- ideal line: horizontal constant model-loop runtime equal to the reference
  pair's average `model_loop_s`, labeled `Ideal`;
- observed line: post-filter average `model_loop_s` for each rank-shape pair,
  labeled `Observed`;
- a marker at every tested rank count;
- x-axis tick labels showing the actual supplied rank counts;
- a readable title, legend, and grid.

Use linear axes for the required primary plot. An optional logarithmic x-axis
plot may follow the established strong-scaling pattern, but it must not replace
`runtime.png`.

### `run_logs/`

Create one log for every MPI launch, including failed and excluded runs. Use
clear names that identify both rank count and repetition, such as:

```text
run_logs/ranks_1_repeat_1.log
run_logs/ranks_4_repeat_3.log
```

Each log must contain the exact MPI command, forwarded workflow arguments, rank
count, global shape, repetition number, captured standard output and error,
process exit code, parsed `model_loop_s`, diagnostic setup/workflow/command
times when available, absolute workflow timing CSV path when available, and
final status. Write each log immediately after its launch.

## Console output

Print concise progress during execution. At completion, print a summary table
containing at least:

- MPI ranks;
- global grid shape;
- raw model-loop runtimes or a clear reference to where they were saved;
- number of excluded model-loop outliers;
- post-filter average model-loop runtime;
- weak-scaling efficiency.

Clearly identify the first pair as the reference and print the absolute path of
the timestamped results directory.

## Generality requirements

The implementation must not:

- select a default workflow;
- branch on `raster_lem_example.py` or `voronoi_lem_workflow.py` filenames;
- assume that ranks are limited to `1 4 16 64`;
- assume that shapes are limited to the default values;
- require square shapes;
- interpret ranks and shapes as a Cartesian product;
- assume equal PyMetis subgrid node counts;
- change workflow time steps;
- automatically derive shapes when the user supplied explicit ones.

Any MPI Python workflow conforming to the documented global-grid-shape interface
and timing-output contract must be usable without modifying
`performance_weak_scaling.py`.

## Validation

Add focused tests under `performance_evaluation/unit tests/`. Cover at least:

- rejection of a missing `--workflow`;
- acceptance of a valid workflow;
- parsing `250,250` as one two-dimensional shape;
- parsing multiple comma-separated shapes;
- rejection of malformed, zero, and negative shape dimensions;
- rejection of duplicate, zero, negative, or non-increasing ranks;
- rejection when the rank and shape list lengths differ;
- confirmation that ranks and shapes pair by position rather than by Cartesian
  product;
- default values for ranks, shapes, and repeats;
- forwarding the current global shape and fixed workflow arguments in the MPI
  command;
- appending the driver-managed `--timing-output-dir` and rejecting conflicting
  user-supplied grid-shape or timing-output arguments;
- parsing a valid one-row workflow timing CSV;
- rejection of missing, malformed, empty, multi-row, and multiple timing CSVs;
- rejection of a timing row whose `mpi_ranks` or normalized `grid_shape` does
  not match the requested launch;
- rejection of changed `time_steps` during one campaign;
- confirmation that calculations use `model_loop_s`, not
  `total_workflow_run_s` or `command_runtime_s`;
- modified-z-score/MAD outlier detection on `model_loop_s`;
- stable measurements for which no model-loop runtime is excluded;
- an obvious outlier that is excluded from the average but preserved in the raw
  results;
- MAD equal to zero without automatic exclusion;
- averaging all `model_loop_s` values retained after the conditional check;
- reference efficiency equal to `1.0`;
- efficiency and ideal model-loop plot-line calculations for higher rank counts;
- CSV contents and column order;
- creation of the model-loop runtime plot from synthetic data;
- preservation of original workflow behavior when no new shape is passed.

Do not run the large default campaign in automated tests. Use mock workflows,
synthetic timing data, or very small grids. Run a lightweight MPI smoke test
only when MPI and workflow dependencies are available. Never claim that a test
passed unless it was actually executed.

## Out of scope

Do not implement these features unless separately requested:

- automatic baseline-shape calibration;
- automatic shape generation from rank counts;
- workflow time-step control;
- PyMetis partition-quality analysis;
- local workload-imbalance correction;
- scheduler job submission;
- absolute performance comparisons between different workflows.

## Acceptance criteria

The task is complete when:

- `--workflow` is required and no default workflow is selected;
- one command sequentially runs every rank-shape pair and all repeats;
- ranks and global shapes are configurable and paired by position;
- the documented default campaign is `1/4/16/64` ranks with
  `250,250/500,500/1000,1000/2000,2000` global shapes;
- both example workflows accept the common global-shape input while preserving
  their prior standalone behavior, and neither example is removed;
- every individual `model_loop_s`, workflow timing CSV, and run log is
  preserved;
- only statistically clear model-loop outliers are excluded, stable groups
  retain all measurements, and all retained model-loop runtimes are averaged;
- `weak_scaling_summary.csv` contains average model-loop runtime, variability,
  and efficiency without a redundant ideal-runtime column;
- `runtime.png` compares observed model-loop runtime with horizontal ideal
  model-loop runtime;
- setup time, total workflow time, and outer command time remain diagnostic and
  never affect outlier filtering, averages, efficiency, or the plot;
- results are written under a new timestamped directory in
  `performance_evaluation/weak_scaling_results/`;
- no time-step control or grid calibration is added;
- focused tests pass and unavailable MPI validation is reported honestly.
