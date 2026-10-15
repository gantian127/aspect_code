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

1. Runtime for each MPI-rank/global-grid-shape pair.
2. Weak-scaling efficiency: `E(n) = T(reference) / T(n)`.
3. A runtime plot comparing measured runtime with ideal constant runtime.

Ideal weak scaling has constant runtime and efficiency `1.0`.

## Scope

All implementation work for this task must remain inside
`performance_evaluation/`.

- Add `performance_evaluation/performance_weak_scaling.py`.
- Make only the minimal changes required to the two example workflows so they
  can receive a global grid shape at runtime.
- Preserve each workflow's existing behavior when it is run without the new
  grid-shape argument.
- Add tests and supporting files only inside `performance_evaluation/`.
- Save generated benchmark artifacts under
  `performance_evaluation/weak_scaling_results/` by default.
- Do not implement an automatic grid-size calibration utility as part of this
  task.

Treat a selected workflow as a black-box MPI executable except for the common
global-grid-shape interface described below. Do not hard-code workflow imports,
model types, scientific components, result-file names, or workflow-specific
branches into the benchmark driver.

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

## Runtime measurement

Use the same workflow-launch and runtime-measurement approach as
`performance_strong_scaling.py`. Apply it consistently to every rank-shape pair
and every repeated run. Failed runs must not be included in calculated results.

## Repeat filtering and average runtime

Retain the runtime of every individual workflow launch. For every rank-shape
pair, independently check for clearly anomalous runtime measurements. Do not
automatically discard one run merely because it is farthest from the median. If
the three measurements are reasonably consistent, retain all three and average
all three.

Use the same robust modified-z-score check as the strong-scaling benchmark. For
the successful runtimes of one rank-shape pair:

```text
median = median(runtime)
MAD = median(abs(runtime - median))
modified_z = 0.67448975 * abs(runtime - median) / MAD
```

Exclude a runtime only when `modified_z > 3.5`. Do not filter automatically when
MAD is zero or when filtering would leave fewer than two measurements. This
means a rank-shape pair may have zero excluded outliers, which is expected when
its measurements are stable.

Calculate the arithmetic mean of every successful runtime retained after this
conditional check. Use that average for weak-scaling efficiency and plotting.
Never delete an excluded raw measurement; preserve it in the raw data and mark
it as excluded.

Require at least one successful run for every rank-shape pair. Stop with a
clear diagnostic only when a rank-shape pair has no successful runs. Outlier
filtering is applied only when at least three successful measurements are
available; with one or two measurements, retain all successful runtimes.

## Weak-scaling calculations

The first user-supplied rank-shape pair is the reference case. It does not have
to use one MPI rank, although the default reference does.

Let `T(reference)` be the post-filter average runtime for the first pair and
`T(n)` be the post-filter average runtime for another pair:

```text
weak_scaling_efficiency(n) = T(reference) / T(n)
ideal_runtime(n) = T(reference)
```

The reference efficiency must be `1.0`. Ideal weak-scaling efficiency is `1.0`
for every rank count. Calculate metrics only after the complete campaign has
finished.

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
workflow,ranks,global_shape,repetition,runtime_s,status,excluded
```

Keep successful, failed, retained, and excluded launches auditable.

### `weak_scaling_summary.csv`

Write one row per rank-shape pair. Include at least:

```text
ranks,global_shape,successful_runs,excluded_outliers,average_runtime_s,ideal_runtime_s,weak_scaling_efficiency
```

The `average_runtime_s` value is the mean of all successful measurements retained
after conditional outlier filtering. Write efficiency as a fraction, with ideal
efficiency `1.0`.

### `runtime.png`

Create the primary runtime plot with:

- x-axis: number of processors / MPI ranks;
- y-axis: runtime in seconds;
- ideal line: horizontal constant runtime equal to the reference runtime,
  labeled `Ideal`;
- observed line: post-filter average runtime for each rank-shape pair, labeled
  `Observed`;
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

Follow the strong-scaling log contents and write each log immediately after its
launch.

## Console output

Print concise progress during execution. At completion, print a summary table
containing at least:

- MPI ranks;
- global grid shape;
- raw runtimes or a clear reference to where they were saved;
- number of excluded outliers;
- post-filter average runtime;
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
must be usable without modifying `performance_weak_scaling.py`.

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
- modified-z-score/MAD outlier detection;
- stable measurements for which no runtime is excluded;
- an obvious outlier that is excluded from the average but preserved in the raw
  results;
- MAD equal to zero without automatic exclusion;
- averaging all runtimes retained after the conditional check;
- reference efficiency equal to `1.0`;
- efficiency and ideal-runtime calculations for higher rank counts;
- CSV contents and column order;
- creation of the runtime plot from synthetic data;
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
  their prior standalone behavior;
- every individual runtime and run log is preserved;
- only statistically clear runtime outliers are excluded, stable groups retain
  all measurements, and all retained runtimes are averaged;
- `weak_scaling_summary.csv` contains runtime, ideal runtime, and efficiency;
- `runtime.png` compares observed runtime with horizontal ideal runtime;
- results are written under a new timestamped directory in
  `performance_evaluation/weak_scaling_results/`;
- no time-step control or grid calibration is added;
- focused tests pass and unavailable MPI validation is reported honestly.
