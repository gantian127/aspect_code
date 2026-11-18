# this is the plotting code for strong scaling results

import os

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt

# setup info
result_dir= '/Users/tiga7385/Desktop/aspect_code/performance_evaluation/performance_summary'
csv_file_name = 'voronoi_results.csv'
grid_type = 'Voronoi'
plot_control= {
    "workflow_runtime": True,
    "scaling":True,
    "speedup":True
}

# load csv data
df = pd.read_csv(os.path.join(result_dir, csv_file_name))
df.columns = df.columns.str.strip()
df = df.sort_values("mpi_ranks")

ranks = df["mpi_ranks"].to_numpy()
time = df["model_loop_s"].to_numpy()  # 如果列名不同，在这里修改

# plot 0: time-rank for each stage, log10-log2 scale
# Plot runtime for each stage
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    ranks, df["global_grid_setup_s"],
    "o-", color="#8c6bb1", label="Global grid setup",
    linewidth=2.5, markersize=8,
)
ax.plot(
    ranks, df["local_grid_setup_s"],
    "o-", color="#169c95", label="Local grid setup",
    linewidth=2.5, markersize=8,
)
ax.plot(
    ranks, time,
    "o-", color="#2775b6", label="Model loop",
    linewidth=2.5, markersize=8,
)
ax.plot(
    ranks, df["total_workflow_run_s"],
    "o-", color="#df7733", label="Total workflow",
    linewidth=2.5, markersize=8,
)

# Log scales with ordinary rank labels
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(ranks, labels=[str(int(p)) for p in ranks])

ax.set_xlabel("MPI ranks", fontsize=14)
ax.set_ylabel("Wall-clock time (s, log scale)", fontsize=14)
ax.set_title(
    f"Runtime of the {grid_type} workflow\n"
    "Shape = 2000 × 2000, Time steps = 200",
    fontsize=14,
)

ax.tick_params(axis="both", labelsize=12)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(True, which="major", alpha=0.2)
ax.set_axisbelow(True)
ax.legend(loc="lower left", frameon=False, fontsize=12)

fig.tight_layout()

output_path = os.path.join(result_dir, f"{grid_type}_workflow_runtime.png")
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)


# plot 1: time-rank scaling for model-loop, log10-log2 scale
ideal_time = time[0] * ranks[0] / ranks

fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title(
    f"Strong scaling for {grid_type} workflow\n"
    f"Shape = 2000 × 2000, Time steps = 200",
    fontsize=14,
)

ax.plot(
    ranks, time,
    "o-", color="navy", linewidth=2, markersize=8,
    label="Model loop",
)
ax.plot(
    ranks, ideal_time,
    "--", color="gray", linewidth=2,
    label="Ideal scaling",
)

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(ranks, labels=[str(int(p)) for p in ranks])
ax.set_xlabel("Number of ranks")
ax.set_ylabel("Time (s)")

ax.legend()
ax.grid(True, which="major", linestyle=":", alpha=0.4)

fig.tight_layout()
fig.savefig(os.path.join(result_dir, f"{grid_type}_scaling.png"), dpi=300)


# plot2: Speedup relative to the smallest rank count
speedup = time[0] / time
ideal_speedup = ranks / ranks[0]

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    ranks, speedup, "o-", color="navy",
    linewidth=2, markersize=8, label="Model loop",
)
ax.plot(
    ranks, ideal_speedup, "--", color="gray",
    linewidth=2, label="Ideal scaling",
)

ax.set_xscale("log", base=2)
ax.set_xticks(ranks, labels=[str(int(p)) for p in ranks])

ax.set_yscale("log", base=2)
ax.set_yticks(speedup, labels=[str(int(p)) for p in speedup])

ax.set_xlabel("Number of ranks")
ax.set_ylabel(f"Speedup relative to {int(ranks[0])} rank(s)")
ax.set_title(
    f"Strong scaling speedup of the {grid_type} model loop\n"
    f"Shape = 2000 × 2000, Time steps = 200",
    fontsize=14,
)
fig.savefig(os.path.join(result_dir, f"{grid_type}_speedup.png"), dpi=300)
plt.close(fig)