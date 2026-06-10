"""
This is a quick mass balance check for the final results of workflow

method:
run rank = 1
run rank = n
run the code below

this needs to export the final results from global grid as .npy file
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("agg")

rank = 5
time = 1
shape = (1000, 1000)
spacing = 10
folder_name = "linear_diffusion"


true_file_dir = os.path.join(
    os.getcwd(), f"{folder_name}_output_png_1_{shape[0]}by{shape[1]}_time{time}"
)
sim_file_dir = os.path.join(
    os.getcwd(), f"{folder_name}_output_png_{rank}_{shape[0]}by{shape[1]}_time{time}"
)

## check elevation difference
init_elev = np.load(os.path.join(sim_file_dir, "elevation_init.npy"))
simulation_elev = np.load(os.path.join(sim_file_dir, f"elevation_result_{rank}.npy"))
true_elev = np.load(os.path.join(true_file_dir, "elevation_result_1.npy"))
diff = simulation_elev - true_elev

# partition_array = np.load(os.path.join(sim_file_dir, f"partition_array.npy"))

print(f"rank {rank} == rank {1}: {np.all(simulation_elev == true_elev)}")
print(f"diff == 0: {np.all(diff == 0)}")

print(
    f"diff min: {diff.min()}, diff max: {diff.max()}, diff mean: {np.mean(diff)}, diff linalg.norm: {np.linalg.norm(diff)}"
)
print(f"diff abs sum: {np.sum(np.abs(diff))}")

idx = np.argsort(np.abs(diff))[-20:]
for i in idx:
    print(i, diff[i])

# find the max diff value point
max_idx = np.argmax(np.abs(diff))
max_val = diff[max_idx]
row, col = np.unravel_index(max_idx, shape)
print("max diff node:", max_idx)
print("max diff value:", max_val)
print("max diff x, y:", [row, col])
# print(f"max diff in rank: {partition_array[max_idx]}")
# print(f"total ranks: {np.unique(partition_array)}")

# make visualization
# from landlab import HexModelGrid
# mg = HexModelGrid(shape, spacing=spacing, node_layout="rect")
#
# sim = mg.add_field("sim_elev", simulation_elev, at="node")
# true = mg.add_field("true_elev", true_elev, at="node")
# diff = mg.add_field("diff_elev", diff, at="node")
#
# max_x = mg.x_of_node[max_idx]
# max_y = mg.y_of_node[max_idx]
# print("location:", max_x, max_y)
#
# for name, data in zip(["sim","true","diff"],[sim, true, diff]):
#     mg.imshow(data)
#     plt.scatter(max_x, max_y, color="red", s=50, marker="o")
#     plt.title(f"{name} elevation on global grid")
#     plt.savefig(os.path.join(sim_file_dir, f"elevation_{name}_{rank}.png"))
#     plt.close()
#
# print("Done!")

# diff global grid plot
diff_grid = diff.reshape(shape)
true_grid = true_elev.reshape(shape)
sim_grid = simulation_elev.reshape(shape)

plt.imshow(diff_grid, cmap="coolwarm", origin="lower")
plt.scatter(col, row, color="black", s=80)
plt.colorbar()
plt.title("Diff map with max point")

for name, grid in [("true", true_grid), ("simulation", sim_grid), ("diff", diff_grid)]:
    plt.figure(figsize=(8, 8))

    if name == "diff":
        plt.imshow(grid, origin="lower", cmap="coolwarm")
    else:
        plt.imshow(grid, origin="lower")

    plt.scatter(col, row, color="red", s=50, marker="o")
    plt.colorbar()
    plt.title(name)
    plt.savefig(os.path.join(sim_file_dir, f"elevation_{name}_{rank}.png"))

# diff window grid plot
w = 2
r0 = max(row - w, 0)
r1 = min(row + w + 1, shape[0])
c0 = max(col - w, 0)
c1 = min(col + w + 1, shape[1])

# part_grid = partition_array.reshape(shape)

plt.figure(figsize=(26, 26))
plt.imshow(diff_grid[r0:r1, c0:c1], origin="lower", cmap="coolwarm")
plt.scatter(col - c0, row - r0, color="black", s=80)
# # partition number text
# for i in range(r0, r1):
#     for j in range(c0, c1):
#         pid = part_grid[i, j]
#         plt.text(j-c0,
#                  i-r0,
#                  str(pid),
#                  ha='center',
#                  va='center',
#                  color='white',
#                  fontsize=5)
plt.colorbar()
plt.title("Local diff around max error")
plt.savefig(os.path.join(sim_file_dir, f"elevation_diff_window{rank}.png"))

# elevation window plot
sim_grid = simulation_elev.reshape(shape)
true_grid = true_elev.reshape(shape)
init_grid = init_elev.reshape(shape)
for name, grid in zip(
    ["parallel", "serial", "initial"], [sim_grid, true_grid, init_grid]
):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid[r0:r1, c0:c1], origin="lower")
    plt.scatter(col - c0, row - r0, color="red", s=20)
    # add elevation value text
    for i in range(r0, r1):
        for j in range(c0, c1):
            elev = grid[i, j]
            plt.text(
                j - c0,
                i - r0,
                f"{elev:.3e}",  # or use f"{elev:.3f}"
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    plt.title(name)
    plt.colorbar()
    plt.savefig(os.path.join(sim_file_dir, f"elevation_{name}_window_{rank}.png"))
