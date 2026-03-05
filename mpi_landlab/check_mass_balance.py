"""
This is a quick mass balance check for the final results of workflow

method:
run rank = 1
run rank = n
run the code below

this needs to export the final results from global grid as .npy file
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("agg")

rank = 3
shape = (17, 17)
true_file_dir = os.path.join(
    os.getcwd(), "mpi_landlab", "linear_diffusion_output_png_1"
)
file_dir = os.path.join(
    os.getcwd(), "mpi_landlab", f"linear_diffusion_output_png_{rank}"
)

## check elevation difference
simulation_elev = np.load(os.path.join(file_dir, f"elevation_result_{rank}.npy"))
true_elev = np.load(os.path.join(true_file_dir, "elevation_result_1.npy"))

diff = simulation_elev - true_elev
diff.reshape(shape)

print(f"rank {rank} == rank {1}: {np.all(simulation_elev == true_elev)}")
print(f"diff == 0: {np.all(diff == 0)}")

plt.imshow(diff.reshape(shape))
plt.savefig(os.path.join(file_dir, f"elevation_diff_{rank}.png"))

plt.imshow(simulation_elev.reshape(shape))
plt.savefig(os.path.join(file_dir, f"elevation_final_{rank}.png"))

plt.imshow(true_elev.reshape(shape))
plt.savefig(os.path.join(true_file_dir, "elevation_final_true.png"))
