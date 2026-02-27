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

rank = 3
shape = [6, 6]
true_file_dir = os.path.join(os.getcwd(), "mpi_landlab","linear_diffusion_output_png_1")
file_dir = os.path.join(os.getcwd(), "mpi_landlab",f"linear_diffusion_output_png_{rank}")

## check elevation difference
simulation_elev = np.load(os.path.join(file_dir, f"elevation_result_{rank}.npy"))
true_elev = np.load(os.path.join(true_file_dir, "elevation_result_1.npy"))

np.all(simulation_elev==true_elev)

diff = simulation_elev - true_elev
diff.reshape(shape)

plt.imshow(diff.reshape(shape))
plt.savefig(os.path.join(file_dir,"compare_final_elev_diff.png" ))


