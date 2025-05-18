"""
This is used to validate SimpleSubmarineDiffuser with voronoi grid without partition

ver2: use hex grid to run the model

https://landlab.csdms.io/tutorials/marine_sediment_transport/simple_submarine_diffuser_tutorial.html
"""
import os

import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import numpy as np

from landlab import HexModelGrid, VoronoiDelaunayGrid
from landlab.components import SimpleSubmarineDiffuser

output = os.path.join(os.getcwd(),"validate2")
os.makedirs(output,exist_ok=True)

## step 1: define hex model grid and assign z values
mg = HexModelGrid((17, 17), spacing=1, node_layout='rect')
z = mg.add_zeros("topographic__elevation", at="node")
cum_depo = mg.add_zeros("total_deposit__thickness", at="node")


midpoint = 8
dx = np.abs(mg.x_of_node - midpoint)
dy = np.abs(mg.y_of_node - midpoint)
ds = np.sqrt(dx * dx + dy * dy)
z[:] = (midpoint - ds) - 3.0
z[z < -3.0] = -3.0
z0 = z.copy()

# plot z
mg.imshow(z, cmap="coolwarm", vmin=-3)
plt.title("Elevation on Global Grid")
plt.savefig(os.path.join(output, "valid_dem_hex.png"))

plt.clf()
plt.plot(mg.x_of_node, z, ".")
plt.plot([0, 17], [0, 0], "b:")
plt.grid(True)
plt.xlabel("Distance (m)")
plt.ylabel("Elevation (m)")
plt.savefig(os.path.join(output,"valid_dem_hex_slice.png"))

# plot total_deposit__thickness
plt.clf()
mg.imshow("total_deposit__thickness", cmap="coolwarm", vmin=-1,vmax=1)
plt.title("Total deposit thickness initiation (m)")
plt.savefig(os.path.join(output,"valid_total_deposit_init.png"))


# step 2: run model
ssd = SimpleSubmarineDiffuser(
    mg, sea_level=0.0, wave_base=1.0, shallow_water_diffusivity=1.0
)
for i in range(50):
    ssd.run_one_step(0.2)
    cum_depo += mg.at_node["sediment_deposit__thickness"]

# step 3: results
# on hex grid
plt.clf()
mg.imshow(cum_depo, cmap="coolwarm")
plt.title("Total deposit thickness results (m)")
plt.savefig(os.path.join(output,"valid_total_deposit_result_hex.png"))
print(cum_depo.max(), cum_depo.min())

diff_z = mg.at_node["topographic__elevation"] - z0
plt.clf()
mg.imshow(diff_z, cmap="coolwarm")
plt.title("Topographic elevation difference (m)")
plt.savefig(os.path.join(output,"valid_dem_diff_hex_model_result.png"))
print(diff_z.max(), diff_z.min())

plt.clf()
mg.imshow("topographic__elevation", cmap="coolwarm")
plt.title("Topographic elevation results (m)")
plt.savefig(os.path.join(output,"valid_dem_hex_result.png"))