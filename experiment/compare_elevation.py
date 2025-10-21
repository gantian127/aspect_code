"""
This is to compare the final results of the elevation for the global grid with different
number of processes
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from landlab import HexModelGrid

compare_np = 5
folder = "/Users/tiga7385/Desktop/aspect_code/experiment/compare_elev"
file_compare = os.path.join(folder, f"ghost_elevation_result_{compare_np}.npy")
file_base = os.path.join(folder, "elevation_result_1.npy")

elev_compare= np.load(file_compare)
elev_base = np.load(file_base)
diff = elev_compare - elev_base


mg = HexModelGrid((17, 17), spacing=1, node_layout='rect')
mg.add_field(f"elev_{compare_np}", elev_compare)
mg.add_field("elev_base", elev_base)
mg.add_field(f"diff_{compare_np}", diff)

for field_name in mg.at_node:
    plt.clf()
    if "elev" in field_name:
        mg.imshow(field_name, cmap="coolwarm", vmin=-3)
        plt.title("Elevation on Global Grid")
        plt.savefig(os.path.join(folder, f"{field_name}.png"))
        plt.close()
    else:
        mg.imshow(field_name, cmap="Blues")
        plt.title(f"Difference of elevation np={compare_np}")
        plt.savefig(os.path.join(folder, f"{field_name}.png"))
        plt.close()

        # visualization
        fig, ax = plt.subplots(figsize=[16, 14])
        sc = ax.scatter(mg.node_x, mg.node_y, c=diff, cmap='coolwarm')
        cbar = fig.colorbar(sc, ax=ax)
        ax.set_title(f"Difference of elevation np={compare_np}")
        for node_id in mg.nodes.flat:
            if diff[node_id]!=0:
                ax.annotate(f"{node_id}",
                            (mg.node_x[node_id], mg.node_y[node_id]),
                            color='black', fontsize=8, ha='center', va='top')
        fig.savefig(os.path.join(folder, f'diff_{compare_np}_point.png'))
        plt.close(fig)


