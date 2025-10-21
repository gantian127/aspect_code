"""
This is to test if passing perimeter link info could create a valid voronoi graph
without extra links or cells
"""
import os
import matplotlib.pyplot as plt

from landlab.graph import DualVoronoiGraph
from landlab.plot.graph import plot_graph

# create output dir
output_dir = os.path.join(os.getcwd(),"experiment","test_vor_graph")
os.makedirs(output_dir, exist_ok=True)

# define x, y, perimeter_link
y = [0.      , 0.      , 0.      , 0.      , 0.866025, 0.866025,
       0.866025, 0.866025, 0.866025, 1.732051, 1.732051, 1.732051,
       1.732051, 1.732051, 1.732051, 2.598076, 2.598076, 2.598076,
       2.598076, 2.598076, 2.598076, 3.464102, 3.464102, 3.464102,
       3.464102, 3.464102, 3.464102, 4.330127, 4.330127, 4.330127,
       4.330127, 4.330127, 4.330127, 5.196152, 5.196152, 5.196152,
       5.196152, 5.196152, 5.196152, 6.062178, 6.062178, 6.062178,
       6.062178, 6.062178, 6.062178, 6.928203, 6.928203, 6.928203,
       6.928203, 6.928203, 6.928203, 6.928203, 7.794229, 7.794229,
       7.794229, 7.794229, 7.794229, 7.794229, 7.794229]

x = [0. , 1. , 2. , 3. , 0.5, 1.5, 2.5, 3.5, 4.5, 0. , 1. , 2. , 3. ,
       4. , 5. , 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0. , 1. , 2. , 3. , 4. ,
       5. , 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0. , 1. , 2. , 3. , 4. , 5. ,
       0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 0. , 1. , 2. , 3. , 4. , 5. , 6. ,
       0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

perimeter_links =[[0, 1], [1, 2], [2, 3], [52, 53], [53, 54], [54, 55], [55, 56],
                  [56, 57], [57, 58], [0, 4], [4, 9], [9, 15], [15, 21], [21, 27],
                  [27, 33], [33, 39], [39, 45], [45, 52], [3, 7], [7, 8], [8, 14],
                  [14, 20], [20, 26], [26, 32], [32, 38], [38, 44], [44, 51], [51, 58]]

vor_graph = DualVoronoiGraph((y, x), sort=True,
                             perimeter_links=perimeter_links)
for option in ['node', 'link', 'cell']:
    fig, ax = plt.subplots(figsize=(16, 16))
    plot_graph(vor_graph, at=option, axes=ax)
    ax.set_title(f'vor_graph_{option}')
    fig.savefig(os.path.join(output_dir, f'vor_graph_{option}.png'))
    plt.close(fig)