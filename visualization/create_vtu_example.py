"""This is the testing code to write vtu files for landlab voronoid grid

references:
Eric's io.py files
https://github.com/mcflugen/landlab-parallel/blob/main/src/landlab_parallel/io.py

landlab legacy_vtk
https://landlab.readthedocs.io/en/latest/generated/api/landlab.io.legacy_vtk.html
"""

from landlab import VoronoiDelaunayGrid
import matplotlib.pyplot as plt
import numpy as np
import os


# Functions from landlab_parallel io.py ############################################
import contextlib
import tempfile
from collections.abc import Sequence
from xml.dom import minidom

import landlab
import meshio


def convert_grid_to_mesh(
    grid: landlab.ModelGrid,
    *,
    include: str = "*",
    exclude: Sequence[str] | None = None,
    z_coord: float = 0.0,
    at: str = "node",
) -> meshio.Mesh:
    fd, vtk_path = tempfile.mkstemp(suffix=".vtk", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(
                landlab.io.legacy_vtk.dump(
                    grid, include=include, exclude=exclude, z_coord=z_coord, at=at
                )
            )
        return meshio.read(vtk_path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(vtk_path)


def write_mesh_to_vtu_string(mesh: meshio.Mesh) -> str:
    fd, vtu_path = tempfile.mkstemp(suffix=".vtu")
    os.close(fd)
    try:
        meshio.write(vtu_path, mesh)
        with open(vtu_path, encoding="utf-8") as stream:
            return stream.read()
    finally:
        with contextlib.suppress(OSError):
            os.remove(vtu_path)


## Testing the code for voronoid grid ###############################################
# output dir
output_dir = os.path.join(os.getcwd(), "experiment/pvtu_vtu")
os.makedirs(output_dir, exist_ok=True)

# define voronoid grid for one partition
rank_0 = {
    "x": [
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        10.5,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
        16.5,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        9.5,
        10.5,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
        16.5,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        9.5,
        10.5,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
        16.5,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        10.5,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
        16.5,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        10.5,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
        16.5,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
        16.5,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
        16.5,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        11.5,
        12.5,
        13.5,
        14.5,
        15.5,
        16.5,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ],
    "y": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.866025,
        0.866025,
        0.866025,
        0.866025,
        0.866025,
        0.866025,
        0.866025,
        1.732051,
        1.732051,
        1.732051,
        1.732051,
        1.732051,
        1.732051,
        1.732051,
        2.598076,
        2.598076,
        2.598076,
        2.598076,
        2.598076,
        2.598076,
        2.598076,
        2.598076,
        3.464102,
        3.464102,
        3.464102,
        3.464102,
        3.464102,
        3.464102,
        3.464102,
        3.464102,
        4.330127,
        4.330127,
        4.330127,
        4.330127,
        4.330127,
        4.330127,
        4.330127,
        4.330127,
        5.196152,
        5.196152,
        5.196152,
        5.196152,
        5.196152,
        5.196152,
        5.196152,
        6.062178,
        6.062178,
        6.062178,
        6.062178,
        6.062178,
        6.062178,
        6.062178,
        6.928203,
        6.928203,
        6.928203,
        6.928203,
        6.928203,
        6.928203,
        6.928203,
        7.794229,
        7.794229,
        7.794229,
        7.794229,
        7.794229,
        7.794229,
        7.794229,
        8.660254,
        8.660254,
        8.660254,
        8.660254,
        8.660254,
        8.660254,
        9.526279,
        9.526279,
        9.526279,
        9.526279,
        9.526279,
        9.526279,
        10.392305,
        10.392305,
        10.392305,
        10.392305,
        10.392305,
        11.25833,
        11.25833,
        11.25833,
        11.25833,
        11.25833,
        11.25833,
        12.124356,
        12.124356,
        12.124356,
        12.124356,
        12.124356,
        12.990381,
        12.990381,
        12.990381,
        12.990381,
        12.990381,
        12.990381,
        13.856406,
        13.856406,
        13.856406,
        13.856406,
        13.856406,
        13.856406,
    ],
}

vmg = VoronoiDelaunayGrid(rank_0["x"], rank_0["y"])
elev = np.arange(vmg.number_of_nodes)
vmg.add_field("topographic_elevation", elev)

vmg.imshow("topographic_elevation")
plt.savefig(os.path.join(output_dir, "elevation.png"))
plt.close()

# dump vtk
lines = landlab.io.legacy_vtk.dump(vmg, z_coord=elev).splitlines()
print(os.linesep.join(lines[:]))

# create vtk mesh (create vtk mesh)
fd, vtk_path = tempfile.mkstemp(suffix=".vtk", text=True)
with os.fdopen(fd, "w", encoding="utf-8") as stream:
    stream.write(landlab.io.legacy_vtk.dump(vmg, z_coord=elev))
vtk_mesh = meshio.read(vtk_path)

with contextlib.suppress(OSError):
    os.remove(vtk_path)

# write vtk to vtu string (create vtu file from vtk mesh)
fd, vtu_path = tempfile.mkstemp(suffix=".vtu")
os.close(fd)

meshio.write(vtu_path, vtk_mesh)
with open(vtu_path, encoding="utf-8") as stream:
    vtk_string = stream.read()

with contextlib.suppress(OSError):
    os.remove(vtu_path)

# write vkt string to a file
content = "\n".join(
    [
        line
        for line in minidom.parseString(vtk_string)
        .toprettyxml(indent="  ")
        .splitlines()
        if line.strip()
    ]
)

with open(os.path.join(output_dir, "rank0.vtu"), "w") as fp:
    fp.write(content)
