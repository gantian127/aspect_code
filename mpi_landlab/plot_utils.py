from __future__ import annotations

import contextlib
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from typing import IO
from xml.dom import minidom

import landlab
import meshio
import numpy as np


def vtu_dump(
    grid: landlab.ModelGrid,
    stream: IO[str] | None = None,
    include: str = "*",
    exclude: Sequence[str] | None = None,
    z_coord: float = 0.0,
    at: str = "node",
) -> str | None:
    """Return a VTU representation of a grid.

    Parameters
    ----------
    grid : landlab.ModelGrid
        Grid containing fields to output.
    stream : file-like object or None, optional
        Stream to write VTU data to. If ``None`` the string is returned.
    include : sequence of str or "*", optional
        Fields to include.
    exclude : sequence of str or None, optional
        Fields to exclude.
    z_coord : float, optional
        Elevation to use for 2D grids.
    at : {"node", "cell"}, optional
        Location of values within the grid.

    Returns
    -------
    str or None
        VTU representation or ``None`` if written to ``stream``.
    """
    mask = grid.status_at_node == grid.BC_NODE_IS_CLOSED
    saved_fields = {
        name: grid.at_node[name]
        for name in grid.at_node
        if np.issubdtype(grid.at_node[name].dtype, np.floating)
    }
    for name, array in saved_fields.items():
        grid.at_node[name] = array.copy()
        grid.at_node[name][mask] = np.nan

    mesh = convert_grid_to_mesh(
        grid, include=include, exclude=exclude, z_coord=z_coord, at=at
    )

    for name, array in saved_fields.items():
        grid.at_node[name] = array

    contents = write_mesh_to_vtu_string(mesh)

    content = "\n".join(
        [
            line
            for line in minidom.parseString(contents)
            .toprettyxml(indent="  ")
            .splitlines()
            if line.strip()
        ]
    )

    if stream is None:
        return content
    else:
        stream.write(content)
        return None


def pvtu_dump(grid: landlab.ModelGrid, vtu_files: Sequence[str] = ()) -> str:
    """Return a PVTU file referencing multiple VTU files.

    Parameters
    ----------
    grid : landlab.ModelGrid
        Grid from which point fields are derived.
    vtu_files : sequence of str, optional
        Paths to VTU files that comprise the full grid.

    Returns
    -------
    str
        XML string describing the parallel unstructured grid.
    """
    vtkfile = ET.Element(
        "VTKFile", type="PUnstructuredGrid", version="1.0", byte_order="LittleEndian"
    )
    pgrid = ET.SubElement(vtkfile, "PUnstructuredGrid", GhostLevel="0")

    ppoints = ET.SubElement(pgrid, "PPoints")
    ET.SubElement(
        ppoints, "PDataArray", type="Float32", NumberOfComponents="3", Name="Points"
    )

    if grid.at_node:
        pdata = ET.SubElement(pgrid, "PPointData")
        for name in grid.at_node:
            ET.SubElement(
                pdata, "PDataArray", type="Float32", Name=name, NumberOfComponents="1"
            )

    for f in vtu_files:
        ET.SubElement(pgrid, "Piece", Source=os.path.basename(f))

    tree = ET.ElementTree(vtkfile)

    root = tree.getroot()
    if root is None:
        raise RuntimeError(
            "ElementTree has no root. Unable to construct a VTK file from"
            " the provided grid."
        )
    parsed = minidom.parseString(ET.tostring(root))

    return parsed.toprettyxml(indent="  ")


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


def create_pvd(pvtu_files, timesteps, output_pvd="simulation.pvd"):
    """
    Create a PVD file from a list of .pvtu files and their timesteps.

    Parameters
    ----------
    pvtu_files : list of str
        List of .pvtu filenames (e.g., ["step0.pvtu", "step1.pvtu", ...])
    timesteps : list of float or int
        Corresponding timesteps for each file
    output_pvd : str
        Output PVD filename
    """
    vtkfile = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(vtkfile, "Collection")

    for file, t in zip(pvtu_files, timesteps):
        ET.SubElement(collection, "DataSet", timestep=str(t), part="0", file=file)

    tree = ET.ElementTree(vtkfile)

    # Get root element
    root = tree.getroot()
    if root is None:
        raise RuntimeError("ElementTree has no root. Unable to construct a VTK file.")

    # Convert to a pretty XML string
    xml_str = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(xml_str)
    pretty_xml = parsed.toprettyxml(indent="  ")

    # Write to file
    with open(output_pvd, "w", encoding="utf-8") as fp:
        fp.write(pretty_xml)
