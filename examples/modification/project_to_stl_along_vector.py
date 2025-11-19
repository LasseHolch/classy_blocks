import numpy as np
from stl import mesh as stlmesh

import classy_blocks as cb

mesh = cb.Mesh()

u_0 = np.array([1, 0, 0])
u_1 = np.array([0, 1, 0])
u_2 = np.array([0, 0, 1])

geometry = {
    "meter": ["type triSurfaceMesh", "name meter", 'file "meter_1mm.stl"'],
}
stl_mesh = stlmesh.Mesh.from_file("./meter_05mm.stl")

cell_size = 1
piezo_d = 16
piezo_l = 8
piezo_angle = np.deg2rad(50)


def create_piezo_mesh(start_point: np.ndarray, piezo_u_0: np.ndarray, mesh_write_path: str):
    radius_point = start_point + piezo_d / 2 * u_1
    end_point = start_point + piezo_l * piezo_u_0
    piezo = cb.Cylinder(start_point, end_point, radius_point)

    for face in piezo.sketch_2.faces:
        face.project_edges_to_stl_along_vector(stl_mesh=stl_mesh, vec=piezo_u_0, n_spline_points=10)
        face.project("meter", edges=False, points=False)

    piezo.chop_axial(start_size=cell_size)
    piezo.chop_radial(start_size=cell_size)
    piezo.chop_tangential(start_size=cell_size)

    piezo.set_outer_patch("meterWall")
    mesh.add(piezo)
    mesh.modify_patch("meterWall", "wall")
    mesh.add_geometry(geometry)
    mesh.write(mesh_write_path)


piezo_1_u_0 = np.cos(piezo_angle) * u_0 - np.sin(piezo_angle) * u_2
piezo_1_start_p = np.array([-41.03, -18.44, 48.98])
piezo_1_path = "./piezo_1/system/blockMeshDict"

piezo_2_u_0 = piezo_1_u_0
piezo_2_start_p = piezo_1_start_p * np.array([1, -1, 1])
piezo_2_path = "./piezo_2/system/blockMeshDict"

piezo_3_u_0 = -piezo_1_u_0
piezo_3_start_p = piezo_1_start_p * np.array([-1, 1, -1])
piezo_3_path = "./piezo_3/system/blockMeshDict"

piezo_4_u_0 = -piezo_1_u_0
piezo_4_start_p = piezo_1_start_p * np.array([-1, -1, -1])
piezo_4_path = "./piezo_4/system/blockMeshDict"

for start_p, direction, write_path in zip(
    [piezo_1_u_0, piezo_2_u_0, piezo_3_u_0, piezo_4_u_0],
    [piezo_1_start_p, piezo_2_start_p, piezo_3_start_p, piezo_4_start_p],
    [piezo_1_path, piezo_2_path, piezo_3_path, piezo_4_path],
):
    create_piezo_mesh(start_p, direction, write_path)
