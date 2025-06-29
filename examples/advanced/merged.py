import os

import classy_blocks as cb

mesh = cb.Mesh()

# two boxes, connected with a geometrically identical
# but different cell count
coarse_box = cb.Box([-0.5, -0.5, 0], [0.5, 0.5, 1])
for i in (0, 1, 2):
    coarse_box.chop(i, count=10)
coarse_box.set_patch("bottom", "inlet")
coarse_box.set_patch("top", "box_slave")
mesh.add(coarse_box)

fine_box = coarse_box.copy().translate([0, 0, 1])
fine_box.chops = {}
for i in (0, 1, 2):
    fine_box.unchop(i)
    fine_box.chop(i, count=25)

fine_box.set_patch("bottom", "box_master")
fine_box.set_patch("top", "cylinder_master")
mesh.add(fine_box)

# merge the boxes
mesh.merge_patches("box_master", "box_slave")

# add another cylinder on top; this will have no
# coincident points
cylinder = cb.Cylinder([0, 0, 2], [0, 0, 3], [0.25, 0, 2])
cylinder.chop_axial(count=10)
cylinder.chop_radial(count=10)
cylinder.chop_tangential(count=10)

cylinder.set_start_patch("cylinder_slave")
cylinder.set_end_patch("outlet")
mesh.add(cylinder)

mesh.merge_patches("cylinder_master", "cylinder_slave")
mesh.set_default_patch("walls", "wall")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
