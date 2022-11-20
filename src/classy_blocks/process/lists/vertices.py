
from typing import Optional, List

from numpy.typing import ArrayLike

from classy_blocks.define.vertex import Vertex
from classy_blocks.define.block import Block

from classy_blocks.util import constants
from classy_blocks.util import functions as f

class VertexList:
    """ Handling of the 'vertices' part of blockMeshDict """
    def __init__(self):
        self.vertices:List[Vertex] = []

    def find(self, position:ArrayLike) -> Optional[Vertex]:
        """checks if any of existing vertices in self.vertices are
        in the same location as the passed one; if so, returns
        the existing vertex"""
        # TODO: optimize (octree/kdtree from scipy)
        for vertex in self.vertices:
            if f.norm(vertex.point - position) < constants.tol:
                return vertex

        return None

    def collect(self, blocks:List[Block], merged_patches:List[List[str]]) -> None:
        """ Collects all vertices from all given blocks,
        checks for duplicates and gives them indexes """
        for block in blocks:
            for i, block_vertex in enumerate(block.vertices):
                found_vertex = self.find(block_vertex.point)

                if found_vertex is None:
                    block.vertices[i].mesh_index = len(self.vertices)
                    self.vertices.append(block_vertex)
                else:
                    block.vertices[i] = found_vertex

        # merged patches: duplicate all points that define slave patches
        duplicated_points = {}  # { original_index:new_vertex }
        slave_patches = [mp[1] for mp in merged_patches]

        for patch in slave_patches:
            for block in blocks:
                if patch in block.patches:
                    patch_sides = block.get_patch_sides(patch)
                    for side in patch_sides:
                        face_indexes = block.get_side_indexes(side, local=True)

                        for i in face_indexes:
                            vertex = block.vertices[i]

                            if vertex.mesh_index not in duplicated_points:
                                new_vertex = Vertex(vertex.point)
                                new_vertex.mesh_index = len(self.vertices)
                                self.vertices.append(new_vertex)

                                block.vertices[i] = new_vertex

                                duplicated_points[vertex.mesh_index] = new_vertex
                            else:
                                block.vertices[i] = duplicated_points[vertex.mesh_index]

    def __getitem__(self, index):
        return self.vertices[index]

    def __len__(self):
        return len(self.vertices)

    def output(self):
        """ Returns a string of vertices to be written to blockMeshDict """
        vlist = "vertices\n"
        vlist += "(\n"

        for vertex in self.vertices:
            vlist += "\t{} // {}\n".format(
                constants.vector_format(vertex.point),
                vertex.mesh_index
        )

        vlist += ");\n\n"

        return vlist