"""Defines a numbered vertex in 3D space and all operations
that can be applied to it."""
from classy_blocks.construct.point import Point
from classy_blocks.types import PointType
from classy_blocks.util.constants import vector_format


class Vertex(Point):
    """A 3D point in space with all transformations and an assigned index"""

    # keep the list as a class variable
    def __init__(self, position: PointType, index: int):
        super().__init__(position)

        # index in blockMeshDict; address of this object when creating edges/blocks
        self.index = index

    def __eq__(self, other):
        return self.index == other.index

    def __repr__(self):
        return f"Vertex {self.index} at {self.pos}"

    @property
    def description(self) -> str:
        """Returns a string representation to be written to blockMeshDict"""
        point = vector_format(self.pos)
        comment = f"// {self.index}"

        if len(self.project_to) > 0:
            return f"project {point} ({' '.join(self.project_to)}) {comment}"

        return f"{point} {comment}"
