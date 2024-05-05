"""Abstract base classes for different Shape types"""

import abc
from typing import Generic, List, Optional, TypeVar

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.construct.edges import Arc
from classy_blocks.construct.flat.sketches.sketch import SketchT
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.types import NPPointType
from classy_blocks.util import functions as f

ShapeT = TypeVar("ShapeT", bound="Shape")


class ShapeCreationError(Exception):
    """Raised when creating a shape from errorneous data"""


class Shape(ElementBase, abc.ABC):
    """A collection of Operations that form a predefined
    parametric shape"""

    @property
    @abc.abstractmethod
    def operations(self) -> List[Operation]:
        """Operations from which the shape is build"""

    @property
    def parts(self):
        return self.operations

    def set_cell_zone(self, cell_zone: str) -> None:
        """Sets cell zone for all blocks in this shape"""
        for operation in self.operations:
            operation.set_cell_zone(cell_zone)

    @property
    def center(self) -> NPPointType:
        """Geometric mean of centers of all operations"""
        return np.average([operation.center for operation in self.operations], axis=0)


class LoftedShape(Shape, abc.ABC, Generic[SketchT]):
    """A Shape, obtained by taking a two and transforming it once
    or twice (middle/end cross-section), then making profiled Lofts
    from calculated cross-sections (Elbow, Cylinder, Ring, ..."""

    def __init__(self, sketch_1: SketchT, sketch_2: SketchT, sketch_mid: Optional[SketchT] = None):
        if len(sketch_1.faces) != len(sketch_2.faces):
            raise ShapeCreationError("Start and end sketch have different number of faces!")

        if sketch_mid is not None and len(sketch_mid.faces) != len(sketch_1.faces):
            raise ShapeCreationError("Mid sketch has a different number of faces from start/end!")

        self.sketch_1 = sketch_1
        self.sketch_2 = sketch_2
        self.sketch_mid = sketch_mid

        self.lofts: List[List[Loft]] = []

        for i, list_1 in enumerate(self.sketch_1.grid):
            self.lofts.append([])

            for j, face_1 in enumerate(list_1):
                face_2 = self.sketch_2.grid[i][j]

                loft = Loft(face_1, face_2)

                # add edges, if applicable
                if self.sketch_mid:
                    face_mid = self.sketch_mid.grid[i][j]

                    for k, point in enumerate(face_mid.points):
                        loft.add_side_edge(k, Arc(point.position))

                self.lofts[-1].append(loft)

    def set_start_patch(self, name: str) -> None:
        """Assign the faces of start sketch to a named patch"""
        for operation in self.operations:
            operation.set_patch("bottom", name)

    def set_end_patch(self, name: str) -> None:
        """Assign the faces of end sketch to a named patch"""
        for operation in self.operations:
            operation.set_patch("top", name)

    @property
    def operations(self):
        return f.flatten_2d_list(self.lofts)

    @property
    def grid(self):
        """Analogous to Sketch's grid but corresponsing operations are returned"""
        return self.lofts
