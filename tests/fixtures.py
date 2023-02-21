"""a test mesh:
3 blocks, extruded in z-direction
(these indexes refer to 'fl' and 'cl' variables,
not vertex.mesh_index)

 ^ y-axis
 |
 |   7---6
     | 2 |
 3---2---5
 | 0 | 1 |
 0---1---4   ---> x-axis

After adding the blocks, the following mesh indexes are
in mesh.vertices:

Bottom 'floor':

 ^ y-axis
 |
 |  13--12
     | 2 |
 3---2---9
 | 0 | 1 |
 0---1---8   ---> x-axis

Top 'floor':

 ^ y-axis
 |
 |  15--14
     | 2 |
 7---6--11
 | 0 | 1 |
 4---5--10   ---> x-axis"""
import unittest

from typing import List, Optional
import dataclasses

from classy_blocks.data.block_data import BlockData

fl:List[List[float]] = [  # points on the 'floor'; z=0
    [0, 0, 0],  # 0
    [1, 0, 0],  # 1
    [1, 1, 0],  # 2
    [0, 1, 0],  # 3
    [2, 0, 0],  # 4
    [2, 1, 0],  # 5
    [2, 2, 0],  # 6
    [1, 2, 0],  # 7
]

cl = [[p[0], p[1], 1] for p in fl]  # points on ceiling; z = 1

@dataclasses.dataclass
class TestBlockData:
    """to store predefined data for test block creation"""
    # points from which to create the block
    indexes:List[int]

    # edges; parameters correspond to block.add_edge() args
    edges:List = dataclasses.field(default_factory=list)

    # chop counts (for each axis, use None to not chop)
    counts:List[Optional[int]] = dataclasses.field(default_factory=lambda: [None, None, None])

    # calls to set_patch()
    patches:List = dataclasses.field(default_factory=list) 

    # other thingamabobs
    description:str = ""
    cell_zone:str = ""

block_data = [
    TestBlockData(
        indexes=[0, 1, 2, 3],
        edges=[ # edges
            [0, 1, 'arc', [0.5, -0.25, 0]],
            [1, 2, 'spline', [[1.1, 0.25, 0], [1.05, 0.5, 0], [1.1, 0.75, 0]]]
        ],
        counts=[6, None, None], # chops
        patches=[
            ["left", "inlet"],
            [["bottom", "top", "front", "back"], "walls", "wall"]
        ],
        description="Test"
    ),
    TestBlockData(
        indexes=[1, 4, 5, 2],
        edges=[
            [3, 0, 'arc', [0.5, -0.1, 1]], # duplicated edge in block 2 that must not be included
            [0, 1, 'arc', [0.5, 0, 0]]  # collinear point; invalid edge must be dropped
        ],
        counts=[5, 6, None],
        patches=[
            [["bottom", "top", "right", "front"], "walls", "wall"],
        ]
    ),
    TestBlockData(
        indexes=[2, 5, 6, 7],
        counts=[None, 8, 7],
        patches=[
            ["back", "outlet"],
            [["bottom", "top", "left", "right"], "walls"]
        ]
    )
]

class FixturedTestCase(unittest.TestCase):
    """Test case with ready-made block data"""
    @staticmethod
    def get_block(index:int) -> BlockData:
        """Returns a list of predefined blocks for testing"""
        data = block_data[index]

    
        points = [fl[i] for i in data.indexes] + [cl[i] for i in data.indexes]
        block = BlockData(points)

        for edge in data.edges:
            block.add_edge(*edge)

        for axis, count in enumerate(data.counts):
            if count is not None:
                block.chop(axis, count=count)

        for patch in data.patches:
            block.set_patch(*patch)

        block.comment = data.description
        block.cell_zone = data.cell_zone

        return block

    @staticmethod
    def get_blocks() -> List[BlockData]:
        """Returns all prepared block data"""
        return [FixturedTestCase.get_block(i) for i in range(len(block_data))]

    @staticmethod
    def get_vertex_indexes(index:int) -> List[int]:
        """Indexes of the points used for block creation;
        will be used in tests to create Vertices manually"""
        return block_data[index].indexes