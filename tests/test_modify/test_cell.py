from parameterized import parameterized

from classy_blocks.modify.cell import Cell
from tests.fixtures.block import BlockTestCase


class CellTests(BlockTestCase):
    @parameterized.expand(
        (
            (0, 1, 4),
            (0, 2, 2),
            (1, 0, 4),
            (1, 2, 4),
        )
    )
    def test_common_vertices(self, index_1, index_2, count):
        block_1 = self.make_block(index_1)
        block_2 = self.make_block(index_2)

        cell_1 = Cell(block_1)
        cell_2 = Cell(block_2)

        self.assertEqual(len(cell_1.get_common_vertices(cell_2)), count)

    @parameterized.expand(((0, 0, 0), (0, 1, 1), (1, 1, 0), (1, 8, 1)))
    def test_get_corner(self, block, vertex, corner):
        cell = Cell(self.make_block(block))

        self.assertEqual(cell.get_corner(vertex), corner)

    @parameterized.expand(((0, 1, "right"), (1, 0, "left"), (1, 2, "back")))
    def test_get_common_side(self, index_1, index_2, orient):
        cell_1 = Cell(self.make_block(index_1))
        cell_2 = Cell(self.make_block(index_2))

        self.assertEqual(cell_1.get_common_side(cell_2), orient)

    def test_quality_good(self):
        cell = Cell(self.make_block(0))

        self.assertLess(cell.quality, 1)

    def test_quality_bad(self):
        block = self.make_block(0)
        block.vertices[0].move_to([-10, -10, -10])

        cell = Cell(block)

        self.assertGreater(cell.quality, 100)
