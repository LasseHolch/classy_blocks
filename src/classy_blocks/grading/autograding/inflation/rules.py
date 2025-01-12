from typing import List, Optional, Tuple

from classy_blocks.grading.autograding.inflation.distributor import DoubleInflationDistributor, InflationDistributor
from classy_blocks.grading.autograding.inflation.layers import BufferLayer, BulkLayer, InflationLayer, LayerStack
from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.autograding.smooth.distributor import SmoothDistributor
from classy_blocks.grading.autograding.smooth.rules import SmoothRules
from classy_blocks.grading.chop import Chop


class InflationRules(SmoothRules):
    """See description of InflationGrader"""

    # TODO: refactor to a reasonable number of 'if' clauses

    def __init__(
        self,
        first_cell_size: float,
        bulk_cell_size: float,
        c2c_expansion: float = 1.2,
        bl_thickness_factor: int = 30,
        buffer_expansion: float = 2,
    ):
        self.first_cell_size = first_cell_size
        self.bulk_cell_size = bulk_cell_size
        self.c2c_expansion = c2c_expansion
        self.bl_thickness_factor = bl_thickness_factor
        self.buffer_expansion = buffer_expansion

        # use SmoothGrader's logic for bulk chops
        self.cell_size = self.bulk_cell_size

    def get_inflation_layer(self, max_length: float) -> InflationLayer:
        return InflationLayer(self.first_cell_size, self.c2c_expansion, self.bl_thickness_factor, max_length)

    def get_buffer_layer(self, start_size, max_length: float) -> BufferLayer:
        return BufferLayer(start_size, self.buffer_expansion, self.bulk_cell_size, max_length)

    def get_bulk_layer(self, remaining_length: float, size_after: float) -> BulkLayer:
        return BulkLayer(self.bulk_cell_size, size_after, remaining_length)

    def get_stack(self, length: float, size_after: Optional[float] = None) -> LayerStack:
        stack = LayerStack(length)

        inflation = self.get_inflation_layer(length)
        if stack.add(inflation):
            return stack

        buffer = self.get_buffer_layer(stack.layers[0].end_size, stack.remaining_length)
        if stack.add(buffer):
            return stack

        if size_after is None:
            size_after = self.bulk_cell_size
        bulk = self.get_bulk_layer(stack.remaining_length, size_after)
        stack.add(bulk)

        return stack

    def get_count(self, length: float, starts_at_wall: bool, ends_at_wall: bool):
        if not (starts_at_wall or ends_at_wall):
            return super().get_count(length, False, False)

        if starts_at_wall and ends_at_wall:
            stack = self.get_stack(length / 2)
            return 2 * stack.count

        stack = self.get_stack(length)
        return stack.count

    def get_sizes(self, info: WireInfo) -> Tuple[float, float]:
        size_before = info.size_before
        if size_before is None:
            if info.starts_at_wall:
                size_before = self.first_cell_size
            else:
                size_before = self.cell_size

        size_after = info.size_after
        if size_after is None:
            if info.ends_at_wall:
                size_after = self.first_cell_size
            else:
                size_after = self.cell_size

        return size_before, size_after

    def is_squeezed(self, count: int, info: WireInfo) -> bool:
        if not (info.starts_at_wall or info.ends_at_wall):
            return super().is_squeezed(count, info)

        length = info.length

        if info.starts_at_wall and info.ends_at_wall:
            length = length / 2

        stack = self.get_stack(length, info.size_after)

        if len(stack.layers) < 3:
            return True

        if stack.count < count:
            return True

        return False

    def get_chops(self, count, info: WireInfo) -> List[Chop]:
        # TODO: un-if-if-if
        size_before, size_after = self.get_sizes(info)

        if not (info.starts_at_wall or info.ends_at_wall):
            distributor = SmoothDistributor(count, size_before, info.length, size_after)
            chop_count = 3
        else:
            params = {
                "count": count,
                "size_before": size_before,
                "length": info.length,
                "size_after": size_after,
                "c2c_expansion": self.c2c_expansion,
                "bl_thickness_factor": self.bl_thickness_factor,
                "buffer_expansion": self.buffer_expansion,
                "bulk_size": self.bulk_cell_size,
            }

            if info.starts_at_wall and info.ends_at_wall:
                distributor = DoubleInflationDistributor(**params)
                chop_count = 5
            else:
                distributor = InflationDistributor(**params)
                chop_count = 3

        chops = distributor.get_chops(chop_count)

        return chops

    def get_squeezed_chops(self, count: int, info: WireInfo) -> List[Chop]:
        return self.get_chops(count, info)
