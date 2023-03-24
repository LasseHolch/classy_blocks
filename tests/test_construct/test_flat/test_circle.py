import unittest
from parameterized import parameterized

import numpy as np

from classy_blocks.construct.flat.disk import QuarterDisk

from classy_blocks.util.constants import TOL
from classy_blocks.util import functions as f

class QuarterDiskTests(unittest.TestCase):
    @property
    def qdisk(self):
        return QuarterDisk([0., 0., 0.], [1., 0., 0.], [0., 0., 1.])

    def assert_coincident(self, qdisk:QuarterDisk):
        pairs = (
            # (i, j, k):
            # core[0].faces[i], shell[j].faces[k]
            (1, 0, 0), # S1: core face's corner 1 and shell's faces[0]'s corner 0
            (2, 0, 3), # D
            (2, 1, 0), # D
            (3, 1, 3), # S2

        )
        for data in pairs:
            core_point = qdisk.core[0].points[data[0]]
            shell_point = qdisk.shell[data[1]].points[data[2]]

            np.testing.assert_array_almost_equal(core_point, shell_point)

    def test_quarter_translate(self):
        """Check that the coincident points remain coincident after translate"""
        qcrc = self.qdisk.translate([1, 1, 1])

        self.assert_coincident(qcrc)

    def test_quarter_rotate(self):
        """Check that the coincident points remain coincident after translate"""
        qcrc = self.qdisk.rotate(np.pi/3, [0, 0, 1], [1, 1, 1])

        self.assert_coincident(qcrc)

    def test_quarter_scale_origin(self):
        """Check that the coincident points remain coincident after translate"""
        qcrc = self.qdisk.translate([1, 1, 1])
        qcrc.scale(0.5, [10, 10, 10])

        self.assert_coincident(qcrc)

    def test_quarter_scale_origin_default(self):
        """Check that the coincident points remain coincident after translate"""
        qcrc = self.qdisk.translate([1, 1, 1])
        qcrc.scale(0.5)

        self.assert_coincident(qcrc)

    def test_quarter_combined(self):
        """Check that the coincident points remain coincident after a combination of transforms"""
        qcrc  = self.qdisk.translate([-1, 0, 0])
        qcrc.rotate(np.pi/3, [0, 0, 1], [1, 1, 1])
        qcrc.scale(2)

        self.assert_coincident(qcrc)

    @parameterized.expand(((0, ), (1, ), (2, )))
    def test_face(self, i_face):
        """Check that quarter disk's faces are properly constructed"""
        # That is, each face has 4 different points
        points = self.qdisk.faces[i_face].points

        self.assertGreater(f.norm(points[1] - points[0]), TOL)
        self.assertGreater(f.norm(points[2] - points[1]), TOL)
        self.assertGreater(f.norm(points[3] - points[2]), TOL)
        self.assertGreater(f.norm(points[0] - points[3]), TOL)

