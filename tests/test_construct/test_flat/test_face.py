import unittest
from typing import cast

import numpy as np

from classy_blocks.base.exceptions import FaceCreationError
from classy_blocks.base.transforms import Rotation
from classy_blocks.construct import edges
from classy_blocks.construct.flat.face import Face
from classy_blocks.util import functions as f


class FaceTests(unittest.TestCase):
    def setUp(self):
        self.points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]

    def test_face_too_few_points(self):
        """Raise an exception if less than 4 points is provided"""
        with self.assertRaises(FaceCreationError):
            Face(self.points[:3])

    def test_face_invalid_points(self):
        """Raise an exception if given points does not have correct shape"""
        with self.assertRaises(FaceCreationError):
            Face([p[:2] for p in self.points])

    def test_face_center(self):
        """The center property"""
        np.testing.assert_array_equal(Face(self.points).center, [0.5, 0.5, 0])

    def test_reverse(self):
        """Reverse face points"""
        face = Face(self.points)
        face_points = np.copy(face.point_array)

        face.invert()

        np.testing.assert_array_equal(np.flip(face_points, axis=0), face.point_array)

    def test_translate_face(self):
        """Face translation, one custom edge"""
        face_edges = [
            edges.Arc([0.5, -0.25, 0]),
            None,
            None,
            None,
        ]

        translate_vector = f.vector(1, 1, 1)

        original_face = Face(self.points, face_edges)
        translated_face = original_face.copy().translate(translate_vector)

        # check points
        for i in range(4):
            p1 = original_face.points[i].position
            p2 = translated_face.points[i].position

            np.testing.assert_almost_equal(p1, p2 - translate_vector)

        # check arc edge
        translated_arc = cast(edges.Arc, translated_face.edges[0])
        orig_arc = cast(edges.Arc, original_face.edges[0])
        np.testing.assert_almost_equal(translated_arc.point.position - translate_vector, orig_arc.point.position)

    def test_rotate_face_center(self):
        """Face rotation"""
        # only test that the Face.rotate function works properly;
        # other machinery (translate, transform...) are tested in
        # test_translate_face above
        origin = [0.5, 0.5, 0]
        angle = np.pi / 3
        axis = np.array([1.0, 1.0, 1.0])

        original_face = Face(self.points)
        rotated_face = original_face.copy().rotate(angle, axis)

        for i in range(4):
            original_point = original_face.points[i].position
            rotated_point = rotated_face.points[i].position

            np.testing.assert_almost_equal(rotated_point, f.rotate(original_point, angle, axis, origin))

    def test_rotate_face_custom_origin(self):
        """Face rotation"""
        # only test that the Face.rotate function works properly;
        # other machinery (translate, transform...) are tested in
        # test_translate_face above
        origin = [2.0, 2.0, 2.0]
        angle = np.pi / 3
        axis = np.array([1.0, 1.0, 1.0])

        original_face = Face(self.points)
        rotated_face = original_face.copy().rotate(angle, axis, origin)

        for i in range(4):
            original_point = original_face.points[i].position
            rotated_point = rotated_face.points[i].position

            np.testing.assert_almost_equal(rotated_point, f.rotate(original_point, angle, axis, origin))

    def test_rotate_default_origin(self):
        """Rotate a face using the default origin"""
        # Default origin will use rotation around center;
        # construct a face aroung center
        original_face = Face(self.points)
        original_face.translate(-original_face.center)

        rotated_face = original_face.copy().transform([Rotation([0, 0, 1], np.pi / 2)])

        for i in range(4):
            np.testing.assert_almost_equal(original_face.points[i].position, rotated_face.points[(i + 3) % 4].position)

    def test_scale_face_center(self):
        """Scale face from its center"""
        face = Face(self.points)
        face.scale(2)

        scaled_points = [[-0.5, -0.5, 0], [1.5, -0.5, 0], [1.5, 1.5, 0], [-0.5, 1.5, 0]]

        np.testing.assert_array_almost_equal(face.point_array, scaled_points)

    def test_scale_face_custom_origin(self):
        """Scale face from custom origin"""
        face = Face(self.points).scale(2, [0, 0, 0])

        scaled_points = np.array(self.points) * 2

        np.testing.assert_array_almost_equal(face.point_array, scaled_points)

    def test_scale_face_edges(self):
        """Scale face with one custom edge and check its new data"""
        face = Face(self.points, [edges.Arc([0.5, -0.25, 0]), None, None, None]).scale(2, origin=[0, 0, 0])

        arc = cast(edges.Arc, face.edges[0])
        np.testing.assert_array_equal(arc.point.position, [1, -0.5, 0])

    def test_add_edge(self):
        """Replace a Line edge with something else"""
        face = Face(self.points)
        face.add_edge(0, edges.Project("terrain"))

        self.assertEqual(face.edges[0].kind, "project")

        with self.assertRaises(FaceCreationError):
            face.add_edge(4, edges.Project("terrain"))

    def test_replace_edge(self):
        face = Face(self.points)
        face.add_edge(0, edges.Project("terrain"))
        face.add_edge(0, None)

        self.assertEqual(face.edges[0].kind, "line")

    def test_reorient_indexes(self):
        face = Face(self.points)

        face.reorient([2, 2, 0])

        orig_points = np.array(self.points)
        new_points = np.array([point.position for point in face.points])

        np.testing.assert_array_equal(np.roll(orig_points, 2, axis=0), new_points)

    def test_reorient_normal(self):
        face = Face(self.points)
        orig_normal = face.normal

        face.reorient([2, 2, 0])
        new_normal = face.normal

        np.testing.assert_array_equal(orig_normal, new_normal)

    def test_update(self):
        face = Face(self.points)

        new_points = np.array(self.points) + f.vector(1, 1, 1)
        face.update(new_points)

        np.testing.assert_array_equal(face.point_array, new_points)

    def test_wrong_edge_count(self):
        with self.assertRaises(FaceCreationError):
            _ = Face(self.points, [edges.Arc([1, 1, 1]), None, None, None, None])

    def test_check_coplanar_raise(self):
        points = self.points
        points[-1] = [1.0, 1.0, 1.0]

        with self.assertRaises(FaceCreationError):
            _ = Face(self.points, check_coplanar=True)

    def test_remove_edge(self):
        face = Face(self.points)
        face.add_edge(0, edges.Project("test1"))
        face.add_edge(1, edges.Project("test2"))
        face.add_edge(2, edges.Project("test3"))

        face.remove_edges([1])

        self.assertIsInstance(face.edges[1], edges.Line)

    def test_remove_edges(self):
        face = Face(self.points)
        face.add_edge(0, edges.Project("test1"))
        face.add_edge(1, edges.Project("test2"))
        face.add_edge(2, edges.Project("test3"))

        face.remove_edges()

        for i in range(4):
            self.assertIsInstance(face.edges[i], edges.Line)

    def test_shear(self):
        face = Face(self.points)
        face.shear([0, 1, 0], [0, 0, 0], [1, 0, 0], np.pi / 4)

        np.testing.assert_almost_equal(face.point_array, [[0, 0, 0], [1, 0, 0], [2, 1, 0], [1, 1, 0]])
