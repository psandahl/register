import cv2 as cv
import numpy as np
import os
import unittest

import register.image.phase as phase
import register.image.util as util


def get_test_path(path: str) -> str:
    return os.path.join(os.getcwd(), 'testdata', path)


class TestPhase(unittest.TestCase):
    def test_zero_shift(self):
        image = cv.imread(get_test_path('gulsparv.png'), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        corr_map = phase.correlate(np.float32(image), np.float32(image), False)

        rows, cols = image.shape
        opt_rows = cv.getOptimalDFTSize(rows)
        opt_cols = cv.getOptimalDFTSize(cols)
        self.assertEqual((opt_rows, opt_cols), corr_map.shape)

        maxloc = phase.max_location(corr_map)
        self.assertEqual((0, 0), maxloc)

    def test_shift_full_positive(self):
        image = cv.imread(get_test_path('gulsparv.png'), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        xt = 40
        yt = 50
        image_shifted = util.shift_image(image, xt, yt)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), True)
        maxloc = phase.max_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((xt, yt)), 2)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), False)
        maxloc = phase.max_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((xt, yt)), 2)

    def test_shift_full_negative(self):
        image = cv.imread(get_test_path('gulsparv.png'), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        xt = -40
        yt = -50
        image_shifted = util.shift_image(image, xt, yt)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), True)
        maxloc = phase.max_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((xt, yt)), 2)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), False)
        maxloc = phase.max_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((xt, yt)), 2)

    def test_shift_subpix_positive(self):
        image = cv.imread(get_test_path('gulsparv.png'), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        xt = 127.7
        yt = 66.1
        image_shifted = util.shift_image(image, xt, yt)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), True)
        maxloc = phase.max_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((xt, yt)), 2)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), False)
        maxloc = phase.max_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((xt, yt)), 2)

    def test_sub_image(self):
        image = cv.imread(get_test_path('gulsparv.png'), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        # Taking a sub image is equal to a negative shift.
        x = 300
        y = 200
        size = 100
        sub = image[y:y+size, x:x+size]

        corr_map = phase.correlate(np.float32(image), np.float32(sub), True)
        maxloc = phase.max_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((-x, -y)), 1)

        corr_map = phase.correlate(np.float32(image), np.float32(sub), False)
        maxloc = phase.max_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((-x, -y)), 1)
