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

        maxloc = phase.peak_location(corr_map, True)
        self.assertEqual((0, 0), maxloc)

    def test_shift_full_positive(self):
        image = cv.imread(get_test_path('gulsparv.png'), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        xt = 40
        yt = 50
        image_shifted = util.shift_image(image, xt, yt)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), True)
        maxloc = phase.peak_location(corr_map, True)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((xt, yt)), 2)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), False)
        maxloc = phase.peak_location(corr_map, True)
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
        maxloc = phase.peak_location(corr_map, True)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((xt, yt)), 2)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), False)
        maxloc = phase.peak_location(corr_map, True)
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
        maxloc = phase.peak_location(corr_map, True)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((xt, yt)), 2)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), False)
        maxloc = phase.peak_location(corr_map, True)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((xt, yt)), 2)

    def test_shift_subpix_negative_nonshifted(self):
        image = cv.imread(get_test_path('gulsparv.png'), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        xt = -127.7
        yt = -66.1
        image_shifted = util.shift_image(image, xt, yt)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), True)
        maxloc = phase.peak_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((-xt, -yt)), 1)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), False)
        maxloc = phase.peak_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((-xt, -yt)), 1)

    def test_sub_image(self):
        image = cv.imread(get_test_path('gulsparv.png'), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        x = 100
        y = 80
        size = 100
        sub = util.subimage(image, x, y, size, size)

        corr_map = phase.correlate(np.float32(image), np.float32(sub), True)
        maxloc = phase.peak_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((x, y)), 1)

        corr_map = phase.correlate(np.float32(image), np.float32(sub), False)
        maxloc = phase.peak_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((x, y)), 1)

        x = 600
        y = 500
        size = 100
        sub = util.subimage(image, x, y, size, size)

        corr_map = phase.correlate(np.float32(image), np.float32(sub), True)
        maxloc = phase.peak_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((x, y)), 0)

        corr_map = phase.correlate(np.float32(image), np.float32(sub), False)
        maxloc = phase.peak_location(corr_map)
        np.testing.assert_array_almost_equal(
            np.array(maxloc), np.array((x, y)), 0)
