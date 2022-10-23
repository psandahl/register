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

        _, _, _, maxloc = cv.minMaxLoc(corr_map)

        self.assertEqual((0, 0), maxloc)

    def test_shift_full_image_wrapped(self):
        image = cv.imread(get_test_path('gulsparv.png'), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        xt = 40
        yt = 50
        image_shifted = util.shift_image(image, xt, yt, True)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), True)
        _, _, _, maxloc = cv.minMaxLoc(corr_map)
        self.assertEqual((xt, yt), maxloc)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), False)
        _, _, _, maxloc = cv.minMaxLoc(corr_map)
        self.assertEqual((xt, yt), maxloc)

    def test_shift_full_image_non_wrapped(self):
        image = cv.imread(get_test_path('gulsparv.png'), cv.IMREAD_GRAYSCALE)
        self.assertIsNotNone(image)

        xt = 40
        yt = 50
        image_shifted = util.shift_image(image, xt, yt)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), True)
        _, _, _, maxloc = cv.minMaxLoc(corr_map)
        self.assertEqual((xt, yt), maxloc)

        corr_map = phase.correlate(np.float32(
            image), np.float32(image_shifted), False)
        _, _, _, maxloc = cv.minMaxLoc(corr_map)
        self.assertEqual((xt, yt), maxloc)
