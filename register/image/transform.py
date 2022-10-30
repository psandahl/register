import cv2 as cv
import logging
import math
import numpy as np

import register.image.phase as phase

logger = logging.getLogger(__name__)


def warp_polar(image: np.ndarray) -> np.ndarray:
    rows = image.shape[0]
    cols = image.shape[1]

    log_polar_size = max(rows, cols)

    log_base = math.exp(math.log(rows * 1.1 / 2.0) / log_polar_size)
    ellipse_coeff = rows / cols

    scales = []
    for i in range(0, log_polar_size):
        scales.append(math.pow(log_base, i))

    scales = np.array(scales, dtype=np.float32).reshape(log_polar_size, 1)
    ones = np.ones(log_polar_size, dtype=np.float32).reshape(log_polar_size, 1)
    scales_matrix = ones @ scales.T

    angles = np.linspace(0, 2 * np.pi, log_polar_size,  # pi or 2 * pi?
                         dtype=np.float32).reshape(log_polar_size, 1)
    angles *= -1.0  # Shall it be like this?
    angles_matrix = angles @ ones.T

    cos_matrix = np.cos(angles_matrix) / ellipse_coeff
    sin_matrix = np.sin(angles_matrix)

    center_x = cols / 2
    center_y = rows / 2

    x_map = scales_matrix * cos_matrix + center_x
    y_map = scales_matrix * sin_matrix + center_y

    return cv.remap(image, x_map, y_map, cv.INTER_CUBIC & cv.INTER_MAX, cv.BORDER_CONSTANT)


def get_scale_and_rotation(pcorr: np.ndarray) -> tuple():
    rows, cols = pcorr.shape
    assert rows == cols

    peak_x, peak_y = phase.peak_location(pcorr, True)
    rotation = (2.0 * -math.pi) * peak_y / rows
    rotation = -math.fmod(rotation, 2 * math.pi)

    log_base = math.exp(math.log(rows * 1.1 / 2.0) / rows)
    scale = math.pow(log_base, peak_x)
    scale = 1.0 / scale

    return scale, math.degrees(rotation)
