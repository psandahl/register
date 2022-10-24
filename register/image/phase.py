import cv2 as cv
import logging
import numpy as np

import register.image.util as util

logger = logging.getLogger(__name__)


def correlate(image1: np.ndarray, image2: np.ndarray, hanning: bool) -> np.ndarray:
    """
    Perform phase correlation. Returns the raw correlation map.

    Parameters:
        image1: Reference image.
        image2: Query image.
        hanning: Flag to tell if images shall have a Hanning window.

    Returns:
        Correlation map.
    """
    assert image1.dtype == np.float32 or image1.dtype == np.float64
    assert image1.dtype == image2.dtype
    assert len(image1.shape) == 2 and len(image2.shape) == 2
    assert image2.shape[0] <= image1.shape[0]
    assert image2.shape[1] <= image1.shape[1]

    rows1, cols1 = image1.shape
    rows2, cols2 = image2.shape

    float_type = cv.CV_32F if image1.dtype == np.float32 else cv.CV_64F

    filter1 = None
    filter2 = None
    if hanning:
        filter1 = cv.createHanningWindow((cols1, rows1), float_type)
        filter2 = cv.createHanningWindow((cols2, rows2), float_type)
        logger.debug(f'Hanning filters created')

    opt_rows = cv.getOptimalDFTSize(rows1)
    opt_cols = cv.getOptimalDFTSize(cols1)
    opt_size = (opt_rows, opt_cols)

    logger.debug(
        f'rows={rows1} opt_rows={opt_rows} cols={cols1} opt_cols={opt_cols}')

    input_image1 = util.filtered_resize(image1, filter1, opt_size)
    input_image2 = util.filtered_resize(image2, filter2, opt_size)

    dft1 = np.fft.fft2(input_image1)
    dft2 = np.fft.fft2(input_image2)

    c = dft1 * dft2.conj()
    cross_spectrum = c / np.abs(c)

    return np.fft.ifft2(cross_spectrum).real


def max_location(corr_map: np.ndarray) -> tuple():
    """
    Compute the max location for the correlation map.

    Parameters:
        corr_map: The correlation map.

    Returns:
        Tuple (x, y): The shift in x, y of image1 to fit image2.
    """
    shifted_corr_map = np.fft.fftshift(corr_map)
    _, _, _, maxloc = cv.minMaxLoc(shifted_corr_map)

    centroid_x, centroid_y = centroid(shifted_corr_map, maxloc)

    rows, cols = corr_map.shape
    center_x = cols / 2
    center_y = rows / 2

    return center_x - centroid_x, center_y - centroid_y


def centroid(corr_map: np.ndarray, center: tuple) -> tuple():
    """
    Compute a 5x5 centroid around the given center.
    """
    rows, cols = corr_map.shape
    center_x, center_y = center
    r = 2

    start_x = max(0, center_x - r)
    end_x = min(cols - 1, center_x + r)
    start_y = max(0, center_y - r)
    end_y = min(rows - 1, center_y + r)

    sum = 0.0
    x_w = 0.0
    y_w = 0.0

    for y in range(start_y, end_y + 1):
        for x in range(start_x, end_x + 1):
            value = corr_map[y, x]
            sum += value
            x_w += x * value
            y_w += y * value

    return x_w / sum, y_w / sum
