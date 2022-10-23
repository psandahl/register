import cv2 as cv
import logging
import numpy as np

import register.image.util as util

logger = logging.getLogger(__name__)


def correlate(image1: np.ndarray, image2: np.ndarray, hanning: bool) -> np.ndarray:
    """
    Perform phase correlation. The resulting corr map will give the
    relation translation of image2 to image1.

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

    c = dft2 * dft1.conj()  # This is different compared to cv, I think ...
    cross_spectrum = c / np.abs(c)

    return np.fft.ifft2(cross_spectrum).real
