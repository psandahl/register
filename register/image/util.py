import cv2 as cv
import logging
import numpy as np

logger = logging.getLogger(__name__)


def shift_image(image: np.ndarray, x: float, y: float, wrap: bool = False) -> np.ndarray:
    """
    Shift an image in x and y.

    Parameters:
        image: The image.
        x: Shift in x.
        y: Shift in y.

    Returns:
        The shifted image.
    """
    m = np.array([1, 0, x, 0, 1, y], dtype=np.float64).reshape(2, 3)
    return cv.warpAffine(image, m, image.shape[::-1],
                         borderMode=cv.BORDER_WRAP if wrap else cv.BORDER_CONSTANT)


def resize_with_border(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    Resize an image by adding a zero valued border.

    Parameters:
        image: The image.
        size: Tuple (wanted rows, wanted cols).

    Returns:
        The resized image.
    """
    assert len(image.shape) == 2

    img_rows, img_cols = image.shape
    new_rows, new_cols = size

    assert new_rows > img_rows or new_cols > img_cols

    return cv.copyMakeBorder(
        image, 0, new_rows - img_rows, 0, new_cols - img_cols,
        cv.BORDER_CONSTANT, value=0.0)


def log_magnitude_spectrum(image: np.ndarray, hanning: bool) -> np.ndarray:
    """
    Compute the logarithmic magnitude spectrum for an image.

    Parameters:
        image: The image.
        hanning: Tell if the image should be filtered with a Hanning window.

    Returns:
        The logarithmic magnitude spectrum.
    """
    assert len(image.shape) == 2
    assert image.dtype == np.float32 or image.dtype == np.float64

    rows, cols = image.shape

    float_type = cv.CV_32F if image.dtype == np.float32 else cv.CV_64F

    filter = None
    if hanning:
        filter = cv.createHanningWindow((cols, rows), float_type)
        logger.debug(f'Hanning filter created')

    opt_rows = cv.getOptimalDFTSize(rows)
    opt_cols = cv.getOptimalDFTSize(cols)

    logger.debug(
        f'rows={rows} opt_rows={opt_rows} cols={cols} opt_cols={opt_cols}')

    input_image = None
    if opt_rows > rows or opt_cols > cols:
        if hanning:
            input_image = resize_with_border(
                image * filter, (opt_rows, opt_cols))
        else:
            input_image = resize_with_border(image, (opt_rows, opt_cols))
    else:
        if hanning:
            input_image = image * filter
        else:
            input_image = image

    dft = np.fft.fft2(input_image)

    magnitude = np.abs(dft)
    magnitude = np.fft.fftshift(magnitude)

    return np.log(magnitude)
