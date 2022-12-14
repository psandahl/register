import cv2 as cv
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


def scale_rotate_image(image: np.ndarray, scale: float, degrees: float) -> np.ndarray:
    """
    Scale and rotate an image.

    Parameters:
        image: The image to be rotated.
        scale: The scale factor to apply to the image
        degrees: The angle to rotate the image with.

    Returns:
        The warped image.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    center = (cols / 2, rows / 2)

    M = cv.getRotationMatrix2D(center, degrees, scale)

    return cv.warpAffine(image, M, (cols, rows), borderMode=cv.BORDER_WRAP)


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


def filtered_resize(image: np.ndarray, filter: np.ndarray, opt_size: tuple) -> np.ndarray:
    """
    Apply a filter, and a resize if necessary.

    Parameters:
        image: The image.
        filter: The filter - can be None.
        opt_size: Tuple (rows, cols) to adapt to.

    Returns:
        The resized image.
    """
    assert len(image.shape) == 2
    assert image.dtype == np.float32 or image.dtype == np.float64

    if not filter is None:
        assert filter.shape == image.shape
        assert filter.dtype == image.dtype

    rows, cols = image.shape
    opt_rows, opt_cols = opt_size

    if opt_rows > rows or opt_cols > cols:
        if not filter is None:
            return resize_with_border(image * filter, opt_size)
        else:
            return resize_with_border(image, opt_size)
    else:
        if not filter is None:
            return image * filter
        else:
            return image


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

    input_image = filtered_resize(image, filter, (opt_rows, opt_cols))
    dft = np.fft.fft2(input_image)

    magnitude = np.abs(dft)
    magnitude = np.fft.fftshift(magnitude)

    return np.log(magnitude)


def subimage(image: np.ndarray, xstart: int, ystart: int, xsize: int, ysize: int) -> np.ndarray:
    """
    Extract a subimage from the image.

    Parameters:
        image: The image to extract from.
        xstart: The start x value.
        ystart: The start y value.
        xsize: The size in x.
        ysize: The size in y.

    Returns:
        The subimage.
    """
    return image[ystart:ystart + ysize, xstart:xstart + xsize]


def high_pass_filter(rows: int, cols: int) -> np.ndarray:
    ys = np.linspace(-math.pi / 2.0, math.pi / 2.0, rows,
                     dtype=np.float32).reshape(rows, 1)
    y_ones = np.ones(cols, dtype=np.float32).reshape(cols, 1)
    y_matrix = ys @ y_ones.T

    xs = np.linspace(-math.pi / 2.0, math.pi / 2.0, cols,
                     dtype=np.float32).reshape(cols, 1)
    x_ones = np.ones(rows, dtype=np.float32).reshape(rows, 1)
    x_matrix = x_ones @ xs.T

    matrix = y_matrix * y_matrix + x_matrix * x_matrix
    filter = np.cos(np.sqrt(matrix))

    filter *= filter
    filter = -filter + 1.0

    return filter
