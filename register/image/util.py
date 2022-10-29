import cv2 as cv
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)


def rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    """
    Rotate an image.

    Parameters:
        image: The image to be rotated.
        degrees: The angle to rotate the image with.

    Returns:
        The warped image.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    center = (cols / 2, rows / 2)

    M = cv.getRotationMatrix2D(center, degrees, 1.0)

    return cv.warpAffine(image, M, (cols, rows))


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


def warp_polar(image: np.ndarray, degree_factor: int = 3) -> np.ndarray:
    """
    Simple and slow polar warp.
    """
    rows_src = image.shape[0]
    cols_src = image.shape[1]

    channels = None if len(image.shape) < 3 else image.shape[2]
    center = np.array((cols_src / 2, rows_src / 2))

    rows_dst = 360 * degree_factor

    max_radius = np.linalg.norm(np.array((0.0, 0.0)) - center)
    cols_dst = round(max_radius) + 1

    dst = None
    if not channels is None:
        dst = np.zeros((rows_dst, cols_dst, channels), dtype=image.dtype)
    else:
        dst = np.zeros((rows_dst, cols_dst), dtype=image.dtype)

    for y in range(0, rows_src):
        for x in range(0, cols_src):
            point = np.array((x, y)) - center

            magnitude = np.linalg.norm(point)
            angle = math.degrees(math.atan2(point[1], point[0]))

            phi = round(magnitude)
            rho = round(angle * degree_factor)
            dst[rho, phi] = image[y, x]

    return dst
