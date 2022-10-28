import argparse
import logging
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import register.image.phase as phase
import register.image.util as util

logger = None
handler = None


def setup_logging() -> None:
    """
    Global setup of logging system. Module loggers then register
    as getLogger(__name__) to end up in logger tree.
    """
    global logger
    logger = logging.getLogger('register')
    logger.setLevel(logging.DEBUG)

    global handler
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)

    formatter = logging.Formatter(
        '[%(levelname)s %(name)s:%(lineno)d] %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)


def display_magnitude_spectrum(path: str, hanning: bool) -> None:
    """
    Display the magnitude spectrum for an image.
    """
    logger.debug(f'display magnitude spectrum path={path} hanning={hanning}')

    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if image is None:
        logger.error(f'Failed to read image')
        return None

    magnitude = util.log_magnitude_spectrum(np.float32(image), hanning)

    fig = plt.figure("Magnitude Spectrum")

    sub1 = fig.add_subplot(1, 2, 1)
    sub1.set_title('Grayscale Image')
    plt.imshow(image, cmap='gray')

    sub2 = fig.add_subplot(1, 2, 2)
    sub2.set_title('Magnitude Spectrum')
    plt.imshow(magnitude, cmap='hot')

    plt.show()


def display_subimage_phase_correlation(path: str, xstart: int, ystart: int, subsize: int, hanning: bool) -> None:
    """
    Display subimage phase correlation.
    """
    logger.debug(
        f'display subimage phase correlation path={path} xstart={xstart} ystart={ystart} subsize={subsize} hanning={hanning}')

    image = cv.imread(path, cv.IMREAD_COLOR)
    if image is None:
        logger.error(f'Failed to read image')
        return None

    # Pick the subimage to register.
    subimage = util.subimage(image, xstart, ystart, subsize, subsize).copy()

    # Draw ground truth rectangle in red.
    cv.rectangle(image, (xstart, ystart), (xstart +
                 subsize, ystart + subsize), (0, 0, 255))

    # Gray convert for phase correlation.
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_subimage = cv.cvtColor(subimage, cv.COLOR_BGR2GRAY)

    # Run phase correlation.
    corr_map = phase.correlate(np.float32(gray_image),
                               np.float32(gray_subimage), hanning)

    # Get the weighted peak value.
    peak_x, peak_y = phase.peak_location(corr_map)
    peak_x = round(peak_x)
    peak_y = round(peak_y)

    # Draw registration rectangle in green.
    cv.rectangle(image, (peak_x, peak_y), (peak_x +
                 subsize, peak_y + subsize), (0, 255, 0))

    # RGB for visualization.
    bi, gi, ri = cv.split(image)
    rgbimage = cv.merge([ri, gi, bi])

    bs, gs, rs = cv.split(subimage)
    rgbsubimage = cv.merge([rs, gs, bs])

    fig = plt.figure("Phase Correlation")

    sub1 = fig.add_subplot(1, 3, 1)
    sub1.set_title('Full Image')
    plt.imshow(rgbimage)

    sub2 = fig.add_subplot(1, 3, 2)
    sub2.set_title('Sub Image')
    plt.imshow(rgbsubimage)

    sub3 = fig.add_subplot(1, 3, 3)
    sub3.set_title('Correlation Map')
    plt.imshow(corr_map, cmap='gray')

    plt.show()


def main() -> None:
    """
    Entry point for the register execution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log',
                        help='set the effective log level (DEBUG, INFO, WARNING or ERROR)')
    parser.add_argument('--magnitude-spectrum', type=str,
                        help='Display magnitude spectrum for the given image')
    parser.add_argument('--subimage-pcorr', type=str,
                        help='Display phase correlation for subimage')
    parser.add_argument('--hanning', action='store_true',
                        help='Apply Hanning window')
    parser.add_argument('--xstart', type=int, default=0,
                        help='x value for subimage')
    parser.add_argument('--ystart', type=int, default=0,
                        help='y value for subimage')
    parser.add_argument('--subsize', type=int, default=100,
                        help='size of subimage square')
    args = parser.parse_args()

    # Check if the effective log level shall be altered.
    if not args.log is None:
        log_level = args.log.upper()
        if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            num_log_level = getattr(logging, log_level)
            handler.setLevel(num_log_level)
        else:
            parser.print_help()
            sys.exit(1)

    if not args.magnitude_spectrum is None:
        display_magnitude_spectrum(args.magnitude_spectrum, args.hanning)
    elif not args.subimage_pcorr is None:
        display_subimage_phase_correlation(
            args.subimage_pcorr, args.xstart, args.ystart, args.subsize, args.hanning)
    else:
        parser.print_help()

    # Successful exit.
    sys.exit(0)


if __name__ == '__main__':
    setup_logging()
    try:
        main()
    except Exception as e:
        logger.exception(f"Global exception handler caught: '{e}'")
        sys.exit(1)
