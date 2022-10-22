import argparse
import logging
import sys

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

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


def show_magnitude_spectrum(path: str, hanning: bool) -> None:
    """
    Display the magnitude spectrum for an image.
    """
    logger.debug(f'show_magnitude path={path} hanning={hanning}')

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


def main() -> None:
    """
    Entry point for the register execution.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log',
                        help='set the effective log level (DEBUG, INFO, WARNING or ERROR)')
    parser.add_argument('--magnitude-spectrum',
                        help='Show magnitude spectrum for the given image')
    parser.add_argument('--hanning', action='store_true',
                        help='Apply Hanning window')
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
        show_magnitude_spectrum(args.magnitude_spectrum, args.hanning)

    # Successful exit.
    sys.exit(0)


if __name__ == '__main__':
    setup_logging()
    try:
        main()
    except Exception as e:
        logger.exception(f"Global exception handler caught: '{e}'")
        sys.exit(1)
