import cv2
import numpy as np


class ImageProcessor:
    """Processor for processing image."""

    def __init__(self, width, height, interpolation=cv2.INTER_AREA):
        """Inits ImageProcessor instance.

        Args:
            width: The target width of input image after resizing.
            height: The target height of input image after resizing.
            interpolation: An optional parameter used to control
                which interpolation algorithm is used when resizing.
        """
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.interpolation = interpolation

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        # resize the image to a fixed size,
        # ignoring the aspect ratio
        return cv2.resize(image, (self.width, self.height),
                          interpolation=self.interpolation)
