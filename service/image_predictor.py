import logging
import sys

import numpy as np
from keras.models import load_model

from model.model import ImageClassifier


class ImagePredictor(object):
    """Class to expose stateless methods for classification to the service."""

    def __init__(self, model_path):
        """Injecting  model dependency."""
        self.preload_model = load_model(model_path)

    def classify_digit(self, digit_img_path):
        """
        Calls the classification method from the model.
        :param digit_img_path: Source image path.
        :return: Predicted classes array.
        """
        try:
            prediction_classes_response = ImageClassifier.predict(digit_img_path, self.preload_model)
            return np.array_str(np.argmax(prediction_classes_response, axis=1))
        # TODO: Apply specific exceptions (custom) and log.
        except:
            logging.error("Prediction Error:", sys.exc_info()[0])
            raise ValueError()
