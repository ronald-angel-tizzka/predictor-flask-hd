import logging
import sys


class ImageTrainer(object):
    """Class to expose stateless methods for training to the service."""

    def __init__(self, model):
        """Injecting  model dependency."""
        self.model = model

    def train_digits(self):
        """Calls the train method from the model."""
        try:
            # TODO: Make decision taking validation into account validation
            metrics_result = self.model.train()
            logging.info("model performance is {}".format(metrics_result))
            return metrics_result is not None
        # TODO: Apply specific exceptions and log,
        except:
            logging.error("Prediction Error:", sys.exc_info()[0])
            raise ValueError()
