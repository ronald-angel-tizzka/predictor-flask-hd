import unittest

import numpy as np

from service.image_predictor import ImagePredictor
from service.image_trainer import ImageTrainer


class ClassificationServiceTest(unittest.TestCase):
    """
    Test for the main service methods.
    """

    # TODO: Complete the test with border cases for services.
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_predictor_wrong_message(self):
        image_predictor = ImagePredictor(None)
        self.assertRaises(ValueError, image_predictor.classify_digit(None))

    def test_trainer_wrong_message(self):
        image_trainer = ImageTrainer(None)
        self.assertRaises(ValueError, image_trainer.train_digits())

    def test_single_prediction_result(self):
        predictor = ImagePredictor("data/dummy_model.h5")
        predicted_array = predictor.classify_digit("data/n5.png")
        predicted_digit = np.array_str(np.argmax(predicted_array, axis=1))
        self.assertEqual(predicted_digit, '5')


if __name__ == '__main__':
    unittest.main()
