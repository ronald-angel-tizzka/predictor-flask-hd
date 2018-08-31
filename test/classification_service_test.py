import unittest

import numpy as np

from service.image_predictor import ImagePredictor
from service.image_trainer import ImageTrainer


class ClassificationServiceTest(unittest.TestCase):
    """
    Test for the main classification/training service methods.
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
        """
        Test a single result for a specif digit test file.
        :return: assert with expected value.
        """
        expected_predicted_digit = "5"
        test_model_path = "data/dummy_model.h5"
        test_img_path = "data/n5.png"

        digits_predictor = ImagePredictor(test_model_path)
        predicted_array = digits_predictor.classify_digit(test_img_path)
        predicted_digit = np.array_str(np.argmax(predicted_array, axis=1))
        self.assertTrue(expected_predicted_digit in predicted_digit)


if __name__ == '__main__':
    unittest.main()
