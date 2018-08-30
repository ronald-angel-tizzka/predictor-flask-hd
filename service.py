from flask import Flask, request, jsonify

from config.config import MODEL_STORAGE_PATH
from model.model import ImageClassifier
from service.exceptions.generic_error import GenericError
from service.image_predictor import ImagePredictor
from service.image_trainer import ImageTrainer

# TODO: Use factory in case of config reading or injections needed.
app = Flask(__name__)


@app.route('/')
def hello():
    return 'Welcome to my Test: Ronald Angel'


@app.errorhandler(GenericError)
def invalid_request(error):
    """
    Converts an error to an error handler class.
    :param error: Error description.
    :return: Json with response.
    """
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


@app.route('/train_digits', methods=['GET', 'POST'])
def train_digits():
    """
    Service that calls the digits training method using the images path.
    :return: Response: Either valid model (True) or invalid (False) as json.
    """
    try:
        if request.json:
            request_data = request.json
            image_path = request_data.get("images_batch_path")
            classifier_model = ImageClassifier(training_path=image_path)
            image_trainer = ImageTrainer(classifier_model)
            training_response = image_trainer.train_digits()
            if training_response:
                return jsonify(training_response), 201
            else:
                return invalid_request(message="Wrong training set",
                                       status_code=400)
        else:
            return invalid_request(message="Source with wrong format",
                                   status_code=400)
    except ValueError:
        return invalid_request(message="Unexpected error during request.",
                               status_code=400)


@app.route('/process_digit', methods=['GET', 'POST'])
def process_digit():
    """
    Service that calls the digits prediction method for a single image.
    :return:
    """
    try:
        if request.json:
            request_data = request.json
            image_path = request_data.get("image_path")
            image_predictor = ImagePredictor(MODEL_STORAGE_PATH)
            predicted_digit = image_predictor.classify_digit(image_path)
            if predicted_digit:
                return jsonify(predicted_digit), 201
            else:
                return invalid_request(message="Wrong Image or not able to predict.",
                                       status_code=400)
        else:
            return invalid_request(message="Source with wrong format",
                                   status_code=400)
    except ValueError:
        return invalid_request(message="Unexpected error during request.",
                               status_code=400)


if __name__ == "__main__":
    app.run(host="0.0.0.0")
