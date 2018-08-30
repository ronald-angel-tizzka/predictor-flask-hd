import keras
import numpy as np
from PIL import Image
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

from config.config import BATCH_SIZE, NUM_CLASSES, EPOCHS, IMG_ROWS, IMG_COLS, MODEL_STORAGE_PATH


class ImageClassifier:
    """
    Trains a simple Convnet on the MNIST dataset.
    """

    def __init__(self, training_path):
        """
        Creates the model configuration.
        :param training_path: Sample batch path.
        """
        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data(path=training_path)

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, IMG_ROWS, IMG_COLS)
            x_test = x_test.reshape(x_test.shape[0], 1, IMG_ROWS, IMG_COLS)
            input_shape = (1, IMG_ROWS, IMG_COLS)
        else:
            x_train = x_train.reshape(x_train.shape[0], IMG_ROWS, IMG_COLS, 1)
            x_test = x_test.reshape(x_test.shape[0], IMG_ROWS, IMG_COLS, 1)
            input_shape = (IMG_ROWS, IMG_COLS, 1)

        # create a sequential convnet model in Keras
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        # define loss, optimizer and evaluation metric
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train(self):
        """
        Train the model using loaded data and save to external checkpoint (zk in future).
        :return: Training result evaluated against a threshold.
        """
        # convert types
        x_train = self.x_train.astype('float32')
        x_test = self.x_test.astype('float32')

        # scale
        x_train /= 255
        x_test /= 255

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(self.y_train, NUM_CLASSES)
        y_test = keras.utils.to_categorical(self.y_test, NUM_CLASSES)

        # finally fit the model on the data
        self.model.fit(x_train, y_train,
                       batch_size=BATCH_SIZE,
                       epochs=EPOCHS,
                       verbose=1,
                       validation_data=(x_test, y_test))
        self.model.save(MODEL_STORAGE_PATH)
        model_metrics = self.model.evaluate(x_test, y_test, verbose=0)
        return model_metrics

    @staticmethod
    def predict(path_image, model):
        """
        Predict class from the features loading module from external source.
        :param path_image: Path to single image for prediction.
        :param model: Injected reference for the model.
        :return: Array with closest classes predicted.
        """
        # TODO: Convert mutability to mapping pipe functional like
        img_loaded = Image.open(path_image).convert("L")
        img_loaded = img_loaded.resize((28, 28))
        img_arr = np.array(img_loaded)
        img_arr = img_arr.reshape(1, 28, 28, 1)
        predicted_classes = model.predict(img_arr)
        return predicted_classes
