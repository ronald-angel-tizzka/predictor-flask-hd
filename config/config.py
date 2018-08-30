# TODO: Read this config from external file depending on the env.
BATCH_SIZE = 200
NUM_CLASSES = 10
EPOCHS = 10
MODEL_STORAGE_PATH = "digits_model.h5"  # TODO: path from HDFS.
# input image dimensions
IMG_ROWS, IMG_COLS = 28, 28
IMAGE_BATCH_PARAM_NAME = "images_batch_path"
SINGLE_IMAGE_PARAM_NAME = "image_path"
