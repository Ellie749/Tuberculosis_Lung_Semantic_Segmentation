import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import tensorflow as tf
from dataset import load_data
from sklearn.model_selection import train_test_split
from dataset.data_preprocess import data_cleansing
from network.unet_architecture import Unet
from model.train_model import train
from visualization import utils

print(tf.config.list_physical_devices(device_type='GPU'))

CHEST_DATA_PATH = 'data\Chest-X-Ray\Chest-X-Ray\image' 
MASK_DATA_PATH = 'data\Chest-X-Ray\Chest-X-Ray\mask'
INPUT_SHAPE = (128, 128, 3)
N_CLASSES = 1
EPOCHS = 10
BATCH_SIZE = 64


def main():
    images = load_data.load_chest_images(CHEST_DATA_PATH)
    masks = load_data.load_mask_images(MASK_DATA_PATH)
    images = data_cleansing(images)
    masks = data_cleansing(masks, is_mask=True)
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2)
    print(f"X_train dimensions: {X_train.shape}")
    print(f"X_test dimensions: {X_test.shape}")
    print(f"y_train dimensions: {y_train.shape}")
    print(f"y_test dimensions: {y_test.shape}")

    UNet = Unet(INPUT_SHAPE, N_CLASSES)
    model = UNet.build()
    print(model.summary())
    H = train(model, X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE)
    utils.plot_metrics(H)


if __name__ == '__main__':
    main()