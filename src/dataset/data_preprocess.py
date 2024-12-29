import cv2
import numpy as np

def data_cleansing(data, is_mask=False):
    images = []
    if is_mask==True:
        for image in data:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY) # the values of the number swould be your last channel (0 or 255)
            img = cv2.resize(img, (128, 128))
            images.append(img)

    else:
        for image in data:
            img = cv2.resize(image, (128, 128))
            images.append(img)

    return (np.array(images))

    



