import pandas as pd
import cv2
import numpy as np
import os


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        if (len(images) % 600 == 0):
            print(len(images) / 600)
    return images

def get_train():
    y_train = pd.read_csv("train.csv", usecols=[1]).to_numpy()
    x_train = np.array(load_images_from_folder("train"))

    return x_train, y_train


if __name__ == "__main__":
    x_train, y_train = get_train()
    cv2.imshow(x_train[0])
    print(y_train[0])
    cv2.show()
