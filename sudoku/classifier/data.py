import numpy as np
import cv2
from scipy import ndimage
from keras.datasets import mnist
import os
import pickle

def _set_digit(digit: int, fontScale=1.5, thickness=3, angle=0, lineType=80, font=cv2.FONT_HERSHEY_SIMPLEX):
    d1 = np.zeros((50, 50)).astype("uint8")
    d1 = cv2.putText(d1, str(digit), (4, 46), fontFace=font, fontScale=fontScale, color=255, thickness=thickness, lineType=lineType)
    #rotated = imutils.rotate(d1, angle)
    d1 = ndimage.rotate(d1, angle)
    #cv2.imshow("d1", d1)
    #cv2.waitKey(0)
    contours, _ = cv2.findContours(d1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    pad = (h-w)//2
    digit = np.zeros((h, h))
    digit[:, pad:pad + w] = d1[y:y + h, x:x + w]
    d1 = cv2.resize(digit, (28, 28))
    return d1


def _generate_data():
    x = []
    y = []
    for i in range(1, 10):
        for angle in [0]:#4, -4, -2):
            for thickness in [1, 2, 3]: #range(1, 3):
                for scale in range(70, 200, 10):
                    for line in [4, 8]:
                        x.append(_set_digit(i, thickness=thickness, fontScale=scale / 100, angle=angle, lineType=line, font=cv2.FONT_HERSHEY_SIMPLEX))
                        y.append(i)
                        x.append(_set_digit(i, thickness=thickness, fontScale=scale / 100, angle=angle, lineType=line, font=cv2.FONT_HERSHEY_PLAIN))
                        y.append(i)
                        x.append(_set_digit(i, thickness=thickness, fontScale=scale / 100, angle=angle, lineType=line, font=cv2.FONT_HERSHEY_DUPLEX))
                        y.append(i)
                        x.append(_set_digit(i, thickness=thickness, fontScale=scale / 100, angle=angle, lineType=line, font=cv2.FONT_HERSHEY_COMPLEX))
                        y.append(i)
                        x.append(_set_digit(i, thickness=thickness, fontScale=scale / 100, angle=angle, lineType=line, font=cv2.FONT_HERSHEY_TRIPLEX))
                        y.append(i)
                        x.append(_set_digit(i, thickness=thickness, fontScale=scale / 100, angle=angle, lineType=line, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX))
                        y.append(i)
                        x.append(_set_digit(i, thickness=thickness, fontScale=scale / 100, angle=angle, lineType=line, font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX))
                        y.append(i)
                        x.append(_set_digit(i, thickness=thickness, fontScale=scale / 100, angle=angle, lineType=line, font=cv2.FONT_HERSHEY_COMPLEX_SMALL))
                        y.append(i)
    return x, y


def _prepare_artificial_data(x, y):
    x_np = np.empty((0, 1, 28, 28))
    y_np = np.array(y)
    for _x in x:
        x_np = np.concatenate((x_np, _x.reshape(1, 1, 28, 28)))
    return x_np, y_np.reshape((len(y), ))


def _plot_data(images):
    rows =[]
    max_rows = len(images) // 9
    for nr in range(9):
        row = images[nr*max_rows]
        for idx in range(1, max_rows):
            row = np.concatenate((row, images[nr*max_rows + idx]), axis=1)
        rows.append(row)
    res = rows[0]
    for r in rows[1:]:
        res = np.concatenate((res, r), axis=0)
    print(F"Generated {len(images)} Images")
    cv2.imshow("res", res)
    cv2.waitKey(0)


def get_mnist_data():
    seed = 7  # Fix random seed for reproducibility
    np.random.seed(seed)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Reshape to be samples*pixels*width*height
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
    # Normalize inputs from 0-255 to 0-1
    x_train = x_train / 255
    x_test = x_test / 255
    print(F"Shape mnist train/test data: {x_train.shape}, {y_train.shape}")
    return x_train, y_train, x_test, y_test


def get_all_data(from_file="pickle_data.p"):
    if os.path.isfile(from_file):
        x_train, y_train, x_test, y_test = pickle.load(open(from_file, "rb"))
    else:
        x, y = _generate_data()
        x_np, y_np = _prepare_artificial_data(x, y)
        print(F"Shape artifical train data: {x_np.shape}, {y_np.shape}")
        x_train, y_train, x_test, y_test = get_mnist_data()
        x_train = np.concatenate((x_train, x_np), axis=0)  # append artificial data
        y_train = np.concatenate((y_train, y_np), axis=0)
        pickle.dump([x_train, y_train, x_test, y_test], open(from_file, "wb"))
    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_all_data()

    #x, y = _generate_data()
    #_plot_data(x)
