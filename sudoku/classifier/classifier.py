import os
from keras.models import model_from_json
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


class DigitClassifier:
    def __init__(self):
        self._model = None

    def load(self, model_dir, model_structure="model.json", model_data="model.h5"):
        model_structure_path = os.path.join(model_dir, model_structure)
        model_data_path = os.path.join(model_dir, model_structure)
        if os.path.isfile(model_structure_path) and os.path.isfile(model_data_path):
            with open(model_structure_path, 'r') as file:
                json_content = file.read()
            self._model = model_from_json(json_content)
            self._model.load_weights(os.path.join(model_dir, model_data))
        else:
            raise FileNotFoundError(F"Model files not found: {model_structure_path} and {model_data_path}")

    def train(self, x_train, y_train, x_test, y_test):
        K.image_data_format()
        # One Hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        num_classes = y_test.shape[1]
        # Create model
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu', data_format='channels_first'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200)
        self._model = model

    def save(self, model_dir="model", model_structure="model.json", model_data="model.h5"):
        model_structure_path = os.path.join(model_dir, model_structure)
        model_data_path = os.path.join(model_dir, model_data)
        model_json = self._model.to_json()
        with open(model_structure_path, "w") as json_file:
            json_file.write(model_json)  # serialize model to JSON
        self._model.save_weights(model_data_path)  # serialize weights to HDF5

    def get_error(self, x_test, y_test):
        if self._model is not None:
            scores = self._model.evaluate(x_test, y_test, verbose=0)
            return 100 - scores[1] * 100
        else:
            raise Exception("Model does not exist. Please load a model first.")
        return 0

    def predict(self, image):
        if self._model is not None:
            resized = cv2.resize(image, (28, 28))
            reshaped = resized.reshape(1, 1, 28, 28)
            prediction = self._model.predict_classes(reshaped, verbose=0)
            return prediction[0]
        else:
            raise Exception("Model does not exist. Please load a model first.")
        return 0


if __name__ == '__main__':
    import data
    model_dir = "models"
    x_train, y_train, x_test, y_test = data.get_all_data()
    #x_train, y_train, x_test, y_test = get_all_data()
    print(F"train/test shape: {x_train.shape}/{x_test.shape}")
    classifier = DigitClassifier()

    train = True
    if train:
        classifier.train(x_train, y_train, x_test, y_test)
        classifier.save(model_dir)
    else:
        classifier.load(model_dir)
        for idx in range(10):
            image = x_test[idx][0]
            result = classifier.predict(image)
            print(F"Prediction is: {result}")
            cv2.imshow("test image", image)
            cv2.waitKey(0)
