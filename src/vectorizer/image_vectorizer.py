import os
from typing import Tuple, List

import cv2
from keras import Sequential, models
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, \
    AveragePooling2D, GlobalAveragePooling2D, Dense, Dropout, SeparableConv2D
from keras.optimizers import Adam
from numpy import array, ndarray
from sklearn.model_selection import train_test_split

from preprocessing.encode_ascii import ASCIIEncoder


class ImageVectorizer:
    base_path = os.path.dirname(os.path.realpath(__file__))

    # https://www.kaggle.com/code/sujoykg/keras-cnn-with-grayscale-images
    def __init__(self):
        self.model = Sequential()

    def build_model(self) -> None:
        # for grayscale:
        # self.model.add(Conv2D(32, (5, 5), strides=(1, 1), name='conv0', input_shape=(128, 128, 3)))
        self.model.add(
            SeparableConv2D(
                filters=200,
                kernel_size=3,
                dilation_rate=3,
                strides=3,
                activation='relu',
                input_shape=(128, 128, 3),
                padding="VALID",
                name="conv0"
            )
        )

        self.model.add(BatchNormalization(axis=3, name='bn0'))
        self.model.add(Activation('relu'))

        self.model.add(MaxPooling2D((2, 2), name='max_pool'))
        self.model.add(Conv2D(256, (3, 3), strides=(1, 1), name="conv1"))
        self.model.add(Activation('relu'))
        self.model.add(AveragePooling2D((3, 3), name='avg_pool'))

        self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(512, activation="relu", name='rl'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(ASCIIEncoder.VECTOR_LENGTH, activation='sigmoid', name='sm'))
        self.model.compile(loss='binary_crossentropy',
                           optimizer=Adam(lr=1e-5),
                           metrics=['accuracy'])

    def train(self,
              image_folder: str,
              sample_folder: str):
        x, y = self._read_data(image_folder, sample_folder)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
        self.model.fit(
            x,
            y,
            epochs=500,
            validation_data=(x_train, y_train),
        )

    def save_model(self, model_folder: str, is_rel_folder: bool = True) -> None:
        if is_rel_folder:
            model_folder = os.path.join(self.base_path, "models", model_folder)
        self.model.save(model_folder)

    def load_model(self, model_folder: str, is_rel_folder: bool = True) -> None:
        if is_rel_folder:
            model_folder = os.path.join(self.base_path, "models", model_folder)
        self.model = models.load_model(model_folder)

    def vectorize_image(self, image_path: str) -> ndarray:
        img_data = cv2.imread(image_path)
        x = array([img_data])
        data = self.model.predict(x)
        return data

    def _read_data(self, image_folder: str, sample_folder: str) -> Tuple[array, array]:
        def get_paths(folder: str) -> List[str]:
            return [file_path.path for file_path in
                    [entry for entry in os.scandir(folder) if entry.is_file()]]

        x_paths = get_paths(image_folder)
        y_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in get_paths(sample_folder)}
        paths: List[Tuple[str, str]] = []
        for x_path in x_paths:
            x_name = os.path.splitext(os.path.basename(x_path))[0]
            y_path = y_paths.get(x_name)
            if not y_path:
                continue
            paths.append((x_path, y_path))

        # read images and data
        x, y = [], []
        for x_path, y_path in paths:
            img_data = cv2.imread(x_path)
            # img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
            # img_data = np.expand_dims(img_data, axis=2)
            x.append(img_data)

            with open(y_path, "r") as f:
                text = f.read()
            data = [float(c) for c in text]
            y.append(data)
        print("Data is loaded")
        return array(x), array(y)
