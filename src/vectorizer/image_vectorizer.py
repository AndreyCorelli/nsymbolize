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
    IMAGE_SIZE = 128

    base_path = os.path.dirname(os.path.realpath(__file__))

    # https://www.kaggle.com/code/sujoykg/keras-cnn-with-grayscale-images
    def __init__(self):
        self.model = Sequential()

    def build_model(self) -> None:
        # build "filter" feature levels from the original image
        self.model.add(
            SeparableConv2D(
                filters=64,  # count of filters (kernels)
                kernel_size=3,  # kernel (matrix) size
                dilation_rate=3,  # extra pixel spacing for the kernels (0..N)
                strides=3,  # stride for the kernel (1..N)
                activation='relu',  # gold standard
                input_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, 3),
                padding="VALID",  # "valid" means no padding
                name="conv0"
            )
        )

        # normalize scalars to 0..1
        self.model.add(BatchNormalization(axis=3, name='bn0'))
        self.model.add(Activation('relu'))

        """
        Downsamples the input along its spatial dimensions (height and width) by taking
        the maximum value over an input window (of size defined by pool_size ) for each channel
        of the input. The window is shifted by strides along each dimension.
        """
        self.model.add(MaxPooling2D((2, 2), name='max_pool'))
        self.model.add(Conv2D(128, (3, 3), strides=(1, 1), name="conv1"))
        self.model.add(Activation('relu'))
        self.model.add(AveragePooling2D((3, 3), name='avg_pool'))

        self.model.add(GlobalAveragePooling2D())

        # a "norman" NN layer
        self.model.add(Dense(512, activation="relu", name='rl'))

        # a layer introduced to reduce over-fitting ("noise"-fitting)
        # it makes L2 normalization non-necessary
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
