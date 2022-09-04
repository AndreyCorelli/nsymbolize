import os

BASE_PATH = os.path.dirname(os.path.realpath(__file__))

DATA_PATH = os.path.join(BASE_PATH, '..', 'data')

DATA_COLLECTION_PATH = os.path.join(BASE_PATH, '..', 'src', 'data_collection')

SRC_IMG_PATH = os.path.join(DATA_PATH, 'src_images')

SZD_IMG_PATH = os.path.join(DATA_PATH, 'szd_images')

ASCII_PATH = os.path.join(DATA_PATH, 'ascii_files')

SAMPLE_OUTPUT_PATH = os.path.join(DATA_PATH, 'sample_output')
