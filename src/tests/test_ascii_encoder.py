from src.paths import ASCII_PATH, SAMPLE_OUTPUT_PATH
from src.preprocessing.encode_ascii import ASCIIEncoder


def test_char_range():
    ASCIIEncoder().determine_codes_from_files(ASCII_PATH)


def test_encode():
    ASCIIEncoder().encode_files_in_folder(ASCII_PATH, SAMPLE_OUTPUT_PATH)
