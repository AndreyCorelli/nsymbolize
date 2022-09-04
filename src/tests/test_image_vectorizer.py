import datetime
import os

from paths import SZD_IMG_PATH, SAMPLE_OUTPUT_PATH
from preprocessing.encode_ascii import ASCIIEncoder
from vectorizer.image_vectorizer import ImageVectorizer


def test_vectorize():
    imv = ImageVectorizer()
    imv.build_model()
    imv.train(SZD_IMG_PATH, SAMPLE_OUTPUT_PATH)
    imv.save_model(f"model_{datetime.datetime.now().strftime('%d-%m-%Y')}")


def test_symbolyze():
    imv = ImageVectorizer()
    imv.load_model("model_01-09-2022")

    src_path = os.path.join(SZD_IMG_PATH, "00000236.jpg")
    out_path = '/home/andrey/Documents/smb01.txt'
    vector = imv.vectorize_image(src_path)
    text = ASCIIEncoder().decode_vector(vector)
    text_line = "\n".join(["".join(l) for l in text])

    with open(out_path, 'w') as f:
        f.write(text_line)

