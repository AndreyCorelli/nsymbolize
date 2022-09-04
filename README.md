## Set up the project
This project is build in Python 3.8.10 virtual environment. Check that your video card drivers
are actual and CUDA is installed. CUDA should not conflict with Pytorch.

Most probably you'll have to install specific version of Pytorch for you CUDA version with
a command like

```shell
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

The project requires the library `backgroundremover==0.1.9`. Most probably the library will require a
different `pytorch` version and there'll be no chance to make this package run inside the project's
venv. The solution is to:

1. open / create another project
2. install `backgroundremover==0.1.9` with all the requirements
3. copy & paste the file `cmd/remove_background.sh` in this project, change the paths respectively and run the scrip in the new project's venv. 

## Vectorize an image

The model used here in example code includes one large file: 
`src/vectorizer/models/model_01-09-2022/variables/variables.data-00000-of-00001`. The file is compressed.
To uncompress the file run a command
```shell
cd cmd && ./unpack_model.sh
```

```python
from preprocessing.encode_ascii import ASCIIEncoder
from vectorizer.image_vectorizer import ImageVectorizer

# image should have 128x128 size
imv = ImageVectorizer()
imv.load_model("<model folder>")
vector = imv.vectorize_image("<source image path>")
text = ASCIIEncoder().decode_vector(vector)
text_line = "\n".join(["".join(l) for l in text])
with open("<output text file path>", 'w') as f:
    f.write(text_line)
```

## Train and save a model

```python
from vectorizer.image_vectorizer import ImageVectorizer

imv = ImageVectorizer()
imv.build_model()
imv.train("<source image folder, all images are 128x128>", 
          "<folder with text files, matching with images by names>")
imv.save_model("<folder to store the model>")
```

## Prepare the data

Source data images may have any format that `cv2.imread()` can read. Colorful images are expected.
Source image folder is `data/src_images`.

### Collect images from Google

To collect images from Google do the following:
1. open Google images, scroll to the bottom to show more images on the page
2. open the Google Chrome console, copy & paste there the script from `src/data_collection/google_js.txt`
3. there's a new text file `urls.txt` with the images' hyperlinks.
4. repeat steps 1 - 3 to collect several files `urls.txt, urls(1).txt, ...`. Combine the files' content into the single file `src/data_collection/urls.txt`.
5. run the script `python src/data_collection/download_google_images.py`

### Make text files out of images

First, remove background from **original** images: `cmd/remove_background.sh`. The script will 
read the source files from `data/src_images` and store the output `data/cln_images`. The script will skip
all the files that are already processed. Processing 2000 files takes ~ 3 hours.

Second, convert the files from the `data/cln_images` folder with the `cmd/symbolize.sh` command. The command
will also skip the files that are already converted. The output (Y for training) will be the files in the 
`data/ascii_files` folder. Each file has 160 columns and 80 lines as it was set up in the script.

Finally, resize the images from the `data/src_images` folders to use them as the training input (X) with the
`cmd/resize.sh` command. The result will be stored in the `data/ascii_files` folder.
