import requests
import cv2
import os

from paths import DATA_COLLECTION_PATH, SRC_IMG_PATH


def download_images():
    def get_file_paths(folder: str):
        return [entry for entry in os.scandir(folder) if entry.is_file()]

    input_file = os.path.join(DATA_COLLECTION_PATH, "urls.txt")
    output_path = SRC_IMG_PATH

    rows = open(input_file).read().strip().split("\n")

    # find the last index from file names
    existing = get_file_paths(output_path)
    last_index = -1
    for file_path in existing:
        file_name = os.path.splitext(os.path.basename(file_path.path))[0]
        number = int(file_name)
        if number > last_index:
            last_index = number
    last_index += 1

    # loop the URLs
    for url in rows:
        try:
            # try to download the image
            r = requests.get(url, timeout=60)
            # save the image to disk
            p = os.path.sep.join([output_path, "{}.jpg".format(
                str(last_index).zfill(8))])
            f = open(p, "wb")
            f.write(r.content)
            f.close()
            # update the counter
            print("[INFO] downloaded: {}".format(p))
            last_index += 1
        # handle if any exceptions are thrown during the download process
        except:
            print("[INFO] error downloading {}...skipping".format(p))

    # loop over the image paths we just downloaded
    for file_path in get_file_paths(output_path):
        image_path = file_path.path
        # initialize if the image should be deleted or not
        delete = False
        # try to load the image
        try:
            image = cv2.imread(image_path)
            # if the image is `None` then we could not properly load it
            # from disk, so delete it
            if image is None:
                delete = True
        # if OpenCV cannot load the image then the image is likely
        # corrupt so we should delete it
        except:
            print("Except")
            delete = True
        # check to see if the image should be deleted
        if delete:
            print(f"[INFO] deleting {image_path}")
            os.remove(image_path)


if __name__ == '__main__':
    download_images()
