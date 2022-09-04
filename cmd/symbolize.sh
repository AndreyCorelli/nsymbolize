#!/bin/bash

to_abs_path() {
    python -c "import os; print(os.path.abspath('$1'))"
}
src_dir=$(to_abs_path "../data/cln_images/*.*")

for src_filename in ${src_dir}; do
    name_only=${src_filename##*/}
    name_only=$(sed 's/\.[^.]*$//' <<< "${name_only}")
    dst_filename="../data/ascii_files/${name_only}.txt"
    if [ ! -f $dst_filename ]
    then
        # convert to ASCII
        echo "${src_filename}"
        # !remove this line!
        cp "${src_filename}" /<mylocalpath>/convert_1.png
        # !remove this line and then replace /home/andrey/convert_1.png with "${src_filename}"
        ascii-image-converter /<mylocalpath>/convert_1.png -d 160,80 > "${dst_filename}"
    else
        echo "File ${dst_filename} exists"
    fi
done
