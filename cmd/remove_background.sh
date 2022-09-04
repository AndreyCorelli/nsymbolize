#!/bin/bash

for src_filename in ../data/src_images/*.*; do
    name_only=${src_filename##*/}
    name_only=$(sed 's/\.[^.]*$//' <<< "${name_only}")
    dst_filename="../data/cln_images/${name_only}.png"
    if [ ! -f $dst_filename ]
    then
        # remove background
        backgroundremover -i "${src_filename}" -o "${dst_filename}"
    else
        echo "File ${dst_filename} exists"
    fi
done