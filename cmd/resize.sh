#!/bin/bash
convert '../data/src_images/*.*[128x128!]' -set filename:base "%[basename]" "../data/szd_images/%[filename:base].jpg"