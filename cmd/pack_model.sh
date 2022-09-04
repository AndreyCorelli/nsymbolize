#!/bin/bash

cd ../src/vectorizer/models/model_01-09-2022/variables/ && zip variables.zip ./variables.data-00000-of-00001 && split -b 45M variables.zip variables_parts