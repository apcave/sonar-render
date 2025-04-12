#!/bin/bash

cd ./src/host
rm cuda_project.so
make all
cd ../..
python ./testing/load_stl_to_cuda.py