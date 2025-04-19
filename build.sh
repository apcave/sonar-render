#!/bin/bash
mkdir -p build
cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release
#cmake ../src -DCMAKE_BUILD_TYPE=Debug

rm libcuda_project.so
make all
cd ..
python ./testing/load_stl_to_cuda.py