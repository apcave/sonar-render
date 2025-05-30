#!/bin/bash
mkdir -p build
cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release
#cmake ../src -DCMAKE_BUILD_TYPE=Debug

rm libcuda_project.so
make all
cd ..
#python ./testing/sphere.py
#python ./testing/rect_bistatic.py
#python ./testing/rect_mono.py
#python ./testing/cube.py
python ./testing/reflector.py