#!/bin/bash
mkdir -p build
cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release
#cmake ../src -DCMAKE_BUILD_TYPE=Debug

rm libcuda_project.so
make -j5 all
cd ..
#python ./run_scripts/sphere.py
#python ./run_scripts/rect_bistatic.py
#python ./run_scripts/rect_mono.py
#python ./run_scripts/cube.py
python ./run_scripts/reflector2.py
#python ./run_scripts/BeTTSi_run1.py