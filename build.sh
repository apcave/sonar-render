#!/bin/bash
mkdir -p build
cd build
cmake ../src -DCMAKE_BUILD_TYPE=Release
#cmake ../src -DCMAKE_BUILD_TYPE=Debug
make all
