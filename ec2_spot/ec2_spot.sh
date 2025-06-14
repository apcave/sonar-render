#!/bin/bash

mkdir -p ~/.aws
mv config ~/.aws/config
mv credentials ~/.aws/credentials

sudo apt update
sudo apt install -y libepoxy-dev libglm-dev nvtop

git clone https://github.com/apcave/sonar-render.git
cd sonar-render

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r ./requirements.txt

mkdir build
cd build
cmake ../src/
make -j5 all

echo "source ~/sonar-render/venv/bin/activate" >>  ~/.bashrc
