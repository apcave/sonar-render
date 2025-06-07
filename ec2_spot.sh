#!/bin/bash



sudo apt update
sudo apt install libepoxy-dev libglm-dev

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r ./requirements.txt