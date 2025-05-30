#!/bin/bash
wget https://developer.download.nvidia.com/compute/cuda/repos/$(lsb_release -cs)/x86_64/cuda-$(lsb_release -cs).pin
sudo mv cuda-$(lsb_release -cs).pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/repos/$(lsb_release -cs)/x86_64/cuda-repo-$(lsb_release -cs)_<version>_amd64.deb
sudo dpkg -i cuda-repo-$(lsb_release -cs)_<version>_amd64.deb
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$(lsb_release -cs)/x86_64/7fa2af80.pub
sudo apt update

nvcc --version

sudo apt update
sudo apt install libglew-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev

sudo docker build -t sonar-render .

sudo docker run -it --entrypoint /bin/bash sonar-render