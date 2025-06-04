#!/bin/bash
sudo apt install -y docker

git clone https://github.com/apcave/sonar-render.git
sudo yum install -y mesa-libGL-devel mesa-libGLU-devel libX11-devel libXrandr-devel libXcursor-devel libXi-devel
sudo yum install -y glew-devel
sudo yum install -y libXinerama-devel
sudo yum install -y ncurses-devel
sudo yum install -y xorg-x11-server-Xvfb
sudo yum install -y mesa-demos

cd /tmp
wget https://github.com/glfw/glfw/releases/download/3.3.8/glfw-3.3.8.zip
unzip glfw-3.3.8.zip
cd glfw-3.3.8
cmake -DBUILD_SHARED_LIBS=ON .
make
sudo make install


cd /tmp
git clone https://github.com/g-truc/glm.git
cd glm
mkdir build && cd build
cmake ..
sudo make install

# Clone and build nvtop
cd /tmp
git clone https://github.com/Syllo/nvtop.git
cd nvtop
cmake .
make
sudo make install



sudo apt-get update
sudo apt-get install -y libgl1-mesa-dev libglew-dev libglfw3-dev libglm-dev libegl1

python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r ./requirements.txt