ARG IMAGE_NAME=nvidia/cuda
#FROM ${IMAGE_NAME}:12.8.1-runtime-ubuntu24.04 AS base
FROM ${IMAGE_NAME}:12.8.1-devel-ubuntu24.04 AS base
#FROM ${IMAGE_NAME}:11.8.0-devel-ubuntu20.04 AS base

RUN apt-get update && \
    apt-get install -y cmake libepoxy-dev \
    libglm-dev pkg-config \
    python3 python3-venv python3-pip git

COPY requirements.txt /tmp/requirements.txt    

RUN mkdir -p /acoustic-render

WORKDIR /acoustic-render    

RUN python3 -m venv venv && \
    /acoustic-render/venv/bin/python -m pip install --upgrade pip && \
    /acoustic-render/venv/bin/pip install -r /tmp/requirements.txt

COPY src /acoustic-render/src
COPY run_scripts /acoustic-render/run_scripts
COPY docs /acoustic-render/docs
COPY .git /acoustic-render/.git

COPY ./lib/* /usr/lib/x86_64-linux-gnu/.
# RUN rm  /usr/lib/x86_64-linux-gnu/libEGL.* && \
#     rm  /usr/lib/x86_64-linux-gnu/libEGL_mesa.* && \
#     ln -sf /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 /usr/lib/x86_64-linux-gnu/libEGL.so.1 && \
#     ln -sf /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 /usr/lib/x86_64-linux-gnu/libEGL.so

RUN   rm  /usr/lib/x86_64-linux-gnu/libEGL_mesa.* && \
    ln -sf /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0

RUN mkdir -p build && \
    cd build && \
    cmake ../src/ && \
    make -j4 all  

#ENTRYPOINT []
#CMD ["bash","/acoustic-render/run_scripts/run_container.sh"]
