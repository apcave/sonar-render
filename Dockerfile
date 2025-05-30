ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:12.8.1-runtime-ubuntu24.04 AS base

ENV NV_CUDA_LIB_VERSION="12.8.1-1"

FROM base AS base-amd64

ENV NV_CUDA_CUDART_DEV_VERSION=12.8.90-1
ENV NV_NVML_DEV_VERSION=12.8.90-1
ENV NV_LIBCUSPARSE_DEV_VERSION=12.5.8.93-1
ENV NV_LIBNPP_DEV_VERSION=12.3.3.100-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-12-8=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_VERSION=12.8.4.1-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-12-8
ENV NV_LIBCUBLAS_DEV_PACKAGE=${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=12.8.1-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-12-8=${NV_CUDA_NSIGHT_COMPUTE_VERSION}

ENV NV_NVPROF_VERSION=12.8.90-1
ENV NV_NVPROF_DEV_PACKAGE=cuda-nvprof-12-8=${NV_NVPROF_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.25.1-1
ENV NCCL_VERSION=2.25.1-1
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.8
FROM base AS base-arm64

ENV NV_CUDA_CUDART_DEV_VERSION=12.8.90-1
ENV NV_NVML_DEV_VERSION=12.8.90-1
ENV NV_LIBCUSPARSE_DEV_VERSION=12.5.8.93-1
ENV NV_LIBNPP_DEV_VERSION=12.3.3.100-1
ENV NV_LIBNPP_DEV_PACKAGE=libnpp-dev-12-8=${NV_LIBNPP_DEV_VERSION}

ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=libcublas-dev-12-8
ENV NV_LIBCUBLAS_DEV_VERSION=12.8.4.1-1
ENV NV_LIBCUBLAS_DEV_PACKAGE=${NV_LIBCUBLAS_DEV_PACKAGE_NAME}=${NV_LIBCUBLAS_DEV_VERSION}

ENV NV_CUDA_NSIGHT_COMPUTE_VERSION=12.8.1-1
ENV NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE=cuda-nsight-compute-12-8=${NV_CUDA_NSIGHT_COMPUTE_VERSION}

ENV NV_LIBNCCL_DEV_PACKAGE_NAME=libnccl-dev
ENV NV_LIBNCCL_DEV_PACKAGE_VERSION=2.25.1-1
ENV NCCL_VERSION=2.25.1-1
ENV NV_LIBNCCL_DEV_PACKAGE=${NV_LIBNCCL_DEV_PACKAGE_NAME}=${NV_LIBNCCL_DEV_PACKAGE_VERSION}+cuda12.8


FROM base-${TARGETARCH}

ARG TARGETARCH

LABEL maintainer="NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-dev-12-8=${NV_CUDA_CUDART_DEV_VERSION} \
    cuda-command-line-tools-12-8=${NV_CUDA_LIB_VERSION} \
    cuda-minimal-build-12-8=${NV_CUDA_LIB_VERSION} \
    cuda-libraries-dev-12-8=${NV_CUDA_LIB_VERSION} \
    cuda-nvml-dev-12-8=${NV_NVML_DEV_VERSION} \
    ${NV_NVPROF_DEV_PACKAGE} \
    ${NV_LIBNPP_DEV_PACKAGE} \
    libcusparse-dev-12-8=${NV_LIBCUSPARSE_DEV_VERSION} \
    ${NV_LIBCUBLAS_DEV_PACKAGE} \
    ${NV_LIBNCCL_DEV_PACKAGE} \
    ${NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE} \
    && rm -rf /var/lib/apt/lists/*

# Keep apt from auto upgrading the cublas and nccl packages. See https://gitlab.com/nvidia/container-images/cuda/-/issues/88
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs

RUN apt-get update && apt-get install -y cmake libgl1-mesa-dev \
    libglew-dev libglfw3-dev libglm-dev \
    python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY /lib/* /usr/local/lib/.    

COPY requirements.txt /tmp/requirements.txt    
RUN python3 -m venv /venv && \
    /venv/bin/python -m pip install --upgrade pip && \
    /venv/bin/pip install -r /tmp/requirements.txt

WORKDIR /acoustic-render
COPY ./src /acoustic-render/src
RUN mkdir -p build && \
    cd build && \
    cmake ../src/ && \
    make -j4

COPY ./testing /acoustic-render/testing
COPY NVIDIA-OptiX-SDK-9.0.0-linux64-x86_64.sh /acoustic-render/.
WORKDIR /
#ENTRYPOINT []
#CMD ["python /batches/cube.py"]