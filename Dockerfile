# -*- mode: dockerfile -*-
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -yq \
        build-essential \
        curl \
        git \
        ninja-build \
        libglib2.0-0 \
        sudo\
        wget \
        python3 \
        python3-pip\
        python3.8-venv


WORKDIR /opt
RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.4/cmake-3.23.4-linux-x86_64.sh
RUN chmod +x cmake-3.23.4-linux-x86_64.sh
RUN ./cmake-3.23.4-linux-x86_64.sh --skip-license --prefix=/usr

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip
RUN pip install numpy torch

WORKDIR /opt
RUN git clone https://github.com/facebookresearch/rlmeta
WORKDIR /opt/rlmeta
RUN git submodule sync && git submodule update --init --recursive
RUN pip install -e .

CMD ["/bin/bash"]

# Docker commands:
#   docker rm moolib -v
#   docker build -t moolib -f Dockerfile .
#   docker run --gpus all --rm --name moolib moolib
# or
#   docker run --gpus all -it --entrypoint /bin/bash moolib
