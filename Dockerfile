# -*- mode: dockerfile -*-
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04

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
        python3-opencv\
        python3 \
        python3-pip\
        python3.8-venv\
    && rm -rf /var/lib/apt/lists/*


WORKDIR /opt

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip
#RUN pip install numpy torch --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
RUN pip install git+https://github.com/d3sm0/experiment_buddy#egg=experiment_buddy --no-cache-dir
ENV BUDDY_IS_DEPLOYED=1

RUN touch /opt/entrypoint.sh && chmod 0755 /opt/entrypoint.sh
# TODO: This quite a hack
RUN echo "#!/bin/bash \n\
    source /opt/venv/bin/activate \n\
    exec \"\$@\"" > /opt/entrypoint.sh

#WORKDIR /opt
#RUN wget https://github.com/Kitware/CMake/releases/download/v3.23.4/cmake-3.23.4-linux-x86_64.sh && \
#    chmod +x cmake-3.23.4-linux-x86_64.sh && \
#    ./cmake-3.23.4-linux-x86_64.sh --skip-license --prefix=/usr/local
#RUN git clone https://github.com/facebookresearch/rlmeta && \
#    cd rlmeta && \
#    git submodule sync &&  \
#    git submodule update --init --recursive && \
#    pip install -e .
## WORKDIR /opt/rlmeta
## RUN
#
#WORKDIR /opt/impala
#COPY . /opt/impala
#RUN pip install -r requirements.txt --no-cache-dir

CMD ["/bin/bash"]
ENTRYPOINT ["/opt/entrypoint.sh"]

# Docker commands:
#   docker rm moolib -v
#   docker build -t moolib -f Dockerfile .
#   docker run --gpus all --rm --name moolib moolib
# or
#   docker run --gpus all -it --entrypoint /bin/bash moolib

