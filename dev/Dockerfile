FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04
LABEL org.opencontainers.image.authors="uljadb@robots.ox.ac.uk"

ENV CUDA_PATH=/usr/local/cuda
ENV CUDA_INCLUDE_PATH=/usr/local/cuda/include
ENV CUDA_LIBRARY_PATH=/usr/local/cuda/lib64

# Set timezone
ENV TZ=Europe/London DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and runtime deps
RUN apt-get update && apt-get install -y software-properties-common && \
    apt-add-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    git \
    curl \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3-setuptools \
    python3-wheel \
    python3.10-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create local user
# https://jtreminio.com/blog/running-docker-containers-as-current-host-user/
ARG UID
ARG GID
RUN if [ ${UID:-0} -ne 0 ] && [ ${GID:-0} -ne 0 ]; then \
    groupadd -g ${GID} duser &&\
    useradd -l -u ${UID} -g duser duser &&\
    install -d -m 0755 -o duser -g duser /home/duser &&\
    chown --changes --silent --no-dereference --recursive ${UID}:${GID} /home/duser \
    ;fi

USER duser
WORKDIR /home/duser


ENV PATH="/home/duser/.local/bin:$PATH"
RUN python3.10 -m pip install --upgrade pip
ARG REQS
RUN python3.10 -m pip install --default-timeout=100 $REQS -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

WORKDIR /home/duser/recap_jax
