#!/bin/bash

echo 'Building Dockerfile with image name recap_jax'
docker build \
    --build-arg UID=$(id -u ${USER}) \
    --build-arg GID=1234 \
    --build-arg REQS="$(cat requirements.txt | tr '\n' ' ')" \
    -t recap_jax \
    -f Dockerfile \
    .
