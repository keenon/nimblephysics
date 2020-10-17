#!/bin/bash
set -e

docker build .
IMAGE_ID=$(docker images | awk '{print $3}' | awk 'NR==2')
echo "Build image ID $IMAGE_ID"
CONTAINER_ID=$(docker create $IMAGE_ID)
echo "Started container ID $CONTAINER_ID"
VERSION=0.0.4
mkdir -p ../../../wheelhouse
docker cp $CONTAINER_ID:/wheelhouse/diffdart-${VERSION}-cp37-cp37m-manylinux2010_x86_64.whl ../../../wheelhouse
docker rm -v $CONTAINER_ID