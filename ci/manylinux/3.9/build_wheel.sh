#!/bin/bash
set -e

VERSION=$(cat ../../../VERSION.txt)
docker build . --build-arg VERSION=${VERSION}
IMAGE_ID=$(docker images | awk '{print $3}' | awk 'NR==2')
echo "Build image ID $IMAGE_ID"
CONTAINER_ID=$(docker create $IMAGE_ID)
echo "Started container ID $CONTAINER_ID"
mkdir -p ../../../wheelhouse
docker cp $CONTAINER_ID:/wheelhouse/diffdart-${VERSION}-cp39-cp39-manylinux2010_x86_64.whl ../../../wheelhouse
docker rm -v $CONTAINER_ID