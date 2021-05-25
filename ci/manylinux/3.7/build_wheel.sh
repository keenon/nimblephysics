#!/bin/bash
set -e

VERSION=$(cat ../../../VERSION.txt)
GIT_HASH=$(git log --pretty=format:'%H' -n 1)
docker build . --build-arg VERSION=${VERSION} --build-arg GIT_HASH=${GIT_HASH}
IMAGE_ID=$(docker images | awk '{print $3}' | awk 'NR==2')
echo "Build image ID $IMAGE_ID"
CONTAINER_ID=$(docker create $IMAGE_ID)
echo "Started container ID $CONTAINER_ID"
mkdir -p ../../../wheelhouse
docker cp $CONTAINER_ID:/wheelhouse/nimblephysics-${VERSION}-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl ../../../wheelhouse
docker rm -v $CONTAINER_ID