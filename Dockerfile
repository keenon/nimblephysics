FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get clean \
    && apt-get update \
    && apt-get -y install --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN apt-get update \
    && apt-get -y install --no-install-recommends \
    libeigen3-dev \
    libassimp-dev \
    libccd-dev \
    libfcl-dev \
    libboost-all-dev \
    libnlopt-cxx-dev \
    coinor-libipopt-dev \
    libbullet-dev \
    liblz4-dev \
    libode-dev \
    liboctomap-dev \
    libflann-dev \
    libtinyxml2-dev \
    liburdfdom-dev \
    libxi-dev \
    libxmu-dev \
    freeglut3-dev \
    libopenscenegraph-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone git://github.com/dartsim/dart.git \
    && cd dart \
    && mkdir build \
    && cd build \
    && cmake \
    -DCMAKE_INSTALL_PREFIX=/usr/ \
    -DCMAKE_BUILD_TYPE=Release .. \
    && make \
    && make install \
    && cmake \
    -DDART_BUILD_DARTPY=ON .. \
    -DCMAKE_INSTALL_PREFIX=/usr/ \
    -DCMAKE_BUILD_TYPE=Release .. \
    && make dartpy \
    && make install

WORKDIR "/"