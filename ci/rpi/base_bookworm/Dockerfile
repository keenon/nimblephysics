# -------------------------------------------------------
# Dockerfile
# -------------------------------------------------------
FROM arm64v8/debian:bookworm

# Update the pkgconfig path
ENV PKG_CONFIG_PATH="/usr/local/lib64/pkgconfig:${PKG_CONFIG_PATH}"

# -------------------------------------------------------
# Install system dependencies
# -------------------------------------------------------
RUN set -ex \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    yasm \
    m4 \
    libgmp3-dev \
    libblas-dev \
    liblapack-dev \
    libboost-all-dev \
    pkg-config \
    ca-certificates \
    gfortran \
    clang \
    lldb \
    lld

# -------------------------------------------------------
# Install Eigen
# -------------------------------------------------------
RUN set -ex \
    && curl https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz > eigen.tar.gz \
    && tar -zxf eigen.tar.gz \
    && cd eigen-3.3.7 \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && make install -j14 \
    && cd ../.. \
    && rm -rf eigen-3.3.7 eigen.tar.gz

# -------------------------------------------------------
# Install CCD
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/danfis/libccd.git \
    && cd libccd \
    && mkdir build && cd build \
    && cmake .. -DENABLE_DOUBLE_PRECISION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && make install -j14 \
    && cd ../.. \
    && rm -rf libccd

# -------------------------------------------------------
# Install ASSIMP
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/assimp/assimp.git \
    && cd assimp \
    && git checkout v5.0.1 \
    && mkdir build && cd build \
    #   cmake .. -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_TESTS=ON \
    #            -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DCMAKE_BUILD_TYPE=Debug \
    && cmake .. -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_TESTS=ON -DASSIMP_BUILD_ASSIMP_TOOLS=OFF \
    && make install -j10 \
    && cd ../.. \
    && rm -rf assimp

# -------------------------------------------------------
# Install MUMPS
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git \
    && cd ThirdParty-Mumps \
    && ./get.Mumps \
    && ./configure \
    && make -j14 \
    && make install \
    && cd .. \
    && rm -rf ThirdParty-Mumps

# -------------------------------------------------------
# Install IPOPT
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/coin-or/Ipopt.git \
    && cd Ipopt \
    && ./configure --with-mumps \
    && make -j14 \
    && make install \
    && cd .. \
    && rm -rf Ipopt \
    && ln -s /usr/local/include/coin-or /usr/local/include/coin

# -------------------------------------------------------
# Install FCL (before octomap)
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/flexible-collision-library/fcl.git \
    && cd fcl \
    && git checkout 0.3.4 \
    # sed -i '1696s/v0_dist/(double)v0_dist/' ...
    && mkdir build && cd build \
    && cmake .. -DFCL_WITH_OCTOMAP=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && make install -j14 \
    && cd ../.. \
    && rm -rf fcl

# -------------------------------------------------------
# Install octomap
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/OctoMap/octomap.git \
    && cd octomap \
    && git checkout v1.10.0 \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && make install \
    && cd ../.. \
    && rm -rf octomap

# -------------------------------------------------------
# Install tinyxml2
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/leethomason/tinyxml2.git \
    && cd tinyxml2 \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && make install -j10 \
    && cd ../.. \
    && rm -rf tinyxml2

# -------------------------------------------------------
# (Optional) Install freeglut
# -------------------------------------------------------
#RUN set -ex \
#    && apt-get update \
#    && apt-get install -y libxi-dev libxmu-dev mesa-libglu-dev \
#    && curl https://managedway.dl.sourceforge.net/project/freeglut/freeglut/3.2.1/freeglut-3.2.1.tar.gz > freeglut.tar.gz \
#    && tar -zxf freeglut.tar.gz \
#    && rm freeglut.tar.gz \
#    && cd freeglut-3.2.1 \
#    && mkdir build && cd build \
#    && cmake .. \
#    && make install -j10 \
#    && cd ../.. \
#    && rm -rf freeglut-3.2.1

# -------------------------------------------------------
# (Optional) Install OpenSceneGraph
# -------------------------------------------------------
#RUN set -ex \
#    && git clone https://github.com/openscenegraph/OpenSceneGraph.git \
#    && cd OpenSceneGraph \
#    && git checkout OpenSceneGraph-3.6.5 \
#    && mkdir build && cd build \
#    && cmake .. \
#    && make install -j10 \
#    && cd ../.. \
#    && rm -rf OpenSceneGraph

# -------------------------------------------------------
# Install tinyxml (tinyxml1)
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/robotology-dependencies/tinyxml.git \
    && cd tinyxml \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && make install -j10 \
    && cd ../.. \
    && rm -rf tinyxml

# -------------------------------------------------------
# Install urdfdom_headers
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/ros/urdfdom_headers.git \
    && cd urdfdom_headers \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && make install -j10 \
    && cd ../.. \
    && rm -rf urdfdom_headers

# -------------------------------------------------------
# Install console_bridge
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/ros/console_bridge.git \
    && cd console_bridge \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && make install -j10 \
    && cd ../.. \
    && rm -rf console_bridge

# -------------------------------------------------------
# Install urdfdom
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/ros/urdfdom.git \
    && cd urdfdom \
    && mkdir build && cd build \
    && cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    && make install -j10 \
    && cd ../.. \
    && rm -rf urdfdom

ENV PROTOBUF_VERSION="29.2"

# -------------------------------------------------------
# Install gRPC (example)
# -------------------------------------------------------
RUN set -ex \
    && git clone --recurse-submodules -b v1.69.0 https://github.com/grpc/grpc \
    && cd grpc \
    # For older systems, you might also patch boringssl build lines
    && cd third_party/protobuf \
    && git checkout v${PROTOBUF_VERSION} \
    && cd ../.. \
    && mkdir -p cmake/build && cd cmake/build \
    && cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON ../.. \
    && make -j10 \
    && make install \
    && cd ../../.. \
    && rm -rf grpc

# -------------------------------------------------------
# Install Google benchmark
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/google/benchmark.git \
    && git clone https://github.com/google/googletest.git benchmark/googletest \
    && cd benchmark \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. \
    && make install \
    && cd ../.. \
    && rm -rf benchmark

# -------------------------------------------------------
# Install MPFR - Arbitrary precision floating point math
# -------------------------------------------------------
RUN set -ex \
    && curl https://www.mpfr.org/mpfr-current/mpfr-4.2.1.tar.gz > mpfr-4.2.1.tar.gz \
    && tar -zxf mpfr-4.2.1.tar.gz \
    && cd mpfr-4.2.1 \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf mpfr-4.2.1 mpfr-4.2.1.tar.gz

# -------------------------------------------------------
# Install MPFRC++
# -------------------------------------------------------
RUN set -ex \
    && wget https://github.com/advanpix/mpreal/archive/refs/tags/mpfrc++-3.6.8.tar.gz \
    && tar -xzf mpfrc++-3.6.8.tar.gz \
    && cd mpreal-mpfrc-3.6.8 \
    && cp mpreal.h /usr/include/ \
    && cd .. \
    && rm -rf mpreal-mpfrc-3.6.8 mpfrc++-3.6.8.tar.gz

# -------------------------------------------------------
# Install ezc3d
# -------------------------------------------------------
RUN set -ex \
    && git clone https://github.com/pyomeca/ezc3d.git \
    && cd ezc3d \
    && git checkout Release_1.4.7 \
    && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. \
    && make install \
    && cd ../.. \
    && rm -rf ezc3d

# -------------------------------------------------------
# (Optional) Test protoc version
# -------------------------------------------------------
RUN protoc --version