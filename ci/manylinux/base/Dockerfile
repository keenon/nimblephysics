FROM quay.io/pypa/manylinux2014_x86_64
MAINTAINER keenonwerling@gmail.com
# TAG keenon/diffdart:base

ENV PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:/usr/local/lib64/pkgconfig/"

# Upgrade to the latest CMake version

RUN yum install -y wget

RUN yum install -y openssl-devel

RUN wget https://github.com/Kitware/CMake/releases/download/v3.19.0/cmake-3.19.0.tar.gz
RUN tar -xvzf cmake-3.19.0.tar.gz
RUN pushd cmake-3.19.0 && \
    ./bootstrap && \
    make && \
    make install
RUN rm -rf cmake-3.19.0
RUN rm cmake-3.19.0.tar.gz

# Install Boost
# RUN yum install epel-release && \
#     rpm -ivh http://repo.okay.com.mx/centos/7/x86_64/release/okay-release-1-3.el6.noarch.rpm? && \
#     yum install -y boost-devel-1.55.0-25.el6.x86_64
RUN yum install -y boost-devel-1.53.0-28.el7.x86_64

# Install Eigen
RUN curl https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz > eigen.tar.gz && \
    tar -zxf eigen.tar.gz && \
    pushd eigen-3.3.7 && \
    mkdir build && \
    pushd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j14 && \
    popd && \
    popd && \
    rm -rf eigen-3.3.7

# Install CCD
RUN git clone https://github.com/danfis/libccd.git && \
    pushd libccd && \
    mkdir build && \
    pushd build && \
    cmake .. -DENABLE_DOUBLE_PRECISION=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j14 && \
    popd && \
    popd && \
    rm -rf libccd

# Install ASSIMP
RUN git clone https://github.com/assimp/assimp.git && \
    pushd assimp && \
    git checkout v5.0.1 && \
    mkdir build && \
    pushd build && \
    # cmake .. -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_TESTS=ON -DASSIMP_BUILD_ASSIMP_TOOLS=OFF -DCMAKE_BUILD_TYPE=Debug && \
    cmake .. -DASSIMP_BUILD_ZLIB=ON -DASSIMP_BUILD_TESTS=ON -DASSIMP_BUILD_ASSIMP_TOOLS=OFF && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf assimp

# Install LAPACK
RUN yum install -y lapack-devel

# Install MUMPS
RUN git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git && \
    pushd ThirdParty-Mumps && \
    ./get.Mumps && \
    ./configure && \
    make -j14 && \
    make install && \
    popd && \
    rm -rf ThirdParty-Mumps

# Install IPOPT
RUN git clone https://github.com/coin-or/Ipopt.git && \
    pushd Ipopt && \
    ./configure --with-mumps && \
    make -j14 && \
    make install && \
    popd && \
    rm -rf Ipopt && \
    ln -s /usr/local/include/coin-or /usr/local/include/coin

# Install FCL
# Key note: this needs to happen before octomap
RUN git clone https://github.com/flexible-collision-library/fcl.git && \
    pushd fcl && \
    git checkout 0.3.4 && \
    # vi include/fcl/narrowphase/detail/convexity_based_algorithm/gjk_libccd-inl.h:1696 # "std::max(1.0, v0_dist)" -> "std::max(1.0, (double)v0_dist)" && \
    # sed -i '1696s/v0_dist/(double)v0_dist/' include/fcl/narrowphase/detail/convexity_based_algorithm/gjk_libccd-inl.h && \
    mkdir build && \
    pushd build && \
    cmake .. -DFCL_WITH_OCTOMAP=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j14 && \
    popd && \
    popd && \
    rm -rf fcl

# Install octomap
RUN git clone https://github.com/OctoMap/octomap.git && \
    pushd octomap && \
    git checkout v1.8.1 && \
    mkdir build && \
    pushd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf octomap

# Install tinyxml2
RUN git clone https://github.com/leethomason/tinyxml2.git && \
    pushd tinyxml2 && \
    mkdir build && \
    pushd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf tinyxml2

# Install freeglut
# RUN yum install -y libXi-devel && \
#     yum install -y mesa-libGLU-devel && \
#     yum install -y libXmu-devel && \
#     curl https://managedway.dl.sourceforge.net/project/freeglut/freeglut/3.2.1/freeglut-3.2.1.tar.gz > freeglut.tar.gz && \
#     tar -zxf freeglut.tar.gz && \
#     rm freeglut.tar.gz && \
#     pushd freeglut-3.2.1 && \
#     mkdir build && \
#     pushd build && \
#     cmake .. && \
#     make install -j10 && \
#     popd && \
#     popd && \
#     rm -rf freeglut-3.2.1

# Install Open Scene Graph
# RUN git clone https://github.com/openscenegraph/OpenSceneGraph.git && \
#     pushd OpenSceneGraph && \
#     git checkout OpenSceneGraph-3.6.5 && \
#     mkdir build && \
#     pushd build && \
#     cmake .. && \
#     make install -j10 && \
#     popd && \
#     popd && \
#     rm -rf OpenSceneGraph

# Install tinyxml1
RUN git clone https://github.com/robotology-dependencies/tinyxml.git && \
    pushd tinyxml && \
    mkdir build && \
    pushd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf tinyxml

# Install urdfdom_headers
RUN git clone https://github.com/ros/urdfdom_headers.git && \
    pushd urdfdom_headers && \
    mkdir build && \
    pushd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf urdfdom_headers

# Install console_bridge
RUN git clone https://github.com/ros/console_bridge.git && \
    pushd console_bridge && \
    mkdir build && \
    pushd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf console_bridge

# Install urdfdom
RUN git clone https://github.com/ros/urdfdom.git && \
    pushd urdfdom && \
    mkdir build && \
    pushd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf urdfdom

# Install PerfUtils
RUN git clone https://github.com/PlatformLab/PerfUtils.git && \
    pushd PerfUtils && \
    sed -i 's/3.11/3.6.1/g' CMakeLists.txt && \
    sed -i '94,$d' CMakeLists.txt && \
    sed -i '30,33d' CMakeLists.txt && \
    sed -i '36i\ \ \ \ CXX_STANDARD 11' CMakeLists.txt && \
    sed -i '36i\ \ \ \ CXX_STANDARD_REQUIRED YES' CMakeLists.txt && \
    sed -i '36i\ \ \ \ CXX_EXTENSIONS NO' CMakeLists.txt && \
    mkdir build && \
    pushd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
    make install && \
    popd && \
    popd && \
    rm -rf PerfUtils

ENV PROTOBUF_VERSION="3.14.0"

# RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protobuf-all-${PROTOBUF_VERSION}.tar.gz && \
#     tar -xvzf protobuf-all-${PROTOBUF_VERSION}.tar.gz && \
#     rm protobuf-all-${PROTOBUF_VERSION}.tar.gz && \
#     pushd protobuf-${PROTOBUF_VERSION} && \
#     ./configure && \
#     make -j16 && \
#     make check -j16 && \
#     make install && \ 
#     popd && \
#     rm -rf protobuf-${PROTOBUF_VERSION}

RUN git clone --recurse-submodules -b v1.33.2 https://github.com/grpc/grpc
RUN pushd grpc && \
    # This fixes the boringssl build on the ancient CentOS we have to use by adding "rt" as an explicit dependency
    sed -i '642s/.*/target_link_libraries(bssl ssl crypto rt)/' third_party/boringssl-with-bazel/CMakeLists.txt && \
    pushd third_party/protobuf && \
    git checkout v${PROTOBUF_VERSION} && \
    popd && \
    mkdir -p cmake/build && \
    pushd cmake/build && \
    cmake -DgRPC_INSTALL=ON \
          -DgRPC_BUILD_TESTS=OFF \
          -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
          ../.. && \
    make -j3 && \
    make install && \
    popd && \
    popd
RUN rm -rf grpc

# Install Google benchmark
RUN git clone https://github.com/google/benchmark.git
RUN git clone https://github.com/google/googletest.git benchmark/googletest
RUN pushd benchmark && \
    mkdir build && \
    pushd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && \
    make install && \
    popd && \
    popd
RUN rm -rf benchmark

# Install MPFR - Arbirtatry precision floating point math
RUN yum install -y gmp-devel
RUN curl https://www.mpfr.org/mpfr-current/mpfr-4.1.0.tar.gz > mpfr-4.1.0.tar.gz && \
    tar -zxf mpfr-4.1.0.tar.gz && \
    pushd mpfr-4.1.0 && \
    ./configure && \
    make && \
    make install && \
    popd && \
    rm -rf mpfr-4.1.0 && \
    rm mpfr-4.1.0.tar.gz

# Install MPIR - Arbirtatry precision integer math
RUN yum install -y yasm m4
RUN curl http://mpir.org/mpir-3.0.0.tar.bz2 > mpir-3.0.0.tar.bz2 && \
    tar -xf mpir-3.0.0.tar.bz2 && \
    pushd mpir-3.0.0 && \
    ./configure --enable-cxx && \
    make && \
    make install && \
    popd && \
    rm -rf mpir-3.0.0 && \
    rm mpir-3.0.0.tar.bz2

# Install MPFRC++
RUN wget https://github.com/advanpix/mpreal/archive/refs/tags/mpfrc++-3.6.8.tar.gz
RUN tar -xzf mpfrc++-3.6.8.tar.gz && \
    pushd mpreal-mpfrc-3.6.8 && \
    cp mpreal.h /usr/include/ && \
    popd && \
    rm -rf mpreal-mpfrc-3.6.8 && \
    rm mpfrc++-3.6.8.tar.gz

# Install ezc3d
RUN git clone https://github.com/pyomeca/ezc3d.git
RUN pushd ezc3d && \
    git checkout Release_1.4.7 && \
    mkdir build && \
    pushd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON .. && \
    make install && \
    popd && \
    popd
RUN rm -rf ezc3d

# RUN ls /usr/local/lib64 | grep assimp
# RUN ls /usr/local/lib | grep assimp
# RUN rm -rf /usr/local/lib64/libassimp.so
# RUN rm -rf /usr/local/lib64/libassimp.5.so
# RUN rm -rf /usr/local/lib64/libassimp.5.0.1.so
RUN protoc --version