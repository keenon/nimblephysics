FROM quay.io/pypa/manylinux2010_x86_64
MAINTAINER keenonwerling@gmail.com

ENV PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:/usr/local/lib64/pkgconfig/"

# Install CMake3
RUN yum install -y cmake3 && \
    rm /usr/bin/cmake && \
    ln -s /usr/bin/cmake3 /usr/bin/cmake

# Install Boost
RUN yum install epel-release && \
    rpm -ivh http://repo.okay.com.mx/centos/6/x86_64/release/okay-release-1-3.el6.noarch.rpm? && \
    yum install -y boost-devel-1.55.0-25.el6.x86_64


# Install Eigen
RUN curl https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz > eigen.tar.gz && \
    tar -zxf eigen.tar.gz && \
    pushd eigen-3.3.7 && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j14 && \
    popd && \
    popd && \
    rm -rf eigen-3.3.7

# Install CCD
RUN git clone https://github.com/danfis/libccd.git && \
    pushd libccd && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j14 && \
    popd && \
    popd && \
    rm -rf libccd

# Install ASSIMP
RUN git clone https://github.com/assimp/assimp.git && \
    pushd assimp && \
    mkdir build && \
    pushd build && \
    cmake .. && \
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

ENV CPATH="${CPATH}:/opt/python/cp38-cp38/include/python3.8/"
# Install pybind11
RUN git clone https://github.com/pybind/pybind11.git && \
    pushd pybind11 && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf pybind11

# Install FCL
# Key note: this needs to happen before octomap
RUN git clone https://github.com/flexible-collision-library/fcl.git && \
    pushd fcl && \
    git checkout 0.3.4 && \
    # vi include/fcl/narrowphase/detail/convexity_based_algorithm/gjk_libccd-inl.h:1696 # "std::max(1.0, v0_dist)" -> "std::max(1.0, (double)v0_dist)" && \
    # sed -i '1696s/v0_dist/(double)v0_dist/' include/fcl/narrowphase/detail/convexity_based_algorithm/gjk_libccd-inl.h && \
    mkdir build && \
    pushd build && \
    cmake .. -DFCL_WITH_OCTOMAP=OFF && \
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
    cmake .. && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf octomap

# Install tinyxml2
RUN git clone https://github.com/leethomason/tinyxml2.git && \
    pushd tinyxml2 && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf tinyxml2

# Install freeglut
RUN yum install -y libXi-devel && \
    yum install -y mesa-libGLU-devel && \
    yum install -y libXmu-devel && \
    curl https://managedway.dl.sourceforge.net/project/freeglut/freeglut/3.2.1/freeglut-3.2.1.tar.gz > freeglut.tar.gz && \
    tar -zxf freeglut.tar.gz && \
    rm freeglut.tar.gz && \
    pushd freeglut-3.2.1 && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf freeglut-3.2.1

# Install Open Scene Graph
RUN git clone https://github.com/openscenegraph/OpenSceneGraph.git && \
    pushd OpenSceneGraph && \
    git checkout OpenSceneGraph-3.6.5 && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf OpenSceneGraph

# Install tinyxml1
RUN git clone https://github.com/robotology-dependencies/tinyxml.git && \
    pushd tinyxml && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf tinyxml

# Install urdfdom_headers
RUN git clone https://github.com/ros/urdfdom_headers.git && \
    pushd urdfdom_headers && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf urdfdom_headers

# Install console_bridge
RUN git clone https://github.com/ros/console_bridge.git && \
    pushd console_bridge && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf console_bridge

# Install urdfdom
RUN git clone https://github.com/ros/urdfdom.git && \
    pushd urdfdom && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf urdfdom

# This is allowed to be empty string, but if it's not it must be prefixed by
ENV VERSION="0.0.3"

ENV PYTHON35="/opt/python/cp35-cp35m/bin/python3.5"
ENV PYTHON35_VERSION="cp35-cp35m"
ENV PYTHON36="/opt/python/cp36-cp36m/bin/python3.6"
ENV PYTHON36_VERSION="cp36-cp36m"
ENV PYTHON37="/opt/python/cp37-cp37m/bin/python3.7"
ENV PYTHON37_VERSION="cp37-cp37m"
ENV PYTHON38="/opt/python/cp38-cp38/bin/python3.8"
ENV PYTHON38_VERSION="cp38-cp38"
ENV PYTHON39="/opt/python/cp39-cp39/bin/python3.9"
ENV PYTHON39_VERSION="cp39-cp39"

ENV PYTHON=$PYTHON37
ENV PYTHON_VERSION=$PYTHON37_VERSION

# Install pytest
RUN ${PYTHON} -m pip install pytest

# Install auditwheel
RUN ${PYTHON} -m pip install auditwheel

# Actually build the code
RUN git clone https://github.com/keenon/diffdart
RUN cd diffdart && \
    git checkout keenon/diff && \
    git pull

RUN cd diffdart && \
    git pull && \
    ${PYTHON} setup.py sdist bdist_wheel && \
    ${PYTHON} -m auditwheel repair dist/diffdart-${VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
# Ensure we have the ABI3 tag, so that pip interpreters will accept us
# RUN mv diffdart/wheelhouse/diffdart-${VERSION}-${PYTHON_VERSION}-manylinux2010_x86_64.whl diffdart/wheelhouse/diffdart-${VERSION}-cp38-abi3-manylinux2010_x86_64.whl