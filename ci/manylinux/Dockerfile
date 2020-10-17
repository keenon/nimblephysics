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
    cmake .. && \
    make install && \
    popd && \
    popd && \
    rm -rf PerfUtils

# This is allowed to be empty string, but if it's not it must be prefixed by
ENV VERSION="0.0.4"

ENV PYTHON35="/opt/python/cp35-cp35m/bin/python3.5"
ENV PYTHON35_VERSION="cp35-cp35m"
ENV PYTHON35_INCLUDE="/opt/python/cp35-cp35m/include/python3.5m/"
ENV PYTHON35_LIB="/opt/python/cp35-cp35m/lib/python3.5"

ENV PYTHON36="/opt/python/cp36-cp36m/bin/python3.6"
ENV PYTHON36_VERSION="cp36-cp36m"
ENV PYTHON36_INCLUDE="/opt/python/cp36-cp36m/include/python3.6m/"
ENV PYTHON36_LIB="/opt/python/cp36-cp36m/lib/python3.6"

ENV PYTHON37="/opt/python/cp37-cp37m/bin/python3.7"
ENV PYTHON37_VERSION="cp37-cp37m"
ENV PYTHON37_INCLUDE="/opt/python/cp37-cp37m/include/python3.7m/"
ENV PYTHON37_LIB="/opt/python/cp37-cp37m/lib/python3.7"

ENV PYTHON38="/opt/python/cp38-cp38/bin/python3.8"
ENV PYTHON38_VERSION="cp38-cp38"
ENV PYTHON38_INCLUDE="/opt/python/cp38-cp38/include/python3.8/"
ENV PYTHON38_LIB="/opt/python/cp38-cp38/lib/python3.8"

ENV PYTHON39="/opt/python/cp39-cp39/bin/python3.9"
ENV PYTHON39_VERSION="cp39-cp39"
ENV PYTHON39_INCLUDE="/opt/python/cp39-cp39/include/python3.9/"
ENV PYTHON39_LIB="/opt/python/cp39-cp39/lib/python3.9"

RUN mkdir /wheelhouse

# Build Python 3.5

ENV PYTHON=$PYTHON35
ENV PYTHON_VERSION=$PYTHON35_VERSION
ENV PYTHON_INCLUDE=$PYTHON35_INCLUDE
ENV PYTHON_LIB=$PYTHON35_LIB
ENV PYTHON_VERSION_NUMBER="3.5"

# Install pybind11
ENV CPATH="${PYTHON_INCLUDE}"
RUN git clone https://github.com/pybind/pybind11.git && \
    pushd pybind11 && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10
# Install pytest
RUN ${PYTHON} -m pip install pytest
RUN ${PYTHON} -m pip install auditwheel
RUN git clone https://github.com/keenon/diffdart
RUN cd diffdart && \
    git pull && \
    cat setup.py && \
    ${PYTHON} setup.py sdist bdist_wheel && \
    ${PYTHON} -m auditwheel repair dist/diffdart-${VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
RUN mv /diffdart/wheelhouse/diffdart-${VERSION}-${PYTHON_VERSION}-manylinux2010_x86_64.whl /wheelhouse
RUN rm -rf /diffdart
# Uninstall pybind11
RUN pushd pybind11 && \
    pushd build && \
    make uninstall && \
    popd && \
    popd && \
    rm -rf pybind11

# Build Python 3.6

ENV PYTHON=$PYTHON36
ENV PYTHON_VERSION=$PYTHON36_VERSION
ENV PYTHON_INCLUDE=$PYTHON36_INCLUDE
ENV PYTHON_LIB=$PYTHON36_LIB
ENV PYTHON_VERSION_NUMBER="3.6"

# Install pybind11
ENV CPATH="${PYTHON_INCLUDE}"
RUN git clone https://github.com/pybind/pybind11.git && \
    pushd pybind11 && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10
# Install pytest
RUN ${PYTHON} -m pip install pytest
RUN ${PYTHON} -m pip install auditwheel
RUN git clone https://github.com/keenon/diffdart
RUN cd diffdart && \
    git pull && \
    cat setup.py && \
    ${PYTHON} setup.py sdist bdist_wheel && \
    ${PYTHON} -m auditwheel repair dist/diffdart-${VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
RUN mv /diffdart/wheelhouse/diffdart-${VERSION}-${PYTHON_VERSION}-manylinux2010_x86_64.whl /wheelhouse
RUN rm -rf /diffdart
# Uninstall pybind11
RUN pushd pybind11 && \
    pushd build && \
    make uninstall && \
    popd && \
    popd && \
    rm -rf pybind11

# Build Python 3.7

ENV PYTHON=$PYTHON37
ENV PYTHON_VERSION=$PYTHON37_VERSION
ENV PYTHON_INCLUDE=$PYTHON37_INCLUDE
ENV PYTHON_LIB=$PYTHON37_LIB
ENV PYTHON_VERSION_NUMBER="3.7"

# Install pybind11
ENV CPATH="${PYTHON_INCLUDE}"
RUN git clone https://github.com/pybind/pybind11.git && \
    pushd pybind11 && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10
# Install pytest
RUN ${PYTHON} -m pip install pytest
RUN ${PYTHON} -m pip install auditwheel
RUN git clone https://github.com/keenon/diffdart
RUN cd diffdart && \
    git pull && \
    cat setup.py && \
    ${PYTHON} setup.py sdist bdist_wheel && \
    ${PYTHON} -m auditwheel repair dist/diffdart-${VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
RUN mv /diffdart/wheelhouse/diffdart-${VERSION}-${PYTHON_VERSION}-manylinux2010_x86_64.whl /wheelhouse
RUN rm -rf /diffdart
# Uninstall pybind11
RUN pushd pybind11 && \
    pushd build && \
    make uninstall && \
    popd && \
    popd && \
    rm -rf pybind11

# Build Python 3.8

ENV PYTHON=$PYTHON38
ENV PYTHON_VERSION=$PYTHON38_VERSION
ENV PYTHON_INCLUDE=$PYTHON38_INCLUDE
ENV PYTHON_LIB=$PYTHON38_LIB
ENV PYTHON_VERSION_NUMBER="3.8"

# Install pybind11
ENV CPATH="${PYTHON_INCLUDE}"
RUN git clone https://github.com/pybind/pybind11.git && \
    pushd pybind11 && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10
# Install pytest
RUN ${PYTHON} -m pip install pytest
RUN ${PYTHON} -m pip install auditwheel
RUN git clone https://github.com/keenon/diffdart
RUN cd diffdart && \
    git pull && \
    cat setup.py && \
    ${PYTHON} setup.py sdist bdist_wheel && \
    ${PYTHON} -m auditwheel repair dist/diffdart-${VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
RUN mv /diffdart/wheelhouse/diffdart-${VERSION}-${PYTHON_VERSION}-manylinux2010_x86_64.whl /wheelhouse
RUN rm -rf /diffdart
# Uninstall pybind11
RUN pushd pybind11 && \
    pushd build && \
    make uninstall && \
    popd && \
    popd && \
    rm -rf pybind11

# Build Python 3.9

ENV PYTHON=$PYTHON39
ENV PYTHON_VERSION=$PYTHON39_VERSION
ENV PYTHON_INCLUDE=$PYTHON39_INCLUDE
ENV PYTHON_LIB=$PYTHON39_LIB
ENV PYTHON_VERSION_NUMBER="3.9"

# Install pybind11
ENV CPATH="${PYTHON_INCLUDE}"
RUN git clone https://github.com/pybind/pybind11.git && \
    pushd pybind11 && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10
# Install pytest
RUN ${PYTHON} -m pip install pytest
RUN ${PYTHON} -m pip install auditwheel
RUN git clone https://github.com/keenon/diffdart
RUN cd diffdart && \
    git pull && \
    cat setup.py && \
    ${PYTHON} setup.py sdist bdist_wheel && \
    ${PYTHON} -m auditwheel repair dist/diffdart-${VERSION}-${PYTHON_VERSION}-linux_x86_64.whl
RUN mv /diffdart/wheelhouse/diffdart-${VERSION}-${PYTHON_VERSION}-manylinux2010_x86_64.whl /wheelhouse
RUN rm -rf /diffdart
# Uninstall pybind11
RUN pushd pybind11 && \
    pushd build && \
    make uninstall && \
    popd && \
    popd && \
    rm -rf pybind11