FROM keenon/diffdart:base

# Re-Install PerfUtils - This needs to be rebuilt on each machine individually, since otherwise it breaks binaries.
RUN git clone https://github.com/PlatformLab/PerfUtils.git && \
    pushd PerfUtils && \
    sed -i 's/3.11/3.6.1/g' CMakeLists.txt && \
    sed -i '94,$d' CMakeLists.txt && \
    sed -i '30,33d' CMakeLists.txt && \
    sed -i '2iset(CMAKE_POSITION_INDEPENDENT_CODE ON)' CMakeLists.txt && \
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

# Re-Install tinyxml2 - Just like PerfUtils, this needs to be built on each machine individually, since otherwise it can be incompatible
RUN git clone https://github.com/leethomason/tinyxml2.git && \
    pushd tinyxml2 && \
    sed -i '20iset(CMAKE_POSITION_INDEPENDENT_CODE ON)' CMakeLists.txt && \
    mkdir build && \
    pushd build && \
    cmake .. && \
    make install -j10 && \
    popd && \
    popd && \
    rm -rf tinyxml2

# This is allowed to be empty string, but if it's not it must be prefixed by
ARG VERSION

RUN mkdir /wheelhouse

# Build Python 3.8

ENV PYTHON="/opt/python/cp38-cp38/bin/python3.8"
ENV PATH="/opt/python/cp38-cp38/bin/:${PATH}"
ENV PYTHON_VERSION="cp38-cp38"
ENV PYTHON_INCLUDE="/opt/python/cp38-cp38/include/python3.8/"
ENV PYTHON_LIB="/opt/python/cp38-cp38/lib/python3.8"
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
RUN cd diffdart

# Install some extra tools that we don't typically need in CI, but are nice in development
RUN yum install -y gdb
RUN curl -sL https://rpm.nodesource.com/setup_10.x | bash -
RUN yum install -y nodejs
RUN protoc --version

# Expose common ports for the web GUI server
EXPOSE 9000
EXPOSE 8080
# Expose port for WebSocket updates for the GUI
EXPOSE 8070