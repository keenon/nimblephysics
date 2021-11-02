FROM keenon/diffdart:base

# This is allowed to be empty string, but if it's not it must be prefixed by
ARG VERSION
ARG GIT_HASH

COPY . /nimblephysics

# Build Python 3.8

ENV PYTHON="/opt/python/cp38-cp38/bin/python3.8"
ENV PYTHON_VERSION="cp38-cp38"
ENV PYTHON_INCLUDE="/opt/python/cp38-cp38/include/python3.8/"
ENV PYTHON_LIB="/opt/python/cp38-cp38/lib/python3.8"
ENV PYTHON_VERSION_NUMBER="3.8"
RUN rm /usr/bin/python && \
    ln -s $PYTHON /usr/bin/python

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
RUN mkdir /Testing
RUN cd nimblephysics && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make tests -j2
RUN cd nimblephysics/build && \
    ctest -T Test; exit 0
RUN mv /nimblephysics/build/Testing /Testing