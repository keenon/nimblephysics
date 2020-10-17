FROM keenon/diffdart:base

# This is allowed to be empty string, but if it's not it must be prefixed by
ENV VERSION="0.0.4"

RUN mkdir /wheelhouse

# Build Python 3.5

ENV PYTHON="/opt/python/cp35-cp35m/bin/python3.5"
ENV PYTHON_VERSION="cp35-cp35m"
ENV PYTHON_INCLUDE="/opt/python/cp35-cp35m/include/python3.5m/"
ENV PYTHON_LIB="/opt/python/cp35-cp35m/lib/python3.5"
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