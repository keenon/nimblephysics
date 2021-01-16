FROM keenon/diffdart:base

# This is allowed to be empty string, but if it's not it must be prefixed by
ARG VERSION

RUN mkdir /wheelhouse

# Build Python 3.6

ENV PYTHON="/opt/python/cp36-cp36m/bin/python3.6"
ENV PYTHON_VERSION="cp36-cp36m"
ENV PYTHON_INCLUDE="/opt/python/cp36-cp36m/include/python3.6m/"
ENV PYTHON_LIB="/opt/python/cp36-cp36m/lib/python3.6"
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
RUN mv /diffdart/wheelhouse/diffdart-${VERSION}-${PYTHON_VERSION}-manylinux2014_x86_64.whl /wheelhouse