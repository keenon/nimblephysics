# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

message(STATUS "Dart finding gRPC")

if(APPLE)
  # set(OPENSSL_ROOT_DIR /usr/local/Cellar/openssl@1.1/1.1.1d/)
  message(STATUS "Setting root dir for OpenSSL")
  set(OPENSSL_ROOT_DIR /usr/local/opt/openssl)

  find_package(OpenSSL REQUIRED)
  include_directories(/usr/local/Cellar/openssl@1.1/1.1.1d/include)
  list(APPEND LIB_LIST /usr/local/Cellar/openssl@1.1/1.1.1d/lib/libssl.dylib)
  list(APPEND LIB_LIST /usr/local/Cellar/openssl@1.1/1.1.1d/lib/libcrypto.dylib)
  message(STATUS "OpenSSL Version: ${OPENSSL_VERSION} ${OPENSSL_INCLUDE_DIR} ${OPENSSL_LIBRARIES}")
endif()

find_package(gRPC CONFIG REQUIRED)

if(gRPC_FOUND AND NOT TARGET gRPC::grpc++)
  add_library(gRPC::grpc++ INTERFACE IMPORTED)
  set_target_properties(gRPC::grpc++ PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${gRPC_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${gRPC_LIBRARIES}"
  )
endif()