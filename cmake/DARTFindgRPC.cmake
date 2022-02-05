# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

message(STATUS "Dart finding gRPC")

if(APPLE)
  # x86_64 macs
  if(EXISTS "/usr/local/opt/openssl")
    # set(OPENSSL_ROOT_DIR /usr/local/Cellar/openssl@1.1/1.1.1d/)
    message(STATUS "Setting root dir for OpenSSL")
    set(OPENSSL_ROOT_DIR /usr/local/opt/openssl)

    find_package(OpenSSL REQUIRED)
    include_directories(/usr/local/Cellar/openssl@1.1/1.1.1d/include)
    list(APPEND LIB_LIST /usr/local/Cellar/openssl@1.1/1.1.1d/lib/libssl.dylib)
    list(APPEND LIB_LIST /usr/local/Cellar/openssl@1.1/1.1.1d/lib/libcrypto.dylib)
    message(STATUS "OpenSSL Version: ${OPENSSL_VERSION} ${OPENSSL_INCLUDE_DIR} ${OPENSSL_LIBRARIES}")
  # ARM64 macs
  elseif(EXISTS "/opt/homebrew/Cellar/openssl@1.1/")
    message(STATUS "Setting root dir for OpenSSL")
    set(OPENSSL_ROOT_DIR /opt/homebrew/Cellar/openssl@1.1/1.1.1l_1)

    find_package(OpenSSL REQUIRED)
    include_directories(/opt/homebrew/Cellar/openssl@1.1/1.1.1l_1/include)
    list(APPEND LIB_LIST /opt/homebrew/Cellar/openssl@1.1/1.1.1l_1/lib/libssl.dylib)
    list(APPEND LIB_LIST /opt/homebrew/Cellar/openssl@1.1/1.1.1l_1/lib/libcrypto.dylib)
    message(STATUS "OpenSSL Version: ${OPENSSL_VERSION} ${OPENSSL_INCLUDE_DIR} ${OPENSSL_LIBRARIES}")
  endif()
endif()

find_package(gRPC CONFIG REQUIRED)

message(STATUS "gRPC_FOUND: ${gRPC_FOUND}")
# get_target_property(gRPC_INCLUDE_DIRS gRPC::grpc++ INTERFACE_INCLUDE_DIRECTORIES)
# get_target_property(gRPC_LIBRARIES gRPC::grpc++ INTERFACE_LINK_LIBRARIES)
message(STATUS "gRPC_INCLUDE_DIRS: ${gRPC_INCLUDE_DIRS}")
message(STATUS "gRPC_LIBRARIES: ${gRPC_LIBRARIES}")

if(gRPC_FOUND AND NOT TARGET gRPC::grpc++)
  add_library(gRPC::grpc++ INTERFACE IMPORTED)
  set_target_properties(gRPC::grpc++ PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${gRPC_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${gRPC_LIBRARIES}"
  )
endif()