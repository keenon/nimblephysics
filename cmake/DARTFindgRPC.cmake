# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

find_package(gRPC CONFIG REQUIRED)

if(gRPC_FOUND AND NOT TARGET gRPC::grpc++)
  add_library(gRPC::grpc++ INTERFACE IMPORTED)
  set_target_properties(gRPC::grpc++ PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${gRPC_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${gRPC_LIBRARIES}"
  )
endif()