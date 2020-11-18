# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

find_package(protobuf QUIET CONFIG)

if(protobuf_FOUND AND NOT TARGET protobuf)
  add_library(protobuf INTERFACE IMPORTED)
  set_target_properties(protobuf PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${protobuf_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${protobuf_LIBRARIES}"
  )
endif()
