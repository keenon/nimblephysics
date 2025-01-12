# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
# https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

# find_package(Protobuf "5.29.2" EXACT REQUIRED)
find_package(Protobuf REQUIRED)

message(STATUS "Protobuf_LIBRARIES: ${Protobuf_LIBRARIES}")

if(Protobuf_FOUND AND NOT TARGET protobuf::libprotobuf)
  add_library(protobuf::libprotobuf INTERFACE IMPORTED)
  set_target_properties(protobuf::libprotobuf PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${Protobuf_LIBRARIES}"
  )
endif()

if(Protobuf_FOUND AND NOT TARGET protobuf::libprotoc)
  add_library(protobuf::libprotoc INTERFACE IMPORTED)
  set_target_properties(protobuf::libprotoc PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${Protobuf_PROTOC_LIBRARIES}"
  )
endif()

if(Protobuf_FOUND AND NOT TARGET protobuf::libprotobuf-lite)
  add_library(protobuf::libprotobuf-lite INTERFACE IMPORTED)
  set_target_properties(protobuf::libprotobuf-lite PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${Protobuf_LITE_LIBRARIES}"
  )
endif()