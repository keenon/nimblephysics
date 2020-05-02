# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

# if(MSVC)
  # Use find module because urdfdom-config.cmake doesn't work on Windows:
  # https://github.com/dartsim/dart/issues/1365
  # TODO: Remove below line and Findurdfdom.cmake once #1365 is resolved in
  # upstream
  # find_package(urdfdom QUIET MODULE)
# else()
  find_package(urdfdom QUIET CONFIG)
# endif()

@PKG_NAME@_INCLUDE_DIRS

message("[DEBUG] urdfdom_FOUND       : ${urdfdom_FOUND}")
message("[DEBUG] urdfdom_INCLUDE_DIRS: ${urdfdom_INCLUDE_DIRS}")
message("[DEBUG] urdfdom_LIBRARIES   : ${urdfdom_LIBRARIES}")
message("[DEBUG] urdfdom_VERSION     : ${urdfdom_VERSION}")

if(urdfdom_FOUND AND NOT TARGET urdfdom)
  add_library(urdfdom INTERFACE IMPORTED)
  set_target_properties(urdfdom PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${urdfdom_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${urdfdom_LIBRARIES}"
  )
endif()
