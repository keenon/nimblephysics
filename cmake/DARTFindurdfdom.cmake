# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

find_package(urdfdom QUIET CONFIG)

if(MSVC)
  list(REMOVE_ITEM urdfdom_INCLUDE_DIRS "/include")

  find_package(tinyxml REQUIRED)
  list(APPEND urdfdom_INCLUDE_DIRS ${tinyxml_INCLUDE_DIRS})

  message(STATUS "[DEBUG] CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "[DEBUG] urdfdom_INCLUDE_DIRS: ${urdfdom_INCLUDE_DIRS}")
  message(STATUS "[DEBUG] TinyXML_INCLUDE_DIRS: ${TinyXML_INCLUDE_DIRS}")
  message(STATUS "[DEBUG] urdfdom_headers_INCLUDE_DIRS: ${urdfdom_headers_INCLUDE_DIRS}")
  message(STATUS "[DEBUG] console_bridge_INCLUDE_DIRS: ${console_bridge_INCLUDE_DIRS}")
endif()

if(urdfdom_FOUND AND NOT TARGET urdfdom)
  add_library(urdfdom INTERFACE IMPORTED)
  set_target_properties(urdfdom PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${urdfdom_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${urdfdom_LIBRARIES}"
  )
endif()
