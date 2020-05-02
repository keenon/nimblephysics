# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

# Find urdfdom
#
# This sets the following variables:
#   urdfdom_FOUND
#   urdfdom_INCLUDE_DIRS
#   urdfdom_LIBRARIES
#   urdfdom_VERSION

find_package(PkgConfig REQUIRED)

# Check if the pkgconfig file is installed
pkg_check_modules(PC_urdfdom urdfdom REQUIRED)

# Include directories
find_path(urdfdom_INCLUDE_DIRS
  NAMES urdf_parser/urdf_parser.h
  HINTS ${PC_urdfdom_INCLUDEDIR}
  PATHS "${CMAKE_INSTALL_PREFIX}/include"
)

# Libraries
set(urdfdom_LIBRARIES )
foreach(lib ${PC_urdfdom_LIBRARIES})
  find_library(urdfdom_LIBRARY NAMES ${lib} HINTS ${PC_urdfdom_LIBDIR})
  list(APPEND urdfdom_LIBRARIES ${urdfdom_LIBRARY})
endforeach()

# Version
set(urdfdom_VERSION ${PC_urdfdom_VERSION})

# Set (NAME)_FOUND if all the variables and the version are satisfied.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(urdfdom
  FAIL_MESSAGE  DEFAULT_MSG
  REQUIRED_VARS urdfdom_INCLUDE_DIRS urdfdom_LIBRARIES
  VERSION_VAR   urdfdom_VERSION
)
