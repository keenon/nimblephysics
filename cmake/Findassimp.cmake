# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

# Find ASSIMP
#
# This sets the following variables:
# ASSIMP_FOUND
# ASSIMP_INCLUDE_DIRS
# ASSIMP_LIBRARIES
# ASSIMP_VERSION

find_package(PkgConfig QUIET)

# Check to see if pkgconfig is installed.
pkg_check_modules(PC_ASSIMP assimp QUIET)

# Include directories
if(MSVC)
  find_path(
    ASSIMP_INCLUDE_DIR
    NAMES assimp/scene.h
    PATHS $ENV{PROGRAMFILES}/include)
else()
  find_path(ASSIMP_INCLUDE_DIRS assimp/scene.h
      HINTS ${PC_ASSIMP_INCLUDEDIR}
      PATHS "${CMAKE_INSTALL_PREFIX}/include")
endif()

# Libraries
if(MSVC)
  find_library(ASSIMP_LIBRARIES
    NAMES assimp
    PATHS $ENV{PROGRAMFILES}/lib)
else()
  find_library(ASSIMP_LIBRARIES
      NAMES assimp
      HINTS ${PC_ASSIMP_LIBDIR})
endif()

# Version
set(ASSIMP_VERSION ${PC_ASSIMP_VERSION})

# Set (NAME)_FOUND if all the variables and the version are satisfied.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(assimp
    FAIL_MESSAGE  DEFAULT_MSG
    REQUIRED_VARS ASSIMP_INCLUDE_DIRS ASSIMP_LIBRARIES
    VERSION_VAR   ASSIMP_VERSION)
