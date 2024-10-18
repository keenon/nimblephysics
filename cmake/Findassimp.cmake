# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
# https://github.com/dartsim/dart/blob/master/LICENSE
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
find_path(ASSIMP_INCLUDE_DIRS assimp/scene.h
  HINTS /usr/local/include ${PC_ASSIMP_INCLUDEDIR}
  PATHS "${CMAKE_INSTALL_PREFIX}/include")

# Libraries
if(MSVC)
  set(ASSIMP_LIBRARIES "assimp$<$<CONFIG:Debug>:d>")

  # Add IrrXML library for MSVC if necessary
  set(IRRXML_LIBRARIES "IrrXML$<$<CONFIG:Debug>:d>")
else()
  find_library(ASSIMP_LIBRARIES
    NAMES assimp
    HINTS /usr/local/lib ${PC_ASSIMP_LIBDIR})
  find_library(IRRXML_LIBRARIES
    NAMES IrrXML
    HINTS /usr/local/lib ${PC_ASSIMP_LIBDIR})
endif()

# Print the status
message(STATUS "PC_ASSIMP_INCLUDEDIR: ${PC_ASSIMP_INCLUDEDIR}")
message(STATUS "PC_ASSIMP_LIBDIR: ${PC_ASSIMP_LIBDIR}")
message(STATUS "ASSIMP_INCLUDE_DIRS: ${ASSIMP_INCLUDE_DIRS}")
message(STATUS "ASSIMP_LIBRARIES: ${ASSIMP_LIBRARIES}")
message(STATUS "IRRXML_LIBRARIES: ${IRRXML_LIBRARIES}")
message(STATUS "PC_ASSIMP_VERSION: ${PC_ASSIMP_VERSION}")
message(STATUS "ASSIMP_VERSION: ${ASSIMP_VERSION}")

# Version
set(ASSIMP_VERSION ${PC_ASSIMP_VERSION})

# Set (NAME)_FOUND if all the variables and the version are satisfied.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(assimp
  FAIL_MESSAGE DEFAULT_MSG
  REQUIRED_VARS ASSIMP_INCLUDE_DIRS ASSIMP_LIBRARIES IRRXML_LIBRARIES
  VERSION_VAR ASSIMP_VERSION)
