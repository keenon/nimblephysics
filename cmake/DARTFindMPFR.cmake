# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
# https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

find_package(mpfr QUIET MODULE)

message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")

if("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    add_library(mpfr SHARED IMPORTED) # or STATIC instead of SHARED
    set_target_properties(mpfr PROPERTIES
        IMPORTED_LOCATION "/usr/local/lib/libmpfr.a"
        INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include/"

        # INTERFACE_INCLUDE_DIRECTORIES "/usr/local/share/"
    )
endif()

# message(STATUS "mpfr_FOUND: ${mpfr_FOUND}")

# message(STATUS "mpfr_INCLUDE_DIRS: ${mpfr_INCLUDE_DIRS}")
# message(STATUS "mpfr_LIBRARIES: ${mpfr_LIBRARIES}")

# if(mpfr_FOUND AND NOT TARGET mpfr::mpfr)
# add_library(mpfr::mpfr INTERFACE IMPORTED)
# set_target_properties(mpfr::mpfr PROPERTIES
# INTERFACE_INCLUDE_DIRECTORIES "${mpfr_INCLUDE_DIRS}"
# INTERFACE_LINK_LIBRARIES "${mpfr_LIBRARIES}"
# )
# endif()