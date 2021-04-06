# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

find_package(MPFR QUIET MODULE)

if("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    add_library(mpfr SHARED IMPORTED) # or STATIC instead of SHARED
    set_target_properties(mpfr PROPERTIES
    IMPORTED_LOCATION "/usr/local/lib/libmpfr.a"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/local/share/"
    )
endif()