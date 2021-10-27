# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

find_package(GMP QUIET MODULE)

if("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    add_library(gmp SHARED IMPORTED) # or STATIC instead of SHARED
    set_target_properties(gmp PROPERTIES
    IMPORTED_LOCATION "/usr/local/lib/libgmp.a"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/local/share/"
    )
endif()