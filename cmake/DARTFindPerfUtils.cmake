# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

find_package(PerfUtils QUIET CONFIG)

if(PerfUtils_FOUND AND NOT TARGET PerfUtils)
  add_library(PerfUtils INTERFACE IMPORTED)
  set_target_properties(PerfUtils PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${PerfUtils_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${PerfUtils_LIBRARIES}"
  )
  add_compile_definitions(HAVE_PERF_UTILS)
endif()
