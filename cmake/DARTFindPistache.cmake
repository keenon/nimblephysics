# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

find_package(Pistache QUIET CONFIG)

# TODO: make this more flexible
set(Pistache_LIBRARIES "/usr/local/lib/libpistache-0.0.002-git20200802.so")
set(Pistache_INCLUDE_DIRS "/usr/local/include/pistache")

if(Pistache_FOUND AND NOT TARGET Pistache::Pistache)
  message(STATUS "Pistache_FOUND: ${Pistache_FOUND}")
  message(STATUS "Pistache_INCLUDE_DIRS: ${Pistache_INCLUDE_DIRS}")
  message(STATUS "Pistache_LIBRARIES: ${Pistache_LIBRARIES}")
  add_library(Pistache::Pistache INTERFACE IMPORTED)
  set_target_properties(Pistache::Pistache PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${Pistache_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${Pistache_LIBRARIES}"
  )
endif()
