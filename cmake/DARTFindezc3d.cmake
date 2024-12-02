# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the "BSD-style" License

find_package(ezc3d REQUIRED)

# Set target ezc3d if not set
if((EZC3D_FOUND OR ezc3d_FOUND) AND NOT TARGET ezc3d)
  add_library(ezc3d INTERFACE IMPORTED)
  set_target_properties(ezc3d PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${EZC3D_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${EZC3D_LIBRARIES}"
  )
endif()
