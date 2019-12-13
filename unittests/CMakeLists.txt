#
# Copyright (c) 2011-2019, The DART development contributors
# All rights reserved.
#
# The list of contributors can be found at:
#   https://github.com/dartsim/dart/blob/master/LICENSE
#
# This file is provided under the following "BSD-style" License:
#   Redistribution and use in source and binary forms, with or
#   without modification, are permitted provided that the following
#   conditions are met:
#   * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#   CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
#   INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
#   MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
#   USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
#   AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#   LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#   POSSIBILITY OF SUCH DAMAGE.
#

# GoogleTest setup
include_directories(BEFORE SYSTEM ${CMAKE_SOURCE_DIR}/unittests/gtest/include)
include_directories(BEFORE SYSTEM ${CMAKE_SOURCE_DIR}/unittests/gtest)
add_library(gtest STATIC gtest/src/gtest-all.cc)
add_library(gtest_main STATIC gtest/src/gtest_main.cc)
target_link_libraries(gtest_main gtest)
if(NOT WIN32)
  target_link_libraries(gtest pthread)
endif()
set_target_properties(
  gtest PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

#===============================================================================
# This function uses following global properties:
# - DART_UNITTESTS
# - DART_${test_type}_TESTS
#
# Usage:
#   dart_add_test("unit" test_UnitTestA) # assumed source is test_UnitTestA.cpp
#   dart_add_test("unit" test_UnitTestB test_SourceB1.cpp)
#   dart_add_test("unit" test_UnitTestA test_SourceC1.cpp test_SourceC2.cpp)
#===============================================================================
function(dart_add_test test_type target_name) # ARGN for source files

  dart_property_add(DART_${test_type}_TESTS ${target_name})

  if(${ARGC} GREATER 2)
    set(sources ${ARGN})
  else()
    set(sources "${target_name}.cpp")
  endif()

  add_executable(${target_name} ${sources})
  add_test(${target_name} ${target_name})

  if(MSVC)
    target_link_libraries(${target_name}
        dart
        optimized gtest debug gtestd
        optimized gtest_main debug gtest_maind
    )
  else()
    target_link_libraries(${target_name} dart gtest gtest_main)
  endif()

  dart_format_add(${sources})

endfunction()

#=========================================================================================
# Build multiple tests. The target name will be
# <test_type>_<prefix>_<directories_with_underscores>_<source_filename_without_extension>.
#
# Usage:
#   dart_add_tests(
#     TYPE <test_type>
#     SOURCES <sources>
#     [PREFIX <prefix>]
#     [INCLUDE_DIRS <include_dependencies>]
#     [LINK_LIBRARIES <library_dependencies>]
#     [TEST_LIST <output_var>]
#     [DEPENDENCIES <dependency targets>]
#   )
function(dart_add_tests)
  set(prefix dart_add_tests)
  set(options )
  set(oneValueArgs TYPE TEST_LIST PREFIX)
  set(multiValueArgs SOURCES INCLUDE_DIRS LINK_LIBRARIES DEPENDENCIES)
  cmake_parse_arguments("${prefix}" "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT dart_add_tests_TYPE)
    message(FATAL_ERROR "DEVELOPER ERROR: You must specify a TYPE for your tests!")
  endif()

  set(test_type ${dart_add_tests_TYPE})

  if(NOT DEFINED dart_add_tests_SOURCES)
    message(STATUS "No tests have been specified for ${test_type}")
  else()
    if(dart_add_tests_SOURCES)
      list(SORT dart_add_tests_SOURCES)
    endif()
    list(LENGTH dart_add_tests_SOURCES num_tests)
    set(test_category ${test_type})
    if(dart_add_tests_PREFIX)
      set(test_category ${test_category}_${dart_add_tests_PREFIX})
    endif()
  endif()

  if(dart_add_tests_TEST_LIST)
    set(${dart_add_tests_TEST_LIST} "")
  endif()

  foreach(source ${dart_add_tests_SOURCES})
    set(target_name ${test_type})

    if(dart_add_tests_PREFIX)
      set(target_name ${target_name}_${dart_add_tests_PREFIX})
    endif()

    get_filename_component(source_dir ${source} DIRECTORY)
    if(source_dir)
      string(REPLACE "/" "_" source_prefix ${source_dir})
      set(target_name ${target_name}_${source_prefix})
    endif()

    get_filename_component(source_name ${source} NAME_WE)
    set(target_name ${target_name}_${source_name})

    add_executable(${target_name} ${source})
    target_include_directories(${target_name} PRIVATE ${dart_add_tests_INCLUDE_DIRS})
    add_test(NAME ${target_name} COMMAND ${target_name})

    # Link gtest
    if(MSVC)
      target_link_libraries(${target_name}
        optimized gtest debug gtestd
        optimized gtest_main debug gtest_maind
      )
    else()
      target_link_libraries(${target_name} gtest gtest_main)
    endif()

    target_link_libraries(${target_name} ${dart_add_tests_LINK_LIBRARIES})

    if(dart_add_tests_DEPENDENCIES)
      add_dependencies(${target_name} ${dart_add_tests_DEPENDENCIES})
    endif()

    if(dart_add_tests_TEST_LIST)
      list(APPEND ${dart_add_tests_TEST_LIST} ${target_name})
    endif()

    dart_property_add(DART_${test_type}_TESTS ${target_name})

  endforeach()

  if(dart_add_tests_TEST_LIST)
    set(${dart_add_tests_TEST_LIST} "${${dart_add_tests_TEST_LIST}}" PARENT_SCOPE)
  endif()

  dart_format_add(${dart_add_tests_SOURCES})

endfunction()

#===============================================================================
# Usage:
#   dart_get_tests("comprehensive" compreshensive_tests)
#   foreach(test ${compreshensive_tests})
#     message(STATUS "Test: ${test})
#   endforeach()
#===============================================================================
function(dart_get_tests output_var test_type)
  get_property(var GLOBAL PROPERTY DART_${test_type}_TESTS)
  set(${output_var} ${var} PARENT_SCOPE)
endfunction()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# We categorize tests as:
# - "INTEGRATION": integration tests to verify the combination of several
#   components are correctly performs together
# - "REGRESSION": issue wise tests to verify that the GitHub issues are still
#   fixed even after further changes are made
# - "UNIT": low level tests for one or few classes and functions to verify that
#   they performs correctly as expected
add_subdirectory(integration)
add_subdirectory(regression)
add_subdirectory(unit)

# Print tests
dart_get_tests(integration_tests "INTEGRATION")
dart_get_tests(regression_tests "REGRESSION")
dart_get_tests(unit_tests "UNIT")

if(DART_VERBOSE)
  message(STATUS "")
  message(STATUS "[ Tests ]")
  foreach(test ${comprehensive_tests})
    message(STATUS "Adding test: comprehensive/${test}")
  endforeach()
  foreach(test ${regression_tests})
    message(STATUS "Adding test: regression/${test}")
  endforeach()
  foreach(test ${unit_tests})
    message(STATUS "Adding test: unit/${test}")
  endforeach()
else()
  list(LENGTH comprehensive_tests comprehensive_tests_len)
  list(LENGTH regression_tests regression_tests_len)
  list(LENGTH unit_tests unit_tests_len)
  math(
    EXPR tests_len
    "${comprehensive_tests_len} + ${regression_tests_len} + ${unit_tests_len}"
  )
  message(STATUS "Adding ${tests_len} tests ("
      "comprehensive: ${comprehensive_tests_len}, "
      "regression: ${regression_tests_len}, "
      "unit: ${unit_tests_len}"
      ")"
  )
endif()

# Add custom target to build all the tests as a single target
add_custom_target(
  tests
  DEPENDS ${integration_tests} ${regression_tests} ${unit_tests}
)
