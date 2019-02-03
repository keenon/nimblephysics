cmake_minimum_required(VERSION 3.5.1)

project(dartpy)

if(NOT DARTPY_PYTHON_VERSION)
  set(DARTPY_PYTHON_VERSION 3.4 CACHE STRING "Choose the target Python version (e.g., 3.4, 2.7)" FORCE)
endif()

find_package(PythonInterp ${DARTPY_PYTHON_VERSION} QUIET)
if(NOT PythonInterp_FOUND)
  message(STATUS "DART_BUILD_DARTPY is ON, but failed to find PythonInterp. "
    "Disabling dartpy."
  )
  return()
endif()
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
  "from distutils.sysconfig import get_python_lib;\
  print(get_python_lib(plat_specific=True, prefix=''))"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(PythonLibs ${DARTPY_PYTHON_VERSION} QUIET)
if(NOT PythonLibs_FOUND)
  message(STATUS "DART_BUILD_DARTPY is ON, but failed to find PythonLibs. "
    "Disabling dartpy."
  )
  return()
endif()

# Find pybind11
# Needs to set PYBIND11_PYTHON_VERSION before finding pybind11
set(PYBIND11_PYTHON_VERSION ${DARTPY_PYTHON_VERSION})
find_package(pybind11 2.2.0 QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "DART_BUILD_DARTPY is ON, but failed to find pybind11 >= "
    "2.2.0. Disabling dartpy."
  )
  return()
endif()

include_directories(SYSTEM
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${PYTHON_INCLUDE_DIRS}
)

#===================
# Chimera Settings
#===================

find_package(chimera QUIET)

# Check if DART is built with octomap so that it supports VoxelGridShape
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_DEFINITIONS "")
set(CMAKE_REQUIRED_FLAGS "-std=c++11")
set(CMAKE_REQUIRED_INCLUDES ${DART_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${DART_LIBRARIES})
check_cxx_source_compiles(
  "
  #include <dart/dart.hpp>
  int main()
  {
    auto voxel = new dart::dynamics::VoxelGridShape();
    delete voxel;
    return 0;
  }
  "
  DART_HAS_VOXELGRIDSHAPE
)

#if(DART_HAS_VOXELGRIDSHAPE)
#  set(CHIMERA_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/chimera/chimera.yml")
#else()
#  set(CHIMERA_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/chimera/chimera_without_voxel.yml")
#endif()
set(CHIMERA_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/chimera/chimera.yml")
if(chimera_FOUND)
  add_chimera_binding(TARGET "${PROJECT_NAME}_CHIMERA"
    MODULE "${PROJECT_NAME}"
    CONFIGURATION "${CHIMERA_CONFIG}"
    SOURCES "chimera/chimera.hpp"
    DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/src/generated"
    DEBUG EXCLUDE_FROM_ALL
  )
  target_include_directories("${PROJECT_NAME}_CHIMERA"
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries("${PROJECT_NAME}_CHIMERA"
    PUBLIC
      ${PYTHON_LIBRARIES}
      dart
      dart-collision-bullet
      dart-gui
  )
  add_custom_target(binding DEPENDS "${PROJECT_NAME}_CHIMERA_REBUILD")
else()
  add_custom_target(binding
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red "ERROR: Cannot generate bindings without \\'chimera\\' installed."
    COMMAND false
  )
endif()

# Create a target to build generated bindings.
set(SOURCES_TXT "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/sources.txt")
if(EXISTS "${SOURCES_TXT}")
  file(STRINGS "${SOURCES_TXT}" SOURCES_GENERATED_RELATIVE NO_HEX_CONVERSION)

  set(SOURCES_GENERATED)
  foreach(relative_path ${SOURCES_GENERATED_RELATIVE})
    list(APPEND SOURCES_GENERATED "${CMAKE_CURRENT_SOURCE_DIR}/src/generated/${relative_path}")
  endforeach()

  add_library("${PROJECT_NAME}" MODULE
    ${SOURCES_GENERATED}
    src/eigen_geometry_py.cpp
#    src/BodyNode.cpp
#    src/Skeleton.cpp
#    src/skel_parser.cpp
#    src/template_registry.cpp
  )
  target_include_directories("${PROJECT_NAME}"
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}
  )
  target_link_libraries("${PROJECT_NAME}"
    PUBLIC
      ${PYTHON_LIBRARIES}
      dart
      dart-collision-bullet
      dart-gui
  )
  set_target_properties("${PROJECT_NAME}" PROPERTIES
    PREFIX ""
    SUFFIX ".so"  # python uses '.so' extension even on macOS
  )
  if(CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options("${PROJECT_NAME}"
      PRIVATE -fabi-version=6
    )
  endif()

  install(TARGETS "${PROJECT_NAME}"
    LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
  )
else()
  add_custom_target(${PROJECT_NAME} ALL
    COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --red "ERROR: Generate bindings using \\'make binding\\' before attempting to build."
    COMMAND false
  )
endif()
