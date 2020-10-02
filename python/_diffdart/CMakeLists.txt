if(NOT DARTPY_PYTHON_VERSION)
  set(DARTPY_PYTHON_VERSION 3.4 CACHE STRING "Choose the target Python version (e.g., 3.4, 2.7)" FORCE)
endif()

find_package(PythonInterp ${DARTPY_PYTHON_VERSION} QUIET)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
  "from distutils.sysconfig import get_python_lib;\
  print(get_python_lib(plat_specific=True, prefix=''))"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT PythonInterp_FOUND)
  message(STATUS "DART_BUILD_DARTPY is ON, but failed to find PythonInterp. "
    "Disabling _diffdart."
  )
  return()
endif()

find_package(PythonLibs ${DARTPY_PYTHON_VERSION} QUIET)
if(NOT PythonLibs_FOUND)
  message(STATUS "DART_BUILD_DARTPY is ON, but failed to find PythonLibs. "
    "Disabling _diffdart."
  )
  return()
endif()

# Find pybind11
# Needs to set PYBIND11_PYTHON_VERSION before finding pybind11
set(PYBIND11_PYTHON_VERSION ${DARTPY_PYTHON_VERSION})
find_package(pybind11 2.2.0 QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "DART_BUILD_DARTPY is ON, but failed to find pybind11 >= "
    "2.2.0. Disabling _diffdart."
  )
  return()
endif()

file(GLOB_RECURSE _diffdart_headers "*.h" "*.hpp")
file(GLOB_RECURSE _diffdart_sources "*.cpp")

# Build a Python extension module:
# pybind11_add_module(<name> [MODULE | SHARED] [EXCLUDE_FROM_ALL]
#                     [NO_EXTRAS] [SYSTEM] [THIN_LTO] source1 [source2 ...])
#
pybind11_add_module(_diffdart
  MODULE
  ${_diffdart_headers}
  ${_diffdart_sources}
)

# Add C++14 support
set_property(TARGET _diffdart PROPERTY CXX_STANDARD 14)

target_include_directories(_diffdart
  SYSTEM PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(_diffdart
  PUBLIC
    dart
    dart-utils
    dart-utils-urdf
    dart-gui
    dart-gui-osg
)
if(TARGET dart-optimizer-nlopt)
  target_link_libraries(_diffdart PUBLIC dart-optimizer-nlopt)
endif()
if(TARGET dart-collision-bullet)
  target_link_libraries(_diffdart PUBLIC dart-collision-bullet)
endif()
if(TARGET dart-collision-ode)
  target_link_libraries(_diffdart PUBLIC dart-collision-ode)
endif()

set_target_properties(_diffdart
  PROPERTIES
    PREFIX ""
    SUFFIX ".so"  # python uses '.so' extension even on macOS
    DEBUG_POSTFIX ""
)

install(TARGETS _diffdart
  LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
)

list(REMOVE_ITEM _diffdart_headers
  ${CMAKE_CURRENT_LIST_DIR}/eigen_geometry_pybind.h
  ${CMAKE_CURRENT_LIST_DIR}/eigen_pybind.h
)
list(REMOVE_ITEM _diffdart_sources
  ${CMAKE_CURRENT_LIST_DIR}/eigen_geometry_pybind.cpp
)
dart_format_add(${_diffdart_headers} ${_diffdart_sources})
