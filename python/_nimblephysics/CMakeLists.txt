if(NOT DARTPY_PYTHON_VERSION)
 set(DARTPY_PYTHON_VERSION 3.11 CACHE STRING "Choose the target Python version (e.g., 3.8, 3.9)" FORCE)
endif()

# find_package(PythonInterp ${DARTPY_PYTHON_VERSION} QUIET)
# execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
#   "from distutils.sysconfig import get_python_lib;\
#   print(get_python_lib(plat_specific=True, prefix=''))"
#   OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
#   OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# if(NOT PythonInterp_FOUND)
#   message(STATUS "DART_BUILD_DARTPY is ON, but failed to find PythonInterp. "
#     "Disabling _nimblephysics."
#   )
#   return()
# endif()

# find_package(PythonLibs ${DARTPY_PYTHON_VERSION} QUIET)
# if(NOT PythonLibs_FOUND)
#   message(STATUS "DART_BUILD_DARTPY is ON, but failed to find PythonLibs. "
#     "Disabling _nimblephysics."
#   )
#   return()
# endif()

if(APPLE)
  find_program(PYTHON_EXECUTABLE python3.11)
else()
  find_program(PYTHON_EXECUTABLE python3)
endif()

execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" "from distutils.sysconfig import get_python_inc; print(get_python_inc())" OUTPUT_VARIABLE Python3_INCLUDE_DIRS)
message(STATUS "Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}")
execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))" OUTPUT_VARIABLE Python3_LIBRARIES)
message(STATUS "Python3_LIBRARIES=${Python3_LIBRARIES}")
# -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())")
# -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
# find_package(Python3 ${DARTPY_PYTHON_VERSION} EXACT REQUIRED COMPONENTS Interpreter Development)

# Find pybind11
# Needs to set PYBIND11_PYTHON_VERSION before finding pybind11
set(PYBIND11_PYTHON_VERSION ${DARTPY_PYTHON_VERSION})
find_package(pybind11 2.2.0 QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "DART_BUILD_DARTPY is ON, but failed to find pybind11 >= "
    "2.2.0. Disabling _nimblephysics."
  )
  return()
endif()

message(STATUS "PYTHON_INCLUDE_DIRS: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "pybind11_INCLUDE_DIRS: ${pybind11_INCLUDE_DIRS}")

file(GLOB_RECURSE _nimblephysics_headers "*.h" "*.hpp")
file(GLOB_RECURSE _nimblephysics_sources "*.cpp")

# Build a Python extension module:
# pybind11_add_module(<name> [MODULE | SHARED] [EXCLUDE_FROM_ALL]
#                     [NO_EXTRAS] [SYSTEM] [THIN_LTO] source1 [source2 ...])
#
pybind11_add_module(_nimblephysics
  MODULE
  ${_nimblephysics_headers}
  ${_nimblephysics_sources}
)

# Add C++14 support
set_property(TARGET _nimblephysics PROPERTY CXX_STANDARD 14)

target_include_directories(_nimblephysics
  SYSTEM PUBLIC
    ${PYTHON_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(_nimblephysics
  PUBLIC
    dart
    # dart-utils
    dart-utils-urdf
)

set_target_properties(_nimblephysics
  PROPERTIES
    PREFIX ""
    SUFFIX ".so"  # python uses '.so' extension even on macOS
    DEBUG_POSTFIX ""
)

# install(TARGETS _nimblephysics
#   LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
# )

list(REMOVE_ITEM _nimblephysics_headers
  ${CMAKE_CURRENT_LIST_DIR}/eigen_geometry_pybind.h
  ${CMAKE_CURRENT_LIST_DIR}/eigen_pybind.h
)
list(REMOVE_ITEM _nimblephysics_sources
  ${CMAKE_CURRENT_LIST_DIR}/eigen_geometry_pybind.cpp
)
dart_format_add(${_nimblephysics_headers} ${_nimblephysics_sources})
