# cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(dart)

add_library(dart my_class.hpp my_class.cpp)
if(CMAKE_VERSION VERSION_LESS 3.8.2)
  target_compile_options(dart PUBLIC -std=c++14)
else()
  target_compile_features(dart PUBLIC cxx_std_14)
endif()

include(GNUInstallDirs)
install(TARGETS dart
  EXPORT dart-targets
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(EXPORT dart-targets
  FILE dart-targets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/DARTConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
