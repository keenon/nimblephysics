cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

add_library(dart my_class.hpp my_class.cpp)
if(CMAKE_VERSION VERSION_LESS 3.8.2)
  target_compile_options(dart PUBLIC -std=c++14)
else()
  target_compile_features(dart PUBLIC cxx_std_14)
endif()
