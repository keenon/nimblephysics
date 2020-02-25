cmake_minimum_required(VERSION 3.5.1)

get_filename_component(example_name ${CMAKE_CURRENT_LIST_DIR} NAME)

project(${example_name})

set(required_components utils-urdf gui-osg)
set(required_libraries dart dart-utils-urdf dart-gui-osg)

if(DART_IN_SOURCE_BUILD)
  dart_build_example_in_source(${example_name} LINK_LIBRARIES ${required_libraries})
  return()
endif()

find_package(DART 6.6.0 REQUIRED COMPONENTS ${required_components} CONFIG)

message(STATUS "[DEBUG] DART_FOUND       : ${DART_FOUND}")
message(STATUS "[DEBUG] DART_VERSION     : ${DART_VERSION}")
message(STATUS "[DEBUG] DART_INCLUDE_DIRS: ${DART_INCLUDE_DIRS}")
message(STATUS "[DEBUG] DART_LIBRARY_DIRS: ${DART_LIBRARY_DIRS}")
message(STATUS "[DEBUG] DART_LIBRARIES   : ${DART_LIBRARIES}")
message(STATUS "[DEBUG] DART_DEFINITIONS : ${DART_DEFINITIONS}")

file(GLOB srcs "*.cpp" "*.hpp")
add_executable(${example_name} ${srcs})
# target_include_directories(${example_name} PUBLIC ${DART_INCLUDE_DIRS})
target_link_libraries(${example_name} PUBLIC ${required_libraries})
# target_link_libraries(${example_name} ${DART_LIBRARIES})
