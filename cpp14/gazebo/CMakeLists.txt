cmake_minimum_required(VERSION 2.8.6 FATAL_ERROR)

project(gazebo)

# Find DART
set(DART_MIN_REQUIRED_VERSION 6.6)
find_package(DART ${DART_MIN_REQUIRED_VERSION} CONFIG)
if(DART_FOUND)
  message(STATUS "Looking for DART - found")
  set(HAVE_DART TRUE)
else()
  message(STATUS "Looking for DART - not found")
  message(STATUS
    "DART ${DART_MIN_REQUIRED_VERSION} not found, for dart physics engine option, please install libdart${DART_MIN_REQUIRED_VERSION}-dev."
    )
  set(HAVE_DART FALSE)
endif()

add_executable(gazebo main.cpp)

# Link in DART support if present
if(HAVE_DART)
  target_link_libraries(gazebo ${DART_LIBRARIES})
endif()
