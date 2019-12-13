file(GLOB_RECURSE test_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "test_*.cpp")

dart_add_tests(
  TYPE ${test_category}
  PREFIX dynamics
  SOURCES ${test_sources}
  LINK_LIBRARIES
    dart
)
