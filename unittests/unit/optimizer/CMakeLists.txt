file(GLOB_RECURSE test_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "test_*.cpp")

dart_add_tests(
  TYPE ${test_category}
  PREFIX optimizer
  SOURCES ${test_sources}
  LINK_LIBRARIES
    dart
    $<$<TARGET_EXISTS:dart-optimizer-ipopt>:dart-optimizer-ipopt>
    $<$<TARGET_EXISTS:dart-optimizer-nlopt>:dart-optimizer-nlopt>
)
