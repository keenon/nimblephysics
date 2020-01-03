file(GLOB_RECURSE test_sources RELATIVE "${CMAKE_CURRENT_LIST_DIR}" "test_*.cpp")

set(link_libraries dart)
if(TARGET dart-optimizer-ipopt)
  list(APPEND link_libraries dart-optimizer-ipopt)
endif()
if(TARGET dart-optimizer-nlopt)
  list(APPEND link_libraries dart-optimizer-nlopt)
endif()

dart_add_tests(
  TYPE ${test_category}
  PREFIX optimizer
  SOURCES ${test_sources}
  LINK_LIBRARIES ${link_libraries}
)
