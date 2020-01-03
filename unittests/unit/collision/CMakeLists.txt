set(link_libraries dart)
if(TARGET dart-collision-bullet)
  list(APPEND link_libraries dart-collision-bullet)
endif()
if(TARGET dart-collision-ode)
  list(APPEND link_libraries dart-collision-ode)
endif()

dart_add_tests(
  TYPE ${test_category}
  PREFIX collision
  SOURCES
    test_CollisionGroups.cpp
  LINK_LIBRARIES
    ${link_libraries}
)
