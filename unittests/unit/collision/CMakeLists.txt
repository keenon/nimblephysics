dart_add_tests(
  TYPE ${test_category}
  PREFIX collision
  SOURCES
    test_CollisionGroups.cpp
  LINK_LIBRARIES
    dart
    $<$<TARGET_EXISTS:dart-collision-bullet>:dart-collision-bullet>
    $<$<TARGET_EXISTS:dart-collision-ode>:dart-collision-ode>
)
