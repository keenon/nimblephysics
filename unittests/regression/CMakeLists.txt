set(test_category regression)

dart_add_tests(
  TYPE ${test_category}
  SOURCES
    test_Issue1243.cpp
    test_Issue892.cpp
    test_Issue895.cpp
    test_Issue986.cpp
    test_Issue1231.cpp
  LINK_LIBRARIES
    dart
)

if(TARGET dart-utils-urdf)
  dart_add_tests(
    TYPE ${test_category}
    SOURCES
      test_Issue838.cpp
    LINK_LIBRARIES
      dart-utils-urdf
  )
endif()

if(TARGET dart-collision-bullet)
  dart_add_tests(
    TYPE ${test_category}
    SOURCES
      test_Issue1184.cpp
    LINK_LIBRARIES
      dart-collision-bullet
  )
endif()

if(TARGET dart-collision-bullet AND TARGET dart-collision-ode AND TARGET dart-utils)
  dart_add_tests(
    TYPE ${test_category}
    SOURCES
      test_Issue1234.cpp
    LINK_LIBRARIES
      dart-collision-bullet
      dart-collision-ode
      dart-utils
  )
endif()
