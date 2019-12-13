dart_add_tests(
  TYPE INTEGRATION
  SOURCES
    test_Building.cpp
    test_Common.cpp
    test_Concurrency.cpp
    test_Constraint.cpp
    test_Frames.cpp
    test_Friction.cpp
    test_InverseKinematics.cpp
    test_NameManagement.cpp
  LINK_LIBRARIES
    dart
)

#if(TARGET dart-optimizer-pagmo)
#  dart_add_tests(
#    TYPE INTEGRATION
#    SOURCES
#      test_MultiObjectiveOptimization.cpp
#    LINK_LIBRARIES
#      dart-optimizer-pagmo
#  )
#endif()

if(TARGET dart-utils)
  dart_add_tests(
    TYPE INTEGRATION
    SOURCES
      test_Collision.cpp
      test_Distance.cpp
      test_Dynamics.cpp
      test_Joints.cpp
      test_Skeleton.cpp
      test_MetaSkeleton.cpp
      test_SoftDynamics.cpp
      test_World.cpp
      test_ForwardKinematics.cpp
    LINK_LIBRARIES
      dart-utils
      $<$<TARGET_EXISTS:dart-collision-bullet>:dart-collision-bullet>
      $<$<TARGET_EXISTS:dart-collision-ode>:dart-collision-ode>
      $<$<TARGET_EXISTS:dart-utils-urdf>:dart-utils-urdf>
  )
endif()

if(TARGET dart-utils-urdf)
  dart_add_library(SharedLibraryWamIkFast SharedLibraryWamIkFast.hpp SharedLibraryWamIkFast.cpp)
  target_link_libraries(SharedLibraryWamIkFast PUBLIC dart)
  target_compile_definitions(SharedLibraryWamIkFast PUBLIC IKFAST_NO_MAIN IKFAST_CLIBRARY)
  target_compile_options(SharedLibraryWamIkFast PRIVATE -w)

  dart_add_library(GeneratedWamIkFast GeneratedWamIkFast.cpp)
  target_link_libraries(GeneratedWamIkFast PUBLIC dart)
  target_compile_definitions(GeneratedWamIkFast PUBLIC IKFAST_NO_MAIN IKFAST_CLIBRARY)
  target_compile_options(GeneratedWamIkFast PRIVATE -w)

  if(BUILD_SHARED_LIBS)
    dart_add_tests(
      TYPE INTEGRATION
      SOURCES
        test_IkFast.cpp
      LINK_LIBRARIES
        dart-utils-urdf
      DEPENDENCIES
        GeneratedWamIkFast SharedLibraryWamIkFast
    )
  endif()
endif()

dart_format_add(
  GeneratedWamIkFast.cpp
  SharedLibraryWamIkFast.hpp
  SharedLibraryWamIkFast.cpp
)
