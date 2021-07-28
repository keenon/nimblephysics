dart_add_test("unit" test_Aspect)
dart_add_test("unit" test_CollisionGroups)
dart_add_test("unit" test_ContactConstraint)
dart_add_test("unit" test_Factory)
dart_add_test("unit" test_GenericJoints)
dart_add_test("unit" test_Geometry)
dart_add_test("unit" test_Inertia)
dart_add_test("unit" test_Lemke)
dart_add_test("unit" test_LocalResourceRetriever)
dart_add_test("unit" test_Math)
dart_add_test("unit" test_Random)
dart_add_test("unit" test_ScrewJoint)
dart_add_test("unit" test_Signal)
dart_add_test("unit" test_Subscriptions)
dart_add_test("unit" test_Uri)
dart_add_test("unit" test_ConstrainedGroupGradientMatrices)
dart_add_test("unit" test_LCPUtils)
dart_add_test("unit" test_PerformanceLog)
dart_add_test("unit" test_RealtimeUtils)
dart_add_test("unit" test_ScrewGeometry)
dart_add_test("unit" test_JointJacobians)
if(DART_USE_ARBITRARY_PRECISION)
dart_add_test("unit" test_MPFR)
endif()

if(TARGET dart-utils)

  dart_add_test("unit" test_AccelerationSmoother)
  target_link_libraries(test_AccelerationSmoother dart-utils)

  dart_add_test("unit" test_CompositeResourceRetriever)
  target_link_libraries(test_CompositeResourceRetriever dart-utils)

  dart_add_test("unit" test_DartResourceRetriever)
  target_link_libraries(test_DartResourceRetriever dart-utils)

  dart_add_test("unit" test_FileInfoWorld)
  target_link_libraries(test_FileInfoWorld dart-utils)

  dart_add_test("unit" test_PackageResourceRetriever)
  target_link_libraries(test_PackageResourceRetriever dart-utils)

  dart_add_test("unit" test_SdfParser)
  target_link_libraries(test_SdfParser dart-utils)

  dart_add_test("unit" test_SkelParser)
  target_link_libraries(test_SkelParser dart-utils)

  dart_add_test("unit" test_VskParser)
  target_link_libraries(test_VskParser dart-utils)

  dart_add_test("unit" test_AMCParser)
  target_link_libraries(test_AMCParser dart-utils)
endif()

if(TARGET dart-utils-urdf)
  dart_add_test("unit" test_DartLoader)
  target_link_libraries(test_DartLoader dart-utils-urdf)

  dart_add_test("unit" test_DARTCollide)
  target_link_libraries(test_DARTCollide dart-utils)
  target_link_libraries(test_DARTCollide dart-utils-urdf)

  dart_add_test("unit" test_AtlasGradients)
  target_link_libraries(test_AtlasGradients dart-utils)
  target_link_libraries(test_AtlasGradients dart-utils-urdf)

  dart_add_test("unit" test_FreeJointGradients)
  target_link_libraries(test_FreeJointGradients dart-utils)
  target_link_libraries(test_FreeJointGradients dart-utils-urdf)

  dart_add_test("unit" test_UniversalLoader)
  target_link_libraries(test_UniversalLoader dart-utils)
  target_link_libraries(test_UniversalLoader dart-utils-urdf)

  dart_add_test("unit" test_RL_API)
  target_link_libraries(test_RL_API dart-utils)
  target_link_libraries(test_RL_API dart-utils-urdf)

  dart_add_test("unit" test_InverseDynamicsForContact)
  target_link_libraries(test_InverseDynamicsForContact dart-utils)
  target_link_libraries(test_InverseDynamicsForContact dart-utils-urdf)
endif()

if(TARGET dart-planning)
  dart_add_test("unit" test_NearestNeighbor)
  target_link_libraries(test_NearestNeighbor dart-planning)
endif()

foreach(collision_engine
    dart-collision-bullet
    dart-collision-ode)

  if(TARGET ${collision_engine})
    target_link_libraries(test_CollisionGroups ${collision_engine})
  endif()

endforeach()

dart_format_add(test_Inertia.cpp)
dart_format_add(test_VskParser.cpp)
