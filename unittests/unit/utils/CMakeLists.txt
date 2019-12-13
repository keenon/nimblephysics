dart_add_tests(
  TYPE ${test_category}
  PREFIX utils
  SOURCES
    test_CompositeResourceRetriever.cpp
    test_DartResourceRetriever.cpp
    test_FileInfoWorld.cpp
    test_PackageResourceRetriever.cpp
    test_MjcfParser.cpp
    test_SdfParser.cpp
    test_SkelParser.cpp
    test_VskParser.cpp
  LINK_LIBRARIES
    dart-utils
)

if(TARGET dart-utils-urdf)
  dart_add_tests(
    TYPE ${test_category}
    PREFIX utils
    SOURCES
      test_DartLoader.cpp
    LINK_LIBRARIES
      dart-utils-urdf
  )
endif()
