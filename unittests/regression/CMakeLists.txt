dart_add_test("regression" test_Issue000Template test_Issue000Template.cpp)

if(TARGET dart-utils-urdf)

  dart_add_test("regression" test_Issue838)
  target_link_libraries(test_Issue838 dart-utils-urdf)

  dart_add_test("regression" test_Issue895)

  dart_add_test("regression" test_Issue986)

  dart_add_test("regression" test_Issue1193)

endif()

dart_format_add(test_Issue1193.cpp)
