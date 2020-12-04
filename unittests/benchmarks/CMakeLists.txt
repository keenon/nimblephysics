find_package(benchmark REQUIRED)

dart_add_test("benchmarks" bench_Basic)
dart_add_test("benchmarks" bench_Featherstone)
dart_add_test("benchmarks" bench_Jacobians)

target_link_libraries(bench_Basic benchmark::benchmark)
target_link_libraries(bench_Featherstone benchmark::benchmark)
target_link_libraries(bench_Jacobians benchmark::benchmark)