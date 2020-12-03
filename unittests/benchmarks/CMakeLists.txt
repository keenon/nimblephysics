find_package(benchmark REQUIRED)

dart_add_test("benchmarks" bench_Basic)
dart_add_test("benchmarks" bench_Featherstone)

target_link_libraries(bench_Basic benchmark::benchmark)
target_link_libraries(bench_Featherstone benchmark::benchmark)