# Search all header and source files
file(GLOB hdrs "*.hpp")
file(GLOB srcs "*.cpp")
dart_add_core_headers(${hdrs})
dart_add_core_sources(${srcs})

# dart_add_utils_headers(${hdrs})
# dart_add_utils_headers(${srcs})

# Generate header for this namespace
dart_get_filename_components(header_names "utils_amc headers" ${hdrs})
dart_generate_include_header_file(
  "${CMAKE_CURRENT_BINARY_DIR}/amc.hpp"
  "dart/utils/amc/"
  ${header_names}
)

set(target_name ${PROJECT_NAME}-utils-amc)

# dart_add_library(${target_name} ${hdrs} ${srcs})

# Install
install(
  FILES ${hdrs} ${CMAKE_CURRENT_BINARY_DIR}/amc.hpp
  DESTINATION include/dart/utils/amc
  COMPONENT headers
)
