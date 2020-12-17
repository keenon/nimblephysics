# Search all header and source files
file(GLOB hdrs "*.hpp")
file(GLOB srcs "*.cpp")
dart_add_core_headers(${hdrs})
dart_add_core_sources(${srcs})
dart_add_core_sources("${CMAKE_SOURCE_DIR}/dart/server/external/jsoncpp.cpp")
dart_add_core_sources("${CMAKE_SOURCE_DIR}/dart/server/external/base64/base64.cpp")

# Generate header for this namespace
dart_get_filename_components(header_names "server headers" ${hdrs})
dart_generate_include_header_file(
  "${CMAKE_CURRENT_BINARY_DIR}/server.hpp"
  "dart/server/"
  ${header_names}
)

# For websockets
SET(CMAKE_CXX_FLAGS -pthread)

# Install
install(
  FILES ${hdrs} ${CMAKE_CURRENT_BINARY_DIR}/server.hpp
  DESTINATION include/dart/server
  COMPONENT headers
)