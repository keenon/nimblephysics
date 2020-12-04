# Search all proto files
include(FindProtobuf)

# Enable protobuf
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

file(GLOB PROTOBUF_FILELIST "*.proto")

# Find gRPC installation
# Looks for gRPCConfig.cmake file installed by gRPC's cmake installation.
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
# Find the GRPC executable
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)

find_program(_PROTOC_EXECUTABLE protoc
             HINTS ${CMAKE_INSTALL_PREFIX}/bin
             REQUIRED)

message(STATUS "Attempting to generate the proto")
message(STATUS "_GRPC_CPP_PLUGIN_EXECUTABLE=${_GRPC_CPP_PLUGIN_EXECUTABLE}")
execute_process(COMMAND "${_PROTOC_EXECUTABLE}" "--proto_path=${CMAKE_CURRENT_SOURCE_DIR}" "--grpc_out=." "--plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}" "--cpp_out=." "${PROTOBUF_FILELIST}" WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Search all header and source files
file(GLOB hdrs "*.hpp" "*.pb.h")
file(GLOB srcs "*.cpp" "*.pb.cc")
message(STATUS ${hdrs})
dart_add_core_headers(${hdrs})
dart_add_core_sources(${srcs})

# Generate header for this namespace
dart_get_filename_components(header_names "proto headers" ${hdrs})
dart_generate_include_header_file(
  "${CMAKE_CURRENT_BINARY_DIR}/proto.hpp"
  "dart/proto/"
  ${header_names}
)

# Install
install(
  FILES ${hdrs} ${CMAKE_CURRENT_BINARY_DIR}/proto.hpp
  DESTINATION include/dart/proto
  COMPONENT headers
)
