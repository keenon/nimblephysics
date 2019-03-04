# Dependent target checks
if (NOT TARGET dart-utils OR NOT TARGET dart-gui-osg)
  return()
endif()

# Search all header and source files
file(GLOB hdrs "*.hpp" "*.h" "detail/*.hpp")
file(GLOB srcs "*.cpp" "*.c" "detail/*.cpp")

# Set local target name
set(target_name ${PROJECT_NAME}-simulator)
set(component_name simulator)

# Add target
dart_add_library(${target_name} ${hdrs} ${srcs})
# target_include_directories(${target_name} PUBLIC ${})
target_link_libraries(${target_name}
  PUBLIC
    dart-utils
    dart-gui-osg
    ${PROJECT_NAME}-external-imgui
)

# Component
add_component(${PROJECT_NAME} ${component_name})
add_component_targets(${PROJECT_NAME} ${component_name} ${target_name})
add_component_dependencies(
  ${PROJECT_NAME}
  ${component_name}
  utils
  gui-osg
)

# Generate header for this namespace
dart_get_filename_components(header_names "simulator headers" ${hdrs})

dart_generate_include_header_file(
  "${CMAKE_CURRENT_BINARY_DIR}/simulator.hpp"
  "dart/simulator/"
  ${header_names}
)

# Install
install(
  FILES ${hdrs} ${CMAKE_CURRENT_BINARY_DIR}/simulator.hpp
  DESTINATION include/dart/simulator
  COMPONENT headers
)

add_executable(dartsim main.cpp)
target_link_libraries(dartsim PRIVATE ${target_name})

dart_format_add(${hdrs} ${srcs})
