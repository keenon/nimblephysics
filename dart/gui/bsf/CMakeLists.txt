# Dependency checks
if (NOT TARGET dart-gui)
  return()
endif()

# OpenSceneGraph
if(DART_BUILD_GUI_BSF)
  find_package(bsf QUIET)
  if (NOT bsf_FOUND)
    message(STATUS "Skipping BS::Framework because bsf is not found")
    return()
  endif()
else()
  message(STATUS "Skipping BS::Framework (DART_BUILD_GUI_BSF == ${DART_BUILD_GUI_BSF})")
  return()
endif()

# Search all header and source files
file(GLOB hdrs "*.hpp")
file(GLOB srcs "*.cpp")

# Set local target name
set(target_name ${PROJECT_NAME}-gui-bsf)
set(component_name gui-bsf)

# Add target
dart_add_library(${target_name} ${hdrs} ${srcs})
target_link_libraries(${target_name} PUBLIC dart-gui bsf::bsf)

# Component
add_component(${PROJECT_NAME} ${component_name})
add_component_targets(${PROJECT_NAME} ${component_name} ${target_name})
add_component_dependencies(${PROJECT_NAME} ${component_name} gui)

# Generate header for this namespace
dart_get_filename_components(header_names "gui bsf headers" ${hdrs})
dart_generate_include_header_file(
  "${CMAKE_CURRENT_BINARY_DIR}/bsf.hpp"
  "dart/gui/bsf/"
  ${header_names}
)

# Install
if(NOT DART_BUILD_DARTPY)
  install(
    FILES ${hdrs} ${CMAKE_CURRENT_BINARY_DIR}/bsf.hpp
    DESTINATION include/dart/gui/bsf
    COMPONENT headers
  )
endif()

dart_format_add(${hdrs} ${srcs})
