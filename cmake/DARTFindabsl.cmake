# This file is provided under the "BSD-style" License

find_package(absl CONFIG REQUIRED)

# Set target absl if not set
if((ABSL_FOUND OR absl_FOUND) AND NOT TARGET absl)
  add_library(absl INTERFACE IMPORTED)
  set_target_properties(absl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ABSL_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${ABSL_LIBRARIES}"
  )
endif()
