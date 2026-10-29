#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "OPS::ops_seq" for configuration "Debug"
set_property(TARGET OPS::ops_seq APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(OPS::ops_seq PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libops_seq.a"
  )

list(APPEND _cmake_import_check_targets OPS::ops_seq )
list(APPEND _cmake_import_check_files_for_OPS::ops_seq "${_IMPORT_PREFIX}/lib/libops_seq.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
