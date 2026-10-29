# Install script for directory: C:/Users/44793/VScode/OPS

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/OPS")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/MSYS2/mingw64/bin/objdump.exe")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/44793/VScode/OPS/build/ops_translator_legacy/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/44793/VScode/OPS/build/ops/c/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_checkpointing.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_cuda_reduction.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_cuda_rt_support.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_device_rt_support.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_distributions.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_exceptions.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_grid_part_seq.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_grid_part_seq_init.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_grid_part_seq_v2.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_hdf5.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_hdf5_common.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_hip_reduction.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_hip_rt_support.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_insert_random_particles.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_instance.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_internal1.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_internal2.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_lib_core.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_macros.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_mpi_core.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_mpi_particle_core.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_opencl_reduction.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_opencl_rt_support.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_particle_grid_seq.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_particle_insert_del.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_particle_internal.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_particle_seq.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_particle_seq_original.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_particle_seq_v1.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_particles_lib_core.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_seq.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_seq_v2.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_sequential_support.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_sycl_reduction.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_sycl_rt_support.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_tridiag.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/ops_util.h"
    "C:/Users/44793/VScode/OPS/ops/c/include/queue.h"
    "C:/Users/44793/VScode/OPS/ops/fortran/include/ops_macros.inc"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "C:/Program Files (x86)/OPS/lib/cmake/OPSConfig.cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "C:/Program Files (x86)/OPS/lib/cmake" TYPE FILE FILES "C:/Users/44793/VScode/OPS/build/OPSConfig.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/Users/44793/VScode/OPS/build/apps/c/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/44793/VScode/OPS/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
