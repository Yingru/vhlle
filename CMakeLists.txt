cmake_minimum_required(VERSION 2.8)
project(vhlle CXX)

#default build type: Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# defult install prefic: ~/.local
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local"
    CACHE PATH "Install path prefix, prepended onto install directories"
     FORCE)
  set(CMAKE_INSTALL_PREFIX_ALT "/usr/lib64"
    CACHE PATH "Install anther path predix"
     FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX} ${CMAKE_INSTALL_PREFIX_ALT}")

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
#find_package(ROOT)
#if(ROOT_FOUND)
#  message(STATUS "Found ROOT libbraty: ${ROOT_LIBRARIES}")
#  include_directories(${ROOT_LIBRARIES})
#  set(ROOT_LIBS ${ROOT_LIBRARIES})
#else()
#  message(FATAL_ERROR "ROOT not found!!")
#endif()


## search for HDF5 C++ library. The CMake FindHDF5 will report success once find C library
find_library(HDF5_CXX_LIB hdf5_cpp)
if("${HDF5_CXX_LIB}" STREQUAL "HDF5_CXX_LIB-NOTFOUND")
  message(STATUS "HDF5 C++ library not found")
else()
  message(STATUS "Found HDF5 C++ library: ${HDF5_CXX_LIB}")
  # Having found the C++ lib, also locate the header and C lib.
  find_path(HDF5_INCLUDE_DIRS H5Cpp.h)
  include_directories(SYSTEM ${HDF5_INCLUDE_DIRS})
  find_library(HDF5_C_LIB hdf5)
  # also link to zlib for HDF5 compression
  set(HDF5_LIBRARIES ${HDF5_CXX_LIB} ${HDF5_C_LIB} z)
  # set preprcessor definition so HDF5 output is compiled.
  add_definitions(-DVHLLE_HDF5)
endif()  

# set default arhitecture flags to optimize for the current system
# but don't override user settings
if(NOT CMAKE_CXX_FLAGS MATCHES "-march|-mtune")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

install(FILES ${PROJECT_NAME}.conf DESTINATION ${PROJECT_NAME})
install(DIRECTORY eos/ DESTINATION ${PROJECT_NAME}/eos/)
#install(FILES ${PROJECT_NAME}.conf DESTINATION bin)
#install(DIRECTORY eos/ DESTINATION bin/eos/)

add_subdirectory(src)
