cmake_minimum_required(VERSION 2.6)
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


# set default arhitecture flags to optimize for the current system
# but don't override user settings
if(NOT CMAKE_CXX_FLAGS MATCHES "-march|-mtune")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

install(FILES ${PROJECT_NAME}.conf DESTINATION ${PROJECT_NAME})
install(DIRECTORY eos/ DESTINATION ${PROJECT_NAME})
add_subdirectory(src)
