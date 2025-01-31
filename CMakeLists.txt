cmake_minimum_required(VERSION 3.16)

project(fv3d LANGUAGES C CXX)

# Various sets
set(CMAKE_CXX_STANDARD 17)

add_subdirectory(external/kokkos)
add_subdirectory(external/highfive)
include_directories(external/inih)


add_executable(fv3d
  main.cpp)

target_link_libraries(fv3d PRIVATE HighFive Kokkos::kokkos)
