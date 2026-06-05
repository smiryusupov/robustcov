cmake_minimum_required(VERSION 3.18)
project(robustcov_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)
pybind11_add_module(_robustcov_cpp src/robustcov_cpp.cpp)

if(OpenMP_CXX_FOUND)
  target_link_libraries(_robustcov_cpp PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(_robustcov_cpp PRIVATE ROBUSTCOV_OPENMP=1)
endif()


if (MSVC)
  target_compile_options(_robustcov_cpp PRIVATE /O2)
else()
  target_compile_options(_robustcov_cpp PRIVATE -O3 -Wall -Wextra)
endif()

install(TARGETS _robustcov_cpp LIBRARY DESTINATION robustcov)
