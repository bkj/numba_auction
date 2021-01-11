cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(auction LANGUAGES C CXX CUDA)

set(ANACONDA_PKG_PATH "$ENV{HOME}/anaconda3/pkgs")
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_PREFIX_PATH  "${ANACONDA_PKG_PATH}/pytorch-1.7.0-py3.7_cuda11.0.221_cudnn8.0.3_0/lib/python3.7/site-packages/torch")
set(CUDNN_LIBRARY_PATH "${ANACONDA_PKG_PATH}/cudnn-8.0.0-cuda11.0_0/lib/libcudnn.so")
set(CUDNN_INCLUDE_PATH "${ANACONDA_PKG_PATH}/cudnn-8.0.0-cuda11.0_0/include")

set(PYTHON_INCLUDE_PATH "$ENV{HOME}/anaconda3/include/python3.7m")
set(PYTHON_LIBRARIES    "$ENV{HOME}/anaconda3/lib/libpython3.7m.so")

find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ./pybind11/include
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# --
# OpenMP

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS          "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# --
# Build auction

add_library(auction SHARED py_auction.cpp)
target_include_directories(
  auction
  PRIVATE
  "${TORCH_INCLUDE_DIRS}"
  ".."
)

target_link_libraries(auction PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
set_target_properties(auction PROPERTIES PREFIX "")