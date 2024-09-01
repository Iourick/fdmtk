cmake_minimum_required(VERSION 3.8)
set(CUDA_PATH "/usr/bin" CACHE PATH "CUDA Toolkit path")
set(CUDACXX "/usr/local/cuda-12.5/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.5/bin/nvcc")


# Project name
project(fdmtk LANGUAGES CXX )

include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

# Set CUDA path and CUB path
#set(CUDA_PATH "/usr/bin" CACHE PATH "CUDA Toolkit path")
include_directories(/usr/local/cuda-12.5/targets/x86_64-linux/include/cub)

# List of source files
set(SRC_FILES
    fdmtk.cu
    array.cu
    boxcar.cu
    CandidateList.cu
    CandidateSink.cu
    CpuTimer.cu
    cudafdmt.cu
    CudaTimer.cu
    DataSource.cu
    fdmt.cu
    rescale.cu
    SigprocFile.cu
    SigprocFileSet.cu
    fdmt_utils.cu
    npy.hpp
    Candidate.cu
    Clusterization.cu
    Segmented_sort.cu
    select_delegates.cu
)

# Set compiler flags
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O3 -Xcompiler -fPIC -Xcompiler -fopenmp --relocatable-device-code=true -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}")

# Set linker flags
#set(CMAKE_CUDA_LINK_FLAGS "${CMAKE_CUDA_LINK_FLAGS} --cudart static --relocatable-device-code=true -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=compute_${CMAKE_CUDA_ARCHITECTURES} -gencode arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES} -link -Xcompiler -fopenmp")

# Add executable target
add_executable(CUFDMT ${SRC_FILES})

# Link CUDA libraries

