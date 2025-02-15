if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -lineinfo -DCUDA ")

# dshear1dMatrix
add_executable( benchmark_dshear1dMatrix
                benchmark_dshear1dMatrix.cu
              )
target_include_directories(benchmark_dshear1dMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET benchmark_dshear1dMatrix PROPERTY CXX_STANDARD 17)
set_property(TARGET benchmark_dshear1dMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)
