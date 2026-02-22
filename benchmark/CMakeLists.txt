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

# fft1dCT
add_executable( benchmark_fft1dCT
                benchmark_fft1dCT.cu
              )
target_include_directories(benchmark_fft1dCT PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET benchmark_fft1dCT PROPERTY CXX_STANDARD 17)
set_property(TARGET benchmark_fft1dCT PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# fft1dStockham
add_executable( benchmark_fft1dStockham
                benchmark_fft1dStockham.cu
              )
target_include_directories(benchmark_fft1dStockham PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET benchmark_fft1dStockham PROPERTY CXX_STANDARD 17)
set_property(TARGET benchmark_fft1dStockham PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# fft1dBailey
add_executable( benchmark_fft1dBailey
                benchmark_fft1dBailey.cu
              )
target_include_directories(benchmark_fft1dBailey PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET benchmark_fft1dBailey PROPERTY CXX_STANDARD 17)

# fft1dCTr2c
add_executable( benchmark_fft1dCTr2c
                benchmark_fft1dCTr2c.cu
              )
target_include_directories(benchmark_fft1dCTr2c PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET benchmark_fft1dCTr2c PROPERTY CXX_STANDARD 17)
set_property(TARGET benchmark_fft1dCTr2c PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# convFFT1dCT
add_executable( benchmark_convFFT1dCT
                benchmark_convFFT1dCT.cu
              )
target_include_directories(benchmark_convFFT1dCT PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET benchmark_convFFT1dCT PROPERTY CXX_STANDARD 17)
set_property(TARGET benchmark_convFFT1dCT PROPERTY CUDA_SEPARABLE_COMPILATION ON)
