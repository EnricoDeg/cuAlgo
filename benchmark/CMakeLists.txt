SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -lineinfo -DCUDA ")
if (CUALGO_VERBOSE)
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUALGO_VERBOSE ")
endif()

# dshear1dMatrix
add_executable( benchmark_dshear1dMatrix
                benchmark_dshear1dMatrix.cu
              )
target_include_directories(benchmark_dshear1dMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries (benchmark_dshear1dMatrix cuAlgo)
set_property(TARGET benchmark_dshear1dMatrix PROPERTY CXX_STANDARD 17)
set_property(TARGET benchmark_dshear1dMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries (test_convolution1dMatrix cuAlgo)
# target_link_libraries(test_convolution1dMatrix GTest::gtest_main)
