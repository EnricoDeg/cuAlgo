set(CMAKE_CUDA_ARCHITECTURES 80)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -lineinfo -DCUDA ")
if (CUALGO_VERBOSE)
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUALGO_VERBOSE ")
endif()

set(SOURCE_EXE  utils.cu
                getRowBlocks.cu
                checkError.cu
                reduction1dVector.cu
                reduction1dMatrix.cu
                convolution1dMatrix.cu
                convolutionReduction1dMatrix.cu
                convolutionTaperReduction1dMatrix.cu
                exclusiveScan1dVector.cu
                transposeMatrix.cu
                gMatMul.cu
                gMatVecMul.cu
                gSpMatVecMulCSRVector.cu
                gSpMatVecMulCSRAdaptive.cu
                gSpMatVecMulELL.cu
                grad2dMatrix.cu
                taper1dMatrix.cu
                fftshift1dVector.cu)

add_library(cuAlgo SHARED ${SOURCE_EXE})
target_include_directories(cuAlgo PRIVATE ${PROJECT_SOURCE_DIR})
set_property(TARGET cuAlgo PROPERTY CXX_STANDARD 17)
set_property(TARGET cuAlgo PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Install the library
install (
  TARGETS cuAlgo
  LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
  ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS_CU cuAlgo.hpp)
install(FILES ${HEADERS_CU} DESTINATION ${PROJECT_SOURCE_DIR}/include)
