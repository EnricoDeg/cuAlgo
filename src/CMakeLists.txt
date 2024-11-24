set(CMAKE_CUDA_ARCHITECTURES 80)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -lineinfo -DCUDA ")
if (CUALGO_VERBOSE)
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUALGO_VERBOSE ")
endif()

set(SOURCE_EXE  internals/utils.cu
                API/getRowBlocks.cu
                internals/checkError.cu
                API/reduction1dVector.cu
                API/normL1Vector.cu
                API/reduction1dMatrix.cu
                API/convolution1dMatrix.cu
                API/correlation1dMatrix.cu
                API/convolutionReduction1dMatrix.cu
                API/convolutionTaperReduction1dMatrix.cu
                API/exclusiveScan1dVector.cu
                API/transposeMatrix.cu
                API/gMatMul.cu
                API/gMatVecMul.cu
                API/gSpMatVecMulCSRVector.cu
                API/gSpMatVecMulCSRAdaptive.cu
                API/gSpMatVecMulELL.cu
                API/grad2dMatrix.cu
                API/taper1dMatrix.cu
                API/fftshift1dVector.cu
                API/normL2Vector.cu
                API/normLInfVector.cu
                API/dotProduct1dVector.cu
                API/normalizeVector.cu
                API/convolution2dMatrixDD.cu
                API/dshear1dMatrix.cu
                API/downsample1dMatrix.cu
                API/fftshift2dMatrix.cu
                API/fliplr1dMatrix.cu)

set(HEADER_FILES internals/checkError.hpp
                 internals/cuAlgoInternal.hpp
                 internals/gOperationAndReduction1dVector.hpp
                 internals/gReduction1dVector.hpp
                 internals/kernelParameters.hpp
                 internals/operations.hpp
                 internals/reductionShMem.hpp
                 internals/templateShMem.hpp
                 internals/utils.hpp)

add_library(cuAlgo SHARED ${SOURCE_EXE} ${HEADER_FILES})
target_include_directories(cuAlgo PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET cuAlgo PROPERTY CXX_STANDARD 17)
set_property(TARGET cuAlgo PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Install the library
install (
  TARGETS cuAlgo
  LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/lib
  ARCHIVE DESTINATION ${PROJECT_SOURCE_DIR}/lib)

set(HEADERS_CU cuAlgo.hpp)
install(FILES ${HEADERS_CU} DESTINATION ${PROJECT_SOURCE_DIR}/include)
