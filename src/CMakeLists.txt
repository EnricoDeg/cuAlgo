if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -lineinfo -DCUDA ")
if (CUALGO_VERBOSE)
  SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUALGO_VERBOSE ")
endif()

set(SOURCE_EXE  internals/utils.cu
                internals/checkError.cu
                API/reduction1dMatrix.cu
                API/gMatVecMul.cu
                API/gSpMatVecMulCSRVector.cu
                API/gSpMatVecMulCSRAdaptive.cu
                API/gSpMatVecMulELL.cu
                API/grad2dMatrix.cu
                API/histogram.cu
                API/taper1dMatrix.cu
                API/normalizeVector.cu
                API/padarray2dMatrix.cu
                API/upsample1dMatrix.cu
                API/ifftshift2dMatrix.cu)

set(HEADER_FILES internals/checkError.hpp
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

set(HEADERS_CU cuAlgo.hpp cuAlgo.h)
install(FILES ${HEADERS_CU} DESTINATION ${PROJECT_SOURCE_DIR}/include)
