set(CMAKE_CUDA_ARCHITECTURES 80)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -lineinfo -DCUDA ")

add_executable(reductionVector reductionVector.cu)
target_link_libraries (reductionVector cuAlgo)
target_include_directories(reductionVector PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET reductionVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    reductionVector # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(transpositionMatrix transpositionMatrix.cu)
target_link_libraries (transpositionMatrix cuAlgo)
target_include_directories(transpositionMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET transpositionMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    transpositionMatrix # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(generalMatrixMultiplication generalMatrixMultiplication.cu)
target_link_libraries (generalMatrixMultiplication cuAlgo)
target_include_directories(generalMatrixMultiplication PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET generalMatrixMultiplication PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    generalMatrixMultiplication # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(generalMatrixVectorMultiplication generalMatrixVectorMultiplication.cu)
target_link_libraries (generalMatrixVectorMultiplication cuAlgo)
target_include_directories(generalMatrixVectorMultiplication PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET generalMatrixVectorMultiplication PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    generalMatrixVectorMultiplication # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(generalSparseMatrixVectorMultiplicationCSRVector generalSparseMatrixVectorMultiplicationCSRVector.cu)
target_link_libraries (generalSparseMatrixVectorMultiplicationCSRVector cuAlgo)
target_include_directories(generalSparseMatrixVectorMultiplicationCSRVector PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET generalSparseMatrixVectorMultiplicationCSRVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    generalSparseMatrixVectorMultiplicationCSRVector # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(generalSparseMatrixVectorMultiplicationCSRAdaptive generalSparseMatrixVectorMultiplicationCSRAdaptive.cu)
target_link_libraries (generalSparseMatrixVectorMultiplicationCSRAdaptive cuAlgo)
target_include_directories(generalSparseMatrixVectorMultiplicationCSRAdaptive PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET generalSparseMatrixVectorMultiplicationCSRAdaptive PROPERTY CUDA_SEPARABLE_COMPRAdaptive ON)

install (TARGETS
    generalSparseMatrixVectorMultiplicationCSRAdaptive # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(generalSparseMatrixVectorMultiplicationELL generalSparseMatrixVectorMultiplicationELL.cu)
target_link_libraries (generalSparseMatrixVectorMultiplicationELL cuAlgo)
target_include_directories(generalSparseMatrixVectorMultiplicationELL PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET generalSparseMatrixVectorMultiplicationELL PROPERTY CUDA_SEPARABLE_COMPRAdaptive ON)

install (TARGETS
    generalSparseMatrixVectorMultiplicationELL # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(reductionMatrix reductionMatrix.cu)
target_link_libraries (reductionMatrix cuAlgo)
target_include_directories(reductionMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET reductionMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    reductionMatrix # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# convolution1dMatrix
add_executable(perf_convolution1dMatrix perf_convolution1dMatrix.cu)
target_link_libraries(perf_convolution1dMatrix cuAlgo)
target_include_directories(perf_convolution1dMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_convolution1dMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
         perf_convolution1dMatrix # executables
         RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# convolutionReduction1dMatrix
add_executable(perf_convolutionReduction1dMatrix perf_convolutionReduction1dMatrix.cu)
target_link_libraries(perf_convolutionReduction1dMatrix cuAlgo)
target_include_directories(perf_convolutionReduction1dMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_convolutionReduction1dMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_convolutionReduction1dMatrix # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(exclusiveScanVector exclusiveScanVector.cu)
target_link_libraries (exclusiveScanVector cuAlgo)
target_include_directories(exclusiveScanVector PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET exclusiveScanVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    exclusiveScanVector # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(gradientMatrix gradientMatrix.cu)
target_link_libraries (gradientMatrix cuAlgo)
target_include_directories(gradientMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET gradientMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    gradientMatrix # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(taperMatrix taperMatrix.cu)
target_link_libraries (taperMatrix cuAlgo)
target_include_directories(taperMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET taperMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    taperMatrix # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(fftshiftVector fftshiftVector.cu)
target_link_libraries (fftshiftVector cuAlgo)
target_include_directories(fftshiftVector PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET fftshiftVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
         fftshiftVector # executables
         RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)
