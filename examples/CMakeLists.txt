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

# exclusiveScan1dVector
add_executable(perf_exclusiveScan1dVector perf_exclusiveScan1dVector.cu)
target_link_libraries(perf_exclusiveScan1dVector cuAlgo)
target_include_directories(perf_exclusiveScan1dVector PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_exclusiveScan1dVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_exclusiveScan1dVector # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# fftshift1dVector
add_executable(perf_fftshift1dVector perf_fftshift1dVector.cu)
target_link_libraries (perf_fftshift1dVector cuAlgo)
target_include_directories(perf_fftshift1dVector PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_fftshift1dVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_fftshift1dVector # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# gMatMul
add_executable(perf_gMatMul perf_gMatMul.cu)
target_link_libraries (perf_gMatMul cuAlgo)
target_include_directories(perf_gMatMul PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_gMatMul PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_gMatMul # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# gMatVecMul
add_executable(perf_gMatVecMul perf_gMatVecMul.cu)
target_link_libraries (perf_gMatVecMul cuAlgo)
target_include_directories(perf_gMatVecMul PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_gMatVecMul PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_gMatVecMul # executables
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
