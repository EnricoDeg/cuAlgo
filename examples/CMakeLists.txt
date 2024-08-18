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

add_executable(convolutionMatrix convolutionMatrix.cu)
target_link_libraries (convolutionMatrix cuAlgo)
target_include_directories(convolutionMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET convolutionMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    convolutionMatrix # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

add_executable(convolutionReductionMatrix convolutionReductionMatrix.cu)
target_link_libraries (convolutionReductionMatrix cuAlgo)
target_include_directories(convolutionReductionMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET convolutionReductionMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
    convolutionReductionMatrix # executables
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)
