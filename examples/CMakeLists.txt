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

