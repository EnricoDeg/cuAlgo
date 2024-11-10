set(CMAKE_CUDA_ARCHITECTURES 80)
SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -lineinfo -DCUDA ")

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
target_link_libraries(perf_fftshift1dVector cuAlgo)
target_include_directories(perf_fftshift1dVector PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_fftshift1dVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_fftshift1dVector # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# gMatMul
add_executable(perf_gMatMul perf_gMatMul.cu)
target_link_libraries(perf_gMatMul cuAlgo)
target_include_directories(perf_gMatMul PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_gMatMul PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_gMatMul # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# gMatVecMul
add_executable(perf_gMatVecMul perf_gMatVecMul.cu)
target_link_libraries(perf_gMatVecMul cuAlgo)
target_include_directories(perf_gMatVecMul PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_gMatVecMul PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_gMatVecMul # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# gSpMatVecMulCSRAdaptive
add_executable(perf_gSpMatVecMulCSRAdaptive perf_gSpMatVecMulCSRAdaptive.cu)
target_link_libraries(perf_gSpMatVecMulCSRAdaptive cuAlgo)
target_include_directories(perf_gSpMatVecMulCSRAdaptive PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_gSpMatVecMulCSRAdaptive PROPERTY CUDA_SEPARABLE_COMPRAdaptive ON)

install(TARGETS
        perf_gSpMatVecMulCSRAdaptive # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# gSpMatVecMulCSRVector
add_executable(perf_gSpMatVecMulCSRVector perf_gSpMatVecMulCSRVector.cu)
target_link_libraries(perf_gSpMatVecMulCSRVector cuAlgo)
target_include_directories(perf_gSpMatVecMulCSRVector PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_gSpMatVecMulCSRVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_gSpMatVecMulCSRVector # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# gSpMatVecMulELL
add_executable(perf_gSpMatVecMulELL perf_gSpMatVecMulELL.cu)
target_link_libraries(perf_gSpMatVecMulELL cuAlgo)
target_include_directories(perf_gSpMatVecMulELL PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_gSpMatVecMulELL PROPERTY CUDA_SEPARABLE_COMPRAdaptive ON)

install(TARGETS
        perf_gSpMatVecMulELL # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# grad2dMatrix
add_executable(perf_grad2dMatrix perf_grad2dMatrix.cu)
target_link_libraries(perf_grad2dMatrix cuAlgo)
target_include_directories(perf_grad2dMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_grad2dMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_grad2dMatrix # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# reduction1dMatrix
add_executable(perf_reduction1dMatrix perf_reduction1dMatrix.cu)
target_link_libraries(perf_reduction1dMatrix cuAlgo)
target_include_directories(perf_reduction1dMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_reduction1dMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_reduction1dMatrix # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# reduction1dVector
add_executable(perf_reduction1dVector perf_reduction1dVector.cu)
target_link_libraries(perf_reduction1dVector cuAlgo)
target_include_directories(perf_reduction1dVector PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_reduction1dVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_reduction1dVector # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# taper1dMatrix
add_executable(perf_taper1dMatrix perf_taper1dMatrix.cu)
target_link_libraries(perf_taper1dMatrix cuAlgo)
target_include_directories(perf_taper1dMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_taper1dMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_taper1dMatrix # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

# transposeMatrix
add_executable(perf_transposeMatrix perf_transposeMatrix.cu)
target_link_libraries(perf_transposeMatrix cuAlgo)
target_include_directories(perf_transposeMatrix PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET perf_transposeMatrix PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(TARGETS
        perf_transposeMatrix # executables
        RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)