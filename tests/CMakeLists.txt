# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# Do NOT install gtest with the project
option(INSTALL_GTEST "Enable installation of googletest." OFF)
FetchContent_MakeAvailable(googletest)

enable_testing()

# fftshift1dVector
add_executable( test_fftshift1dVector
                test_fftshift1dVector.cu
              )
target_include_directories(test_fftshift1dVector PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_fftshift1dVector cuAlgo)
target_link_libraries(test_fftshift1dVector GTest::gtest_main)

# convolution1dMatrix
add_executable( test_convolution1dMatrix
                test_convolution1dMatrix.cu
              )
target_include_directories(test_convolution1dMatrix PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_convolution1dMatrix cuAlgo)
target_link_libraries(test_convolution1dMatrix GTest::gtest_main)

# convolutionReduction1dMatrix
add_executable( test_convolutionReduction1dMatrix
                test_convolutionReduction1dMatrix.cu
              )
target_include_directories(test_convolutionReduction1dMatrix PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_convolutionReduction1dMatrix cuAlgo)
target_link_libraries(test_convolutionReduction1dMatrix GTest::gtest_main)

# exclusiveScan1dVector
add_executable( test_exclusiveScan1dVector
                test_exclusiveScan1dVector.cu
              )
target_include_directories(test_exclusiveScan1dVector PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_exclusiveScan1dVector cuAlgo)
target_link_libraries(test_exclusiveScan1dVector GTest::gtest_main)

# gMatMul
add_executable( test_gMatMul
                test_gMatMul.cu
              )
target_include_directories(test_gMatMul PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_gMatMul cuAlgo)
target_link_libraries(test_gMatMul GTest::gtest_main)

# gMatVecMul
add_executable( test_gMatVecMul
                test_gMatVecMul.cu
              )
target_include_directories(test_gMatVecMul PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_gMatVecMul cuAlgo)
target_link_libraries(test_gMatVecMul GTest::gtest_main)

# gSpMatVecMulCSRAdaptive
add_executable( test_gSpMatVecMulCSRAdaptive
                test_gSpMatVecMulCSRAdaptive.cu
              )
target_include_directories(test_gSpMatVecMulCSRAdaptive PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_gSpMatVecMulCSRAdaptive cuAlgo)
target_link_libraries(test_gSpMatVecMulCSRAdaptive GTest::gtest_main)

# gSpMatVecMulCSRVector
add_executable( test_gSpMatVecMulCSRVector
                test_gSpMatVecMulCSRVector.cu
              )
target_include_directories(test_gSpMatVecMulCSRVector PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_gSpMatVecMulCSRVector cuAlgo)
target_link_libraries(test_gSpMatVecMulCSRVector GTest::gtest_main)

# gSpMatVecMulELL
add_executable( test_gSpMatVecMulELL
                test_gSpMatVecMulELL.cu
              )
target_include_directories(test_gSpMatVecMulELL PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_gSpMatVecMulELL cuAlgo)
target_link_libraries(test_gSpMatVecMulELL GTest::gtest_main)

# grad2dMatrix
add_executable( test_grad2dMatrix
                test_grad2dMatrix.cu
              )
target_include_directories(test_grad2dMatrix PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_grad2dMatrix cuAlgo)
target_link_libraries(test_grad2dMatrix GTest::gtest_main)

# normL1Vector
add_executable( test_normL1Vector
                test_normL1Vector.cu
              )
target_include_directories(test_normL1Vector PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_normL1Vector cuAlgo)
target_link_libraries(test_normL1Vector GTest::gtest_main)

# reduction1dMatrix
add_executable( test_reduction1dMatrix
                test_reduction1dMatrix.cu
              )
target_include_directories(test_reduction1dMatrix PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_reduction1dMatrix cuAlgo)
target_link_libraries(test_reduction1dMatrix GTest::gtest_main)

# reduction1dVector
add_executable( test_reduction1dVector
                test_reduction1dVector.cu
              )
target_include_directories(test_reduction1dVector PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_reduction1dVector cuAlgo)
target_link_libraries(test_reduction1dVector GTest::gtest_main)

# taper1dMatrix
add_executable( test_taper1dMatrix
                test_taper1dMatrix.cu
              )
target_include_directories(test_taper1dMatrix PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(test_taper1dMatrix cuAlgo)
target_link_libraries(test_taper1dMatrix GTest::gtest_main)

# transposeMatrix
add_executable( test_transposeMatrix
                test_transposeMatrix.cu
              )
target_include_directories(test_transposeMatrix PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_transposeMatrix cuAlgo)
target_link_libraries(test_transposeMatrix GTest::gtest_main)

# Add all tests to GoogleTest
include(GoogleTest)
gtest_discover_tests(test_convolution1dMatrix)
gtest_discover_tests(test_convolutionReduction1dMatrix)
gtest_discover_tests(test_exclusiveScan1dVector)
gtest_discover_tests(test_fftshift1dVector)
gtest_discover_tests(test_gMatMul)
gtest_discover_tests(test_gMatVecMul)
gtest_discover_tests(test_gSpMatVecMulCSRAdaptive)
gtest_discover_tests(test_gSpMatVecMulCSRVector)
gtest_discover_tests(test_gSpMatVecMulELL)
gtest_discover_tests(test_grad2dMatrix)
gtest_discover_tests(test_reduction1dMatrix)
gtest_discover_tests(test_reduction1dVector)
gtest_discover_tests(test_taper1dMatrix)
gtest_discover_tests(test_transposeMatrix)
