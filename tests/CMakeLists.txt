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

# gMatMul
add_executable(
  test_gMatMul
  test_gMatMul.cu
)
target_include_directories(test_gMatMul PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_gMatMul cuAlgo)
target_link_libraries(test_gMatMul GTest::gtest_main)

# transposeMatrix
add_executable(
  test_transposeMatrix
  test_transposeMatrix.cu
)
target_include_directories(test_transposeMatrix PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_transposeMatrix cuAlgo)
target_link_libraries(test_transposeMatrix GTest::gtest_main)

# fftshiftVector
add_executable(
  test_fftshiftVector
  test_fftshiftVector.cu
)
target_include_directories(test_fftshiftVector PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries (test_fftshiftVector cuAlgo)
target_link_libraries(test_fftshiftVector GTest::gtest_main)

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

include(GoogleTest)
gtest_discover_tests(test_gMatMul)
gtest_discover_tests(test_transposeMatrix)
gtest_discover_tests(test_fftshiftVector)
gtest_discover_tests(test_convolution1dMatrix)
gtest_discover_tests(test_convolutionReduction1dMatrix)
