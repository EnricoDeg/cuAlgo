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

include(GoogleTest)
gtest_discover_tests(test_gMatMul)
