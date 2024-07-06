add_executable(reductionVector reductionVector.cu)
target_link_libraries (reductionVector cuAlgo)
target_include_directories(reductionVector PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_property(TARGET reductionVector PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install (TARGETS
  reductionVector # executables
  RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/examples)

