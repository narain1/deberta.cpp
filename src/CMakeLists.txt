include_directories(${CMAKE_SOURCE_DIR}/ggml/src ${CMAKE_SOURCE_DIR}/ggml/include/ggml)

add_library(deberta deberta.cpp deberta.h)
target_compile_features(deberta PUBLIC cxx_std_20)
set_target_properties(deberta PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  INSTALL_RPATH "$ORIGIN")
target_link_libraries(deberta PRIVATE ggml)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE deberta ggml)

# add_executable(quantize quantize.cpp)
# target_link_libraries(quantize PRIVATE deberta ggml)

install(TARGETS deberta LIBRARY DESTINATION deberta_cpp)
