cmake_minimum_required(VERSION 3.12)
project(deberta_cpp LANGUAGES C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# standard Required
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD_REQUIRED true)

# Threads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# options
option(GGML_ALL_WARNINGS "Enable all compiler warnings" OFF)
option(GGML_PERF "Enable performance logging" OFF)
option(GGML_AVX "Enable AVX" ON)
option(GGML_AVX2 "Enable AVX2" ON)
option(GGML_FMA "Enable FMA" ON)
option(GGML_CUBLAS "Use cuBlas" OFF)
option(GGML_METAL "Use Metal" OFF)

# Compile flags
add_compile_options(-Wno-format)

if (GGML_ALL_WARNINGS)
  add_compile_options(-Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function)
endif()

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

function(check_processor_options)
    message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
    if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
        message(STATUS "x86 detected")
        add_compile_options(-mf16c)
        if (GGML_FMA)
            add_compile_options(-mfma)
        endif()
        if (GGML_AVX)
            add_compile_options(-mavx)
        endif()
        if (GGML_AVX2)
            add_compile_options(-mavx2)
        endif()
    else()
        message(STATUS "Unknown architecture")
    endif()
endfunction()

# processor specific settings
check_processor_options()

# backend specific defenitions
if (GGML_CUBLAS)
  add_compile_definitions(GGML_USE_CUBLAS)
endif()

# subdirectories
add_subdirectory(ggml EXCLUDE_FROM_ALL)
set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
install(TARGETS ggml LIBRARY DESTINATION dedberta_cpp)

add_subdirectory(src)
