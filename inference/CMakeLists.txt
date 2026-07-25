cmake_minimum_required(VERSION 3.14)

# Use clang/clang++ as the default compiler
if (NOT CMAKE_C_COMPILER)
    set(CMAKE_C_COMPILER "clang")
endif()
if (NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "clang++")
endif()

project("llama-nano" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message("CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Config options
option(LLAMA_NATIVE "llama-nano: optimize for current CPU" ON)
option(LLAMA_OPENMP "llama-nano: use OpenMP" ON)
option(LLAMA_BUILD_MAIN "llama-nano: build inference binary" ON)

# For ggml subdirectory build
set(GGML_STANDALONE OFF)
set(GGML_NATIVE ${LLAMA_NATIVE})

# Disable all non-CPU backends
set(GGML_CUDA OFF)
set(GGML_HIP OFF)
set(GGML_METAL OFF)
set(GGML_VULKAN OFF)
set(GGML_SYCL OFF)
set(GGML_RPC OFF)
set(GGML_OPENCL OFF)
set(GGML_CANN OFF)
set(GGML_WEBGPU OFF)
set(GGML_ET OFF)
set(GGML_HEXAGON OFF)
set(GGML_ZDNN OFF)
set(GGML_ZENDNN OFF)
set(GGML_OPENVINO OFF)
set(GGML_VIRTGPU OFF)
set(GGML_MUSA OFF)
set(GGML_BACKEND_DL OFF)
set(GGML_CPU_ALL_VARIANTS OFF)
set(GGML_BLAS OFF)
set(GGML_CPU_KLEIDIAI OFF)
set(GGML_CPU_HBM OFF)
set(GGML_LLAMAFILE OFF)

# Disable tests/examples in ggml
set(GGML_BUILD_TESTS OFF)
set(GGML_BUILD_EXAMPLES OFF)

# OpenMP
if (LLAMA_OPENMP)
    set(GGML_OPENMP ON)
else()
    set(GGML_OPENMP OFF)
endif()

# Add path to modules (needed by ggml)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# Build ggml library
add_subdirectory(ggml)

# Build llama library
add_library(llama
    src/llama.cpp
    src/llama-adapter.cpp
    src/llama-arch.cpp
    src/llama-batch.cpp
    src/llama-chat.cpp
    src/llama-context.cpp
    src/llama-cparams.cpp
    src/llama-grammar.cpp
    src/llama-graph.cpp
    src/llama-hparams.cpp
    src/llama-impl.cpp
    src/llama-io.cpp
    src/llama-kv-cache.cpp
    src/llama-kv-cache-iswa.cpp
    src/llama-kv-cache-dsa.cpp
    src/llama-kv-cache-dsv4.cpp
    src/llama-memory.cpp
    src/llama-memory-hybrid.cpp
    src/llama-memory-hybrid-iswa.cpp
    src/llama-memory-recurrent.cpp
    src/llama-mmap.cpp
    src/llama-model-loader.cpp
    src/llama-model-saver.cpp
    src/llama-model.cpp
    src/llama-quant.cpp
    src/llama-sampler.cpp
    src/llama-vocab.cpp
    src/unicode-data.cpp
    src/unicode.cpp
)

# Add all model architectures
file(GLOB LLAMA_MODELS_SOURCES "src/models/*.cpp")
target_sources(llama PRIVATE ${LLAMA_MODELS_SOURCES})

target_include_directories(llama PRIVATE src)
target_include_directories(llama PUBLIC include)
target_compile_features(llama PRIVATE cxx_std_17)
target_link_libraries(llama PUBLIC ggml)

# Build inference binary
if (LLAMA_BUILD_MAIN)
    add_executable(llama-nano main.cpp)
    target_link_libraries(llama-nano PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
    target_compile_features(llama-nano PRIVATE cxx_std_17)
    install(TARGETS llama-nano RUNTIME)
endif()

# Build quantization tool
add_executable(llama-quantize quantize.cpp)
target_link_libraries(llama-quantize PRIVATE llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-quantize PRIVATE cxx_std_17)
install(TARGETS llama-quantize RUNTIME)

install(TARGETS llama LIBRARY)
install(FILES include/llama.h DESTINATION include)
