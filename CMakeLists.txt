cmake_minimum_required(VERSION 3.13)

# Set vcpkg toolchain file
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
else()
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project(GPT2)

# Configure C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add vcpkg installation path
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/vcpkg/installed/x64-windows")

# UTF-8 support for MSVC
if(MSVC)
    add_compile_options(/utf-8)
endif()

# Project directory structure
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INCLUDE_DIR ${PROJECT_ROOT}/include)
set(SRC_DIR ${PROJECT_ROOT}/src)
set(UTILS_DIR ${PROJECT_ROOT}/utils)
set(LAYERS_DIR ${PROJECT_ROOT}/layers)
set(OPERATIONS_DIR ${PROJECT_ROOT}/operations)

# Find required packages
find_package(nlohmann_json CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)

# Create interface library for GPT2
add_library(gpt2_interface INTERFACE)
target_include_directories(gpt2_interface INTERFACE
    ${INCLUDE_DIR}
    ${UTILS_DIR}/include
    ${LAYERS_DIR}/Embedding/include
    ${LAYERS_DIR}/Normalization/include
    ${LAYERS_DIR}/Attention/include
    ${LAYERS_DIR}/MLP/include
    ${OPERATIONS_DIR}/include
)

target_link_libraries(gpt2_interface INTERFACE
    nlohmann_json::nlohmann_json
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS
)

# Utility libraries
add_library(gpt_tokenizer
    ${UTILS_DIR}/src/tokenizer.cpp
)

target_include_directories(gpt_tokenizer PUBLIC 
    ${UTILS_DIR}/include
)

target_link_libraries(gpt_tokenizer PUBLIC
    nlohmann_json::nlohmann_json
    gpt2_interface
)

# Parameter loader library
add_library(parameter_loader
    ${UTILS_DIR}/src/Loader.cpp
)

target_include_directories(parameter_loader PUBLIC
    ${UTILS_DIR}/include
)

target_link_libraries(parameter_loader PUBLIC
    gpt2_interface
)

# Embedding layer library
add_library(embedding_layer
    ${LAYERS_DIR}/Embedding/src/input_embedding.cpp
)

target_include_directories(embedding_layer PUBLIC
    ${LAYERS_DIR}/Embedding/include
)

target_link_libraries(embedding_layer PUBLIC
    gpt2_interface
)

# Normalization layer library
add_library(normalization_layer
    ${LAYERS_DIR}/Normalization/src/layer_normalization.cpp
)

target_include_directories(normalization_layer PUBLIC
    ${LAYERS_DIR}/Normalization/include
)

target_link_libraries(normalization_layer PUBLIC
    gpt2_interface
)

# Activations library
add_library(activations
    ${OPERATIONS_DIR}/src/activations.cpp
)

target_include_directories(activations PUBLIC
    ${OPERATIONS_DIR}/include
)

target_link_libraries(activations PUBLIC
    gpt2_interface
)

# Attention libraries
add_library(scaled_dot_attention
    ${LAYERS_DIR}/Attention/src/scaled_dot_attention.cpp
)

target_include_directories(scaled_dot_attention PUBLIC
    ${LAYERS_DIR}/Attention/include
)

target_link_libraries(scaled_dot_attention PUBLIC
    gpt2_interface
)

add_library(multi_head_attention
    ${LAYERS_DIR}/Attention/src/multihead_self_attention.cpp
)

target_include_directories(multi_head_attention PUBLIC
    ${LAYERS_DIR}/Attention/include
)

target_link_libraries(multi_head_attention PUBLIC
    gpt2_interface
    scaled_dot_attention
)

# MLP layer library
add_library(mlp_layer
    ${LAYERS_DIR}/MLP/src/mlp.cpp
)

target_include_directories(mlp_layer PUBLIC
    ${LAYERS_DIR}/MLP/include
)

target_link_libraries(mlp_layer PUBLIC
    gpt2_interface
)

# Main executable
add_executable(${PROJECT_NAME}
    ${SRC_DIR}/main.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    gpt2_interface
    gpt_tokenizer
    parameter_loader
    embedding_layer
    normalization_layer
    activations
    scaled_dot_attention
    multi_head_attention
    mlp_layer
)

# Print configuration summary
function(print_status_message)
    message(STATUS "Configuration Summary:")
    message(STATUS "Project: ${PROJECT_NAME}")
    message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
    message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
    message(STATUS "nlohmann_json found: ${nlohmann_json_FOUND}")
    message(STATUS "xtensor found: ${xtensor_FOUND}")
    message(STATUS "xtensor-blas found: ${xtensor-blas_FOUND}")
    message(STATUS "OpenBLAS found: ${OpenBLAS_FOUND}")
endfunction()

print_status_message()