# CMakeLists configuration file for the 'src' directory.
# =================================================================
# Project Initialization
# =================================================================
cmake_minimum_required(VERSION 3.13)

# Set vcpkg toolchain file - IMPORTANT: This must come before project()
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
else()
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
endif()

project(GPT2)  # TODO: Replace with your project name

# Configure C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add vcpkg installation path to help find packages
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/vcpkg/installed/x64-windows")

# UTF-8 support for MSVC
if(MSVC)
    add_compile_options(/utf-8)
endif()

# =================================================================
# Project Structure and Directory Setup
# =================================================================
# TODO: Customize these paths according to your project structure
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(UTILS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/utils")
set(LAYERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/layers")
set(OPERATIONS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/operations")
# =================================================================
# Find Required Packages
# =================================================================
find_package(nlohmann_json CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)

# TODO: Add other required packages here
# Example:
# find_package(Boost REQUIRED)
# find_package(OpenCV REQUIRED)
# find_package(Qt6 COMPONENTS Core Gui Widgets REQUIRED)

# =================================================================
# Common Include Directories
# =================================================================
set(COMMON_INCLUDE_DIRS
    ${INCLUDE_DIR}
    ${LIB_DIR}
    ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include
    ${UTILS_DIR}/include
    ${LAYERS_DIR}/Embedding/include
    ${LAYERS_DIR}/Normalization/include
    ${LAYERS_DIR}/Attention/include
    ${OPERATIONS_DIR}/include
    # TODO: Add other include directories as needed
)

# =================================================================
# Utility Libraries
# =================================================================
# GPTTokenizer library
add_library(gpt_tokenizer
    ${UTILS_DIR}/src/tokenizer.cpp
)

target_include_directories(gpt_tokenizer PUBLIC 
    ${COMMON_INCLUDE_DIRS}
    ${UTILS_DIR}/include  # Ensure tokenizer header files are found
)

target_link_libraries(gpt_tokenizer PUBLIC
    nlohmann_json::nlohmann_json
    # Add any other libraries that GPTTokenizer depends on
)

# Parameter loader library
add_library(parameter_loader
    ${UTILS_DIR}/src/Loader.cpp
)

target_include_directories(parameter_loader PUBLIC
    ${COMMON_INCLUDE_DIRS}
    ${UTILS_DIR}/include  # Ensure loader header files are found
)

target_link_libraries(parameter_loader PUBLIC
    nlohmann_json::nlohmann_json
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS

    # Add any other libraries that Loader depends on
)

# Embedding layer library
add_library(embedding_layer
    ${LAYERS_DIR}/Embedding/src/input_embedding.cpp
)

target_include_directories(embedding_layer PUBLIC
    ${COMMON_INCLUDE_DIRS}
    ${LAYERS_DIR}/Embedding/include  # Ensure embedding header files are found
)

target_link_libraries(embedding_layer PUBLIC
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS
    # Add any other libraries that EmbeddingLayer depends on
)


# Normalization layer library
add_library(normalization_layer
    ${LAYERS_DIR}/Normalization/src/layer_normalization.cpp
)

target_include_directories(normalization_layer PUBLIC
    ${COMMON_INCLUDE_DIRS}
    ${LAYERS_DIR}/Normalization/include  # Ensure normalization header files are found
)

target_link_libraries(normalization_layer PUBLIC
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS
    # Add any other libraries that NormalizationLayer depends on
)

# Activations library
add_library(activations
    ${OPERATIONS_DIR}/src/activations.cpp
)

target_include_directories(activations PUBLIC
    ${COMMON_INCLUDE_DIRS}
    ${OPERATIONS_DIR}/include  # Ensure activation header files are found
)

target_link_libraries(activations PUBLIC
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS
    # Add any other libraries that Activations depends on
)


# Scaled Dot Product Attention library
add_library(scaled_dot_attention
    ${LAYERS_DIR}/Attention/src/scaled_dot_attention.cpp
)

target_include_directories(scaled_dot_attention PUBLIC
    ${COMMON_INCLUDE_DIRS}
    ${LAYERS_DIR}/Attention/include  # Ensure attention header files are found

)

target_link_libraries(scaled_dot_attention PUBLIC
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS
    # Add any other libraries that ScaledDotAttention depends on
)

# Multi-Head Attention library
add_library(multi_head_attention
    ${LAYERS_DIR}/Attention/src/multihead_self_attention.cpp
)

target_include_directories(multi_head_attention PUBLIC
    ${COMMON_INCLUDE_DIRS}
    ${LAYERS_DIR}/Attention/include  # Ensure attention header files are found

)

target_link_libraries(multi_head_attention PUBLIC
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS
    # Add any other libraries that MultiHeadAttention depends on
)


# MLP layer library
add_library(mlp_layer
    ${LAYERS_DIR}/MLP/src/mlp.cpp
)

target_include_directories(mlp_layer PUBLIC
    ${COMMON_INCLUDE_DIRS}
    ${LAYERS_DIR}/MLP/include  # Ensure MLP header files are found
)

target_link_libraries(mlp_layer PUBLIC
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS
    # Add any other libraries that MLPLayer depends on
)


# =================================================================
# Library Targets
# =================================================================
# Function to add library targets with common configuration
function(add_project_library name sources)
    add_library(${name} ${sources})
    target_include_directories(${name} PUBLIC ${COMMON_INCLUDE_DIRS})
    target_link_libraries(${name} PUBLIC
        xtensor
        xtensor-blas
        OpenBLAS::OpenBLAS
        nlohmann_json::nlohmann_json
        # TODO: Add other common dependencies here
    )
endfunction()

# TODO: Add your library targets here
# Example:
# add_project_library(my_library_name
#     "${LIB_DIR}/src/my_library.cpp"
#     "${LIB_DIR}/src/another_file.cpp"
# )

# =================================================================
# Executable Targets
# =================================================================
# Main executable
add_executable(${PROJECT_NAME}
    "${SRC_DIR}/GPT2.cpp"
    # TODO: Add other source files here
)

target_include_directories(${PROJECT_NAME} PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE
    gpt_tokenizer
    embedding_layer
    normalization_layer
    activations
    scaled_dot_attention
    multi_head_attention
    mlp_layer
    parameter_loader
    xtensor
    xtensor-blas
    OpenBLAS::OpenBLAS
    nlohmann_json::nlohmann_json
    # TODO: Add your library targets and other dependencies here
)

# =================================================================
# Test Configuration
# =================================================================
# TODO: Uncomment and modify if you want to add tests
# add_executable(tests
#     "${TEST_DIR}/main_test.cpp"
#     "${TEST_DIR}/test1.cpp"
#     "${TEST_DIR}/test2.cpp"
# )
# target_include_directories(tests PRIVATE ${COMMON_INCLUDE_DIRS})
# target_link_libraries(tests PRIVATE
#     # Add your library targets and test dependencies here
# )

# =================================================================
# Installation Configuration
# =================================================================
# TODO: Uncomment and modify for installation targets
# install(TARGETS ${PROJECT_NAME}
#     RUNTIME DESTINATION bin
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
# )

# =================================================================
# Configuration Summary
# =================================================================
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
    # TODO: Add status messages for other dependencies
endfunction()

print_status_message()