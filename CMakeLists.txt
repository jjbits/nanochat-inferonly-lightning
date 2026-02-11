cmake_minimum_required(VERSION 3.18)
project(nanochat LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

# Corrosion for Rust integration
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5
)
FetchContent_MakeAvailable(Corrosion)

# CUTLASS headers only (skip library build)
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v3.4.1
)
FetchContent_GetProperties(cutlass)
if(NOT cutlass_POPULATED)
    FetchContent_Populate(cutlass)
endif()

# Header-only HTTP server
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3
)
FetchContent_MakeAvailable(httplib)

# Header-only JSON
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Import rustbpe from submodule
corrosion_import_crate(MANIFEST_PATH extern/rustbpe/Cargo.toml)

# ============================================================
# Tokenizer conversion: pkl -> json
# Only re-runs when tokenizer.pkl changes (no C++ recompile)
# ============================================================
set(TOKENIZER_PKL "${CMAKE_SOURCE_DIR}/assets/tokenizer.pkl")
set(TOKENIZER_JSON "${CMAKE_SOURCE_DIR}/assets/tokenizer.json")

add_custom_command(
    OUTPUT ${TOKENIZER_JSON}
    COMMAND ${CMAKE_SOURCE_DIR}/.venv/bin/python
            ${CMAKE_SOURCE_DIR}/utils/export_tokenizer.py
            ${TOKENIZER_PKL} ${TOKENIZER_JSON}
    DEPENDS ${TOKENIZER_PKL} ${CMAKE_SOURCE_DIR}/utils/export_tokenizer.py
    COMMENT "Converting tokenizer.pkl -> tokenizer.json"
    VERBATIM
)

# Standalone target: cmake --build build --target tokenizer
add_custom_target(tokenizer ALL DEPENDS ${TOKENIZER_JSON})

# Kernel library
add_library(kernels STATIC
    kernels/rmsnorm.cu
    kernels/rope.cu
    kernels/flash_attention.cu
    kernels/ops.cu
    kernels/gemm.cu
)
target_include_directories(kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(kernels SYSTEM PUBLIC ${cutlass_SOURCE_DIR}/include)
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>)

# Main executable
add_executable(nanochat
    src/model.cpp
    src/engine.cpp
    src/main.cpp
)
target_include_directories(nanochat PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/rustbpe/include
)
target_include_directories(nanochat SYSTEM PRIVATE ${cutlass_SOURCE_DIR}/include)
target_link_libraries(nanochat PRIVATE kernels CUDA::cudart rustbpe)

# Server executable
add_executable(nanochat-server
    utils/server/server.cpp
    src/model.cpp
    src/engine.cpp
)
target_include_directories(nanochat-server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/extern/rustbpe/include
)
target_include_directories(nanochat-server SYSTEM PRIVATE ${cutlass_SOURCE_DIR}/include)
target_link_libraries(nanochat-server PRIVATE
    kernels
    CUDA::cudart
    rustbpe
    httplib::httplib
    nlohmann_json::nlohmann_json
)
