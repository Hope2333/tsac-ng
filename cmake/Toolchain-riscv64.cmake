# Toolchain file for RISC-V 64-bit cross-compilation
# Usage: cmake -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-riscv64.cmake ..

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Cross-compiler prefix (adjust to your toolchain)
set(TOOLCHAIN_PREFIX riscv64-linux-gnu)
if(NOT DEFINED CMAKE_C_COMPILER)
    find_program(CMAKE_C_COMPILER NAMES ${TOOLCHAIN_PREFIX}-gcc)
endif()
if(NOT DEFINED CMAKE_CXX_COMPILER)
    find_program(CMAKE_CXX_COMPILER NAMES ${TOOLCHAIN_PREFIX}-g++)
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# RISC-V baseline: RV64GC (IMAFDC)
# For RVV (Vector Extension 1.0): add -march=rv64gcv
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=rv64gc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=rv64gc")

# Disable GPU backends (cross-compile CPU-only)
set(USE_CUDA OFF CACHE BOOL "")
set(USE_HIP OFF CACHE BOOL "")
set(USE_VULKAN OFF CACHE BOOL "")
set(USE_LLVM OFF CACHE BOOL "")
