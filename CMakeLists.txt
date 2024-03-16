cmake_minimum_required(VERSION 3.10)
project(MySynapseProject)

option(USE_SIMD "Enable SIMD optimizations" ON)

if(USE_SIMD)
    # Add compiler flags to enable SIMD optimizations
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        add_compile_options("-mavx2")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        add_compile_options("/arch:AVX2")
#        add_compile_options("/fp:fast")
    endif()
endif()
add_subdirectory(synapse)
add_subdirectory(sandbox)
