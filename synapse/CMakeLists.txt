add_library(synapse STATIC
        src/tensor.cpp
)

target_include_directories(synapse PUBLIC
        include
)

target_compile_definitions(synapse PRIVATE SYNAPSE_CORE)
