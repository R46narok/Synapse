add_library(synapse STATIC
        src/tensor.cpp
        include/synapse/tensor.h
        include/synapse/activation.h
        include/synapse/graph.h
)

target_include_directories(synapse PUBLIC
        include
)

target_compile_definitions(synapse PRIVATE SYNAPSE_CORE)
