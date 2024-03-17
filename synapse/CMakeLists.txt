add_library(synapse STATIC
        src/tensor.cpp
        include/synapse/tensor.h
        include/synapse/activation.h
        include/synapse/graph.h
        src/activation.cpp
        include/synapse/mem.h
        include/synapse/layers/dense.h
        src/layers/dense.cpp
        include/synapse/layers/base_layer.h
        src/layers/base_layer.cpp
        src/graph.cpp
        include/synapse/layers/softmax.h
        src/layers/softmax.cpp
)

target_include_directories(synapse PUBLIC
        include
)

target_compile_definitions(synapse PRIVATE SYNAPSE_CORE)
