set(EXTENSION_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx_skl2onnx.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx_pruning.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx_pushdown.cpp
        ${EXTENSION_SOURCES}
        PARENT_SCOPE
)