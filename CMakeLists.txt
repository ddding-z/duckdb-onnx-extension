cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME duckonnx)
set(CMAKE_CXX_STANDARD 17)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})
include_directories(src/include)
add_subdirectory(src)

add_definitions(-DONNX_ML=1)
add_definitions(-DONNX_NAMESPACE=onnx)

include_directories(third_party/onnx_optimizer/onnx_optimizer)
include_directories(third_party/onnx_optimizer/onnx_optimizer/third_party/onnx)
include_directories(third_party/onnxruntime/include)

# set(EXTENSION_SOURCES src/duckonnx_extension.cpp)

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

find_library(ONNXRUNTIME_LIB onnxruntime REQUIRED PATHS third_party/onnxruntime/lib)
message(STATUS "ONNXRUNTIME_LIB found: ${ONNXRUNTIME_LIB}")
target_link_libraries(${EXTENSION_NAME} ${ONNXRUNTIME_LIB})
target_link_libraries(${LOADABLE_EXTENSION_NAME} ${ONNXRUNTIME_LIB})

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
