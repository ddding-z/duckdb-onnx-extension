add_subdirectory(functions)
add_subdirectory(optimizer)
add_subdirectory(utils)

set(EXTENSION_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/module.cpp
        ${EXTENSION_SOURCES}
        PARENT_SCOPE
    )