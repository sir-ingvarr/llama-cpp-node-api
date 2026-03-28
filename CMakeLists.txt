cmake_minimum_required(VERSION 3.15)
project(llama_node)

# cmake-js variables
include_directories(${CMAKE_JS_INC})

# On Windows, generate node.lib from the .def file provided by cmake-js
if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF}
                    /out:${CMAKE_JS_NODELIB_TARGET}
                    ${CMAKE_STATIC_LINKER_FLAGS})
endif()

# Resolve node-addon-api include path
execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
# Strip surrounding quotes that node -p may include
string(REGEX REPLACE "^\"(.*)\"$" "\\1" NODE_ADDON_API_DIR "${NODE_ADDON_API_DIR}")
include_directories(${NODE_ADDON_API_DIR})

# llama.cpp integration — three mutually exclusive modes:
#
#   LLAMA_ROOT=<path>      Point to a pre-built llama.cpp source tree.
#                          Headers are taken from <path>/include and
#                          <path>/ggml/include; libllama is searched in
#                          <path>/build/bin and <path>/build.
#                          Use this when llama.cpp is already built next
#                          to this package (e.g. sibling directory).
#
#   LLAMA_NODE_VENDOR=ON   Build the bundled vendor/llama.cpp submodule
#                          as part of this project (static, no extras).
#                          Use this for a fully self-contained build.
#
#   (neither)              Expect a parent CMake project to have already
#                          defined the `llama` target before add_subdirectory.
#
set(LLAMA_ROOT "" CACHE PATH "Path to a pre-built llama.cpp source tree")
# Also accept LLAMA_ROOT as an environment variable (useful with cmake-js,
# which does not reliably forward --CD flags to cmake configure).
if(NOT LLAMA_ROOT AND DEFINED ENV{LLAMA_ROOT})
    set(LLAMA_ROOT "$ENV{LLAMA_ROOT}")
endif()
option(LLAMA_NODE_VENDOR "Use vendor/llama.cpp submodule" OFF)

# Auto-enable vendor mode when the submodule is present and no other source
# is specified. This means `npm run build` works out of the box after
# `git submodule update --init --recursive` with no extra flags.
if(NOT LLAMA_ROOT AND NOT LLAMA_NODE_VENDOR)
    if(EXISTS "${CMAKE_SOURCE_DIR}/vendor/llama.cpp/CMakeLists.txt")
        set(LLAMA_NODE_VENDOR ON)
        message(STATUS "llama-node: auto-detected vendor/llama.cpp submodule")
    endif()
endif()

if(LLAMA_NODE_VENDOR)
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    # Disable host-native CPU flags: cmake-js may cross-compile (e.g. x86_64
    # host → arm64 target via CMAKE_OSX_ARCHITECTURES), so -mcpu=native is
    # invalid. Metal covers GPU acceleration; generic ARM code is fine.
    set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)
    add_subdirectory(vendor/llama.cpp)
endif()

# The addon target
add_library(llama_node SHARED
    src/addon.cpp
    src/llama_model.cpp
    src/generate_worker.cpp
)

set_target_properties(llama_node PROPERTIES
    PREFIX ""
    SUFFIX ".node"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_compile_definitions(llama_node PRIVATE
    NAPI_VERSION=8
    NAPI_DISABLE_CPP_EXCEPTIONS
)

if(LLAMA_ROOT)
    # Pre-built llama.cpp tree: wire up headers and find the library directly.
    find_library(LLAMA_LIBRARY llama
        REQUIRED
        HINTS "${LLAMA_ROOT}/build/bin" "${LLAMA_ROOT}/build"
        NO_DEFAULT_PATH
    )
    message(STATUS "llama-node: using pre-built llama at ${LLAMA_LIBRARY}")
    target_include_directories(llama_node PRIVATE
        ${LLAMA_ROOT}/include
        ${LLAMA_ROOT}/ggml/include
    )
    target_link_libraries(llama_node PRIVATE
        ${LLAMA_LIBRARY}
        ${CMAKE_JS_LIB}
    )
else()
    # Vendor submodule or parent-scope target: llama target already defined,
    # its PUBLIC include dirs propagate transitively.
    target_link_libraries(llama_node PRIVATE
        llama
        ${CMAKE_JS_LIB}
    )
endif()

# macOS: link system frameworks needed by Metal backend
if(APPLE)
    target_link_libraries(llama_node PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalKit"
    )
endif()
