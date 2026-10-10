cmake_minimum_required(VERSION 3.15)
project(llama_node)

# Dedupe transitive libraries during link. Both `llama` and `llama-common`
# link `ggml` PUBLIC, which produces an "ignoring duplicate libraries"
# warning from ld64 without this. CMake 3.27+; harmless on older CMake
# (the link line just keeps the duplicates and ld64 warns).
if(POLICY CMP0156)
    cmake_policy(SET CMP0156 NEW)
endif()

# All targets need PIC: vendor static libs get linked into the shared .node addon
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# Backend-DL mode (vendor mode only): build ggml backends (CPU variants,
# Vulkan, …) as runtime-loadable libraries placed next to the addon instead
# of linking one CPU backend statically. The addon then picks the best CPU
# variant at runtime and skips GPU backends whose system libraries are
# missing (e.g. no Vulkan loader) instead of failing to load entirely.
# This is how the published linux/win prebuilds are built. Accepted as a
# cmake option or, like LLAMA_ROOT, as an environment variable (cmake-js
# does not reliably forward -D vars to cmake configure).
option(LLAMA_NODE_BACKEND_DL "Build ggml backends as runtime-loadable libraries" OFF)
if(NOT LLAMA_NODE_BACKEND_DL AND DEFINED ENV{LLAMA_NODE_BACKEND_DL}
   AND NOT "$ENV{LLAMA_NODE_BACKEND_DL}" STREQUAL ""
   AND NOT "$ENV{LLAMA_NODE_BACKEND_DL}" STREQUAL "0")
    set(LLAMA_NODE_BACKEND_DL ON)
endif()

# Enable the ggml Vulkan backend (vendor mode only). Requires the Vulkan SDK
# (glslc + headers) at build time; at runtime only the system Vulkan loader
# is needed. Combine with LLAMA_NODE_BACKEND_DL so machines without the
# loader fall back to CPU instead of failing to load the addon.
option(LLAMA_NODE_VULKAN "Enable the ggml Vulkan backend" OFF)
if(NOT LLAMA_NODE_VULKAN AND DEFINED ENV{LLAMA_NODE_VULKAN}
   AND NOT "$ENV{LLAMA_NODE_VULKAN}" STREQUAL ""
   AND NOT "$ENV{LLAMA_NODE_VULKAN}" STREQUAL "0")
    set(LLAMA_NODE_VULKAN ON)
endif()

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
    set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
    if(LLAMA_NODE_BACKEND_DL)
        # GGML_BACKEND_DL requires shared libs: llama/ggml/ggml-base become
        # .so/.dll files shipped next to the addon, and each backend becomes
        # a MODULE library that ggml dlopens at runtime.
        set(BUILD_SHARED_LIBS ON  CACHE BOOL "" FORCE)
        set(GGML_BACKEND_DL   ON  CACHE BOOL "" FORCE)
        # One module per CPU microarchitecture (SSE4.2 … AVX512/AMX on x86,
        # armv8.x/armv9.x on ARM); ggml scores them at runtime and loads the
        # best one. Unsupported on Apple platforms (mac prebuilds stay
        # static anyway — Metal covers acceleration there).
        if(NOT APPLE)
            set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
        endif()
        # Everything is shipped flat next to llama_node.node: resolve the
        # shared-lib deps relative to each binary, not the build tree.
        if(APPLE)
            set(CMAKE_INSTALL_RPATH "@loader_path")
        else()
            set(CMAKE_INSTALL_RPATH "$ORIGIN")
        endif()
        set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    else()
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    endif()
    if(LLAMA_NODE_VULKAN)
        set(GGML_VULKAN ON CACHE BOOL "" FORCE)
    endif()
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
    # Disable host-native CPU flags: cmake-js may cross-compile (e.g. x86_64
    # host → arm64 target via CMAKE_OSX_ARCHITECTURES), so -mcpu=native is
    # invalid. Metal covers GPU acceleration; generic ARM code is fine.
    set(GGML_NATIVE          OFF CACHE BOOL "" FORCE)
    # Strip dead code across translation units
    set(GGML_LTO             ON  CACHE BOOL "" FORCE)
    add_subdirectory(vendor/llama.cpp)
endif()

# The addon target
add_library(llama_node SHARED
    src/addon.cpp
    src/llama_model.cpp
    src/generate_worker.cpp
    src/quantize_worker.cpp
    src/inspect_worker.cpp
    src/load_worker.cpp
    src/embed_worker.cpp
    src/state_worker.cpp
    src/chat_templates.cpp
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
    find_library(LLAMA_COMMON_LIBRARY
        NAMES llama-common common
        REQUIRED
        HINTS "${LLAMA_ROOT}/build/common" "${LLAMA_ROOT}/build/bin" "${LLAMA_ROOT}/build"
        NO_DEFAULT_PATH
    )
    message(STATUS "llama-node: using pre-built llama at ${LLAMA_LIBRARY}")
    message(STATUS "llama-node: using pre-built common at ${LLAMA_COMMON_LIBRARY}")
    target_include_directories(llama_node PRIVATE
        ${LLAMA_ROOT}/include
        ${LLAMA_ROOT}/ggml/include
        ${LLAMA_ROOT}/common
        ${LLAMA_ROOT}/vendor
    )
    target_link_libraries(llama_node PRIVATE
        ${LLAMA_COMMON_LIBRARY}
        ${LLAMA_LIBRARY}
        ${CMAKE_JS_LIB}
    )
else()
    # Vendor submodule or parent-scope target: llama target already defined,
    # its PUBLIC include dirs propagate transitively. `llama-common` brings
    # the Jinja chat templates + OAI-compat parsers used by
    # applyChatTemplateJinja.
    target_link_libraries(llama_node PRIVATE
        llama
        llama-common
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

# Backend-DL packaging: collect every vendor shared/MODULE library (llama,
# ggml, ggml-base, llama-common when shared, and all dlopen'd backends —
# whatever this llama.cpp version produces) and stage them next to
# llama_node.node so the directory is self-contained. The addon loads
# backends from its own directory at runtime (src/addon.cpp).
if(LLAMA_NODE_VENDOR AND LLAMA_NODE_BACKEND_DL)
    # Tells addon.cpp to load backends from the addon's directory instead of
    # relying on statically registered ones.
    target_compile_definitions(llama_node PRIVATE LLAMA_NODE_BACKEND_DL)

    function(llama_node_collect_runtime_targets out_var dir)
        set(result "")
        get_property(dir_targets DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(tgt IN LISTS dir_targets)
            get_target_property(tgt_type ${tgt} TYPE)
            if(tgt_type STREQUAL "SHARED_LIBRARY" OR tgt_type STREQUAL "MODULE_LIBRARY")
                list(APPEND result ${tgt})
            endif()
        endforeach()
        get_property(subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
        foreach(sub IN LISTS subdirs)
            llama_node_collect_runtime_targets(sub_result "${sub}")
            list(APPEND result ${sub_result})
        endforeach()
        set(${out_var} ${result} PARENT_SCOPE)
    endfunction()

    llama_node_collect_runtime_targets(LLAMA_NODE_RUNTIME_TARGETS
        "${CMAKE_SOURCE_DIR}/vendor/llama.cpp")

    foreach(tgt IN LISTS LLAMA_NODE_RUNTIME_TARGETS)
        # Strip VERSION/SOVERSION: npm packages cannot contain symlinks, so
        # every library must be a single real file with its plain name
        # (libggml-vulkan.so, not libggml-vulkan.so.0.9.5 + symlinks) — the
        # ggml backend scan also only matches the plain ".so" extension.
        set_property(TARGET ${tgt} PROPERTY VERSION)
        set_property(TARGET ${tgt} PROPERTY SOVERSION)
        add_custom_command(TARGET llama_node POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:${tgt}>" "$<TARGET_FILE_DIR:llama_node>")
    endforeach()
    list(LENGTH LLAMA_NODE_RUNTIME_TARGETS _n)
    message(STATUS "llama-node: backend-DL mode, staging ${_n} runtime libs next to the addon: ${LLAMA_NODE_RUNTIME_TARGETS}")
endif()
