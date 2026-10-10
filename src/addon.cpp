#include <napi.h>
#include <chrono>
#include "llama.h"
#include "ggml-backend.h"
#include "addon_state.h"
#include "llama_model.h"
#include "quantize_worker.h"
#include "inspect_worker.h"
#include "load_worker.h"

#ifdef LLAMA_NODE_BACKEND_DL
#include <string>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// Directory containing this addon (.node). In backend-DL builds the ggml
// backends (ggml-cpu-*, ggml-vulkan, …) are shipped next to the addon, not
// next to the node/electron executable, so ggml's default search path
// (executable dir + cwd) would never find them.
static std::string AddonDirectory() {
#ifdef _WIN32
    HMODULE mod = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&AddonDirectory), &mod)) {
        return {};
    }
    std::wstring wpath(32768, L'\0');
    DWORD n = GetModuleFileNameW(mod, &wpath[0], static_cast<DWORD>(wpath.size()));
    if (n == 0 || n >= wpath.size()) {
        return {};
    }
    wpath.resize(n);
    const size_t slash = wpath.find_last_of(L"\\/");
    if (slash == std::wstring::npos) {
        return {};
    }
    wpath.resize(slash);
    // ggml takes UTF-8 paths (fs::u8path internally)
    const int len = WideCharToMultiByte(CP_UTF8, 0, wpath.data(),
                                        static_cast<int>(wpath.size()),
                                        nullptr, 0, nullptr, nullptr);
    if (len <= 0) {
        return {};
    }
    std::string path(static_cast<size_t>(len), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wpath.data(), static_cast<int>(wpath.size()),
                        &path[0], len, nullptr, nullptr);
    return path;
#else
    Dl_info info;
    if (dladdr(reinterpret_cast<void *>(&AddonDirectory), &info) == 0 ||
        info.dli_fname == nullptr) {
        return {};
    }
    const std::string path(info.dli_fname);
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) {
        return {};
    }
    return path.substr(0, slash);
#endif
}
#endif // LLAMA_NODE_BACKEND_DL

static Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
    // Suppress info/warn log levels; only pass through errors
    llama_log_set(
        [](enum ggml_log_level level, const char * text, void *) {
            if (level >= GGML_LOG_LEVEL_ERROR) {
                fprintf(stderr, "%s", text);
            }
        },
        nullptr
    );

#ifdef LLAMA_NODE_BACKEND_DL
    // Backends are separate dynamic libraries shipped next to the addon:
    // ggml scores the CPU variants and loads the best one, and skips GPU
    // backends whose system libraries are missing (e.g. no Vulkan loader)
    // instead of failing. GGML_BACKEND_PATH is still honoured on top for
    // out-of-tree backends. Empty dir (path lookup failed) falls back to
    // ggml's default search (executable dir + cwd).
    const std::string addon_dir = AddonDirectory();
    ggml_backend_load_all_from_path(addon_dir.empty() ? nullptr : addon_dir.c_str());
#else
    // Initialize all statically compiled backends (CPU, Metal, CUDA, …)
    ggml_backend_load_all();
#endif
    llama_backend_init();

    // Per-env state for worker lifecycle tracking. Owned by the env (auto-freed
    // via SetInstanceData); workers fetch it via env.GetInstanceData<AddonState>().
    auto * state = new AddonState();
    env.SetInstanceData(state);

    // On env teardown, signal in-flight workers to exit promptly, wait briefly
    // for them to drain off the libuv pool, then free the llama backend. Without
    // this, llama_backend_free() can race a worker mid-llama_decode (UAF).
    napi_add_env_cleanup_hook(env,
        [](void * data) {
            auto * s = static_cast<AddonState *>(data);
            s->shutting_down.store(true, std::memory_order_release);
            // Workers in their main loops poll this flag (via abort_callback for
            // llama_decode); give them up to 5s to unwind before freeing.
            s->wait_for_drain(std::chrono::milliseconds(5000));
            llama_backend_free();
        },
        state);

    LlamaModel::Init(env, exports);
    RegisterQuantize(env, exports);
    RegisterInspect(env, exports);
    RegisterLoadModel(env, exports);

    // llamaVersion() — upstream llama.cpp version string, for diagnostics.
    exports.Set("llamaVersion", Napi::Function::New(env,
        [](const Napi::CallbackInfo & info) -> Napi::Value {
            return Napi::String::New(info.Env(), llama_version());
        }, "llamaVersion"));

    return exports;
}

NODE_API_MODULE(llama_node, InitModule)
