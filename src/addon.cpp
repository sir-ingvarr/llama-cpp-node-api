#include <napi.h>
#include <chrono>
#include "llama.h"
#include "ggml-backend.h"
#include "addon_state.h"
#include "llama_model.h"
#include "quantize_worker.h"
#include "inspect_worker.h"
#include "load_worker.h"

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

    // Initialize all compiled backends (CPU, Metal, CUDA, …)
    ggml_backend_load_all();
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
    return exports;
}

NODE_API_MODULE(llama_node, InitModule)
