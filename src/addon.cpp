#include <napi.h>
#include "llama.h"
#include "ggml-backend.h"
#include "llama_model.h"

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

    // Register a cleanup hook so llama_backend_free() is called when the
    // Node.js environment tears down (e.g. process exit or worker thread end).
    env.AddCleanupHook([]() { llama_backend_free(); });

    LlamaModel::Init(env, exports);
    return exports;
}

NODE_API_MODULE(llama_node, InitModule)
