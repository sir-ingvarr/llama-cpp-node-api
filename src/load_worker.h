#pragma once

#include <napi.h>
#include <cstdint>
#include <string>

#include "llama.h"

struct AddonState;

// Maps the JS-facing useMmap/useMlock booleans onto llama_load_mode (which
// replaced the use_mmap/use_mlock bools in llama_model_params as of llama.cpp
// v0.6.0). When neither option is supplied, AUTO lets llama.cpp pick based on
// device capabilities. When either is supplied, the unspecified one keeps its
// historical default (mmap=true, mlock=false) so existing callers see the same
// behaviour as before the migration.
inline llama_load_mode ResolveLoadMode(bool has_opt, bool use_mmap, bool use_mlock) {
    if (!has_opt)               return LLAMA_LOAD_MODE_AUTO;
    if (use_mmap && use_mlock)  return LLAMA_LOAD_MODE_MMAP_MLOCK;
    if (use_mmap)               return LLAMA_LOAD_MODE_MMAP;
    if (use_mlock)              return LLAMA_LOAD_MODE_MLOCK;
    return LLAMA_LOAD_MODE_NONE;
}

// Owns a loaded llama_model* between the worker thread that produces it and
// the JS-thread constructor that consumes it. Wrapped in Napi::External so
// the GC can free the model if the JS side never picks it up.
struct LoadHandle {
    llama_model * model        = nullptr;
    uint32_t      n_ctx        = 2048;
    int32_t       n_gpu_layers = 99;
    bool          embeddings   = false;
    // -1 = LLAMA_POOLING_TYPE_UNSPECIFIED (let the model decide).
    int32_t       pooling_type = -1;
    // -1 = leave llama_context_default_params() default (currently F16).
    int32_t       cache_type_k = -1;
    int32_t       cache_type_v = -1;
    // -1 = LLAMA_FLASH_ATTN_TYPE_AUTO (let llama.cpp decide).
    int32_t       flash_attn_type = -1;
    // -1 = leave llama_context_default_params() default.
    int32_t       n_threads       = -1;
    int32_t       n_threads_batch = -1;
    // Model-load mode (applied before llama_model_load_from_file), already
    // resolved from the useMmap/useMlock JS options via ResolveLoadMode.
    llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO;
    // Per-env state, used only by the External finalizer to detect env teardown
    // (so it can skip llama_model_free and avoid racing llama_backend_free).
    // Non-owning; AddonState outlives the handle.
    AddonState *  state      = nullptr;
};

// LoadWorker runs llama_model_load_from_file on a libuv worker thread so the
// JS event loop doesn't block for seconds while the model is mmap'd / GPU-
// uploaded. On success, hands the loaded model to JS via an External<LoadHandle>;
// `LlamaModel`'s constructor takes ownership when invoked with that handle.
class LoadWorker : public Napi::AsyncWorker {
public:
    LoadWorker(Napi::Function & done_cb,
               std::string      path,
               LoadHandle *     handle);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error & error) override;

private:
    Napi::FunctionReference done_cb_;
    std::string             path_;
    LoadHandle *            handle_;  // owned; transferred to JS on success
    AddonState *            state_ = nullptr;
};

// Registers `loadModel(path, opts, done)` as a module-level function on
// `exports`.
void RegisterLoadModel(Napi::Env env, Napi::Object exports);
