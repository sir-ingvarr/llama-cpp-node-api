#pragma once

#include <napi.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

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

// Reads the load mode from a JS options object. The explicit `loadMode` int
// (normalized from the string surface in js/index.js; covers 'directio' etc.)
// wins over the legacy useMmap/useMlock booleans.
inline llama_load_mode ResolveLoadModeFromOpts(Napi::Object opts) {
    if (opts.Has("loadMode") && opts.Get("loadMode").IsNumber()) {
        const int32_t m = opts.Get("loadMode").As<Napi::Number>().Int32Value();
        if (m >= LLAMA_LOAD_MODE_AUTO && m <= LLAMA_LOAD_MODE_DIRECT_IO) {
            return (llama_load_mode) m;
        }
    }
    bool has_opt   = false;
    bool use_mmap  = true;
    bool use_mlock = false;
    if (opts.Has("useMmap") && opts.Get("useMmap").IsBoolean()) {
        use_mmap = opts.Get("useMmap").As<Napi::Boolean>().Value();
        has_opt  = true;
    }
    if (opts.Has("useMlock") && opts.Get("useMlock").IsBoolean()) {
        use_mlock = opts.Get("useMlock").As<Napi::Boolean>().Value();
        has_opt   = true;
    }
    return ResolveLoadMode(has_opt, use_mmap, use_mlock);
}

// Parses the `loras` model-load option: an array of file paths or
// { path, scale? } objects. Invalid entries are skipped.
inline std::vector<std::pair<std::string, float>> ParseLoraSpecs(Napi::Object opts) {
    std::vector<std::pair<std::string, float>> specs;
    if (!opts.Has("loras") || !opts.Get("loras").IsArray()) return specs;
    Napi::Array arr = opts.Get("loras").As<Napi::Array>();
    for (uint32_t i = 0; i < arr.Length(); ++i) {
        Napi::Value v = arr.Get(i);
        if (v.IsString()) {
            specs.emplace_back(v.As<Napi::String>().Utf8Value(), 1.0f);
        } else if (v.IsObject()) {
            Napi::Object o = v.As<Napi::Object>();
            if (!o.Has("path") || !o.Get("path").IsString()) continue;
            float scale = 1.0f;
            if (o.Has("scale") && o.Get("scale").IsNumber()) {
                scale = o.Get("scale").As<Napi::Number>().FloatValue();
            }
            specs.emplace_back(o.Get("path").As<Napi::String>().Utf8Value(), scale);
        }
    }
    return specs;
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
    // resolved from the loadMode / useMmap / useMlock JS options.
    llama_load_mode load_mode = LLAMA_LOAD_MODE_AUTO;
    // LoRA adapter specs (path, scale) requested at load; the worker fills
    // `loras`/`lora_scales` with the loaded adapters, which the consuming
    // LlamaModel takes over. Adapters are owned by (and freed with) the model.
    std::vector<std::pair<std::string, float>> lora_specs;
    std::vector<llama_adapter_lora *>          loras;
    std::vector<float>                         lora_scales;
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
