#pragma once

#include <napi.h>
#include <cstdint>
#include <string>

#include "llama.h"

struct AddonState;

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
