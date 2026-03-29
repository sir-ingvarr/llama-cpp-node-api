#pragma once

#include <napi.h>
#include <string>
#include <mutex>
#include <atomic>

#include "llama.h"

class LlamaModel : public Napi::ObjectWrap<LlamaModel> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    explicit LlamaModel(const Napi::CallbackInfo & info);
    ~LlamaModel();

private:
    // JS-facing methods
    void Generate(const Napi::CallbackInfo & info);
    void Abort(const Napi::CallbackInfo & info);
    void Dispose(const Napi::CallbackInfo & info);
    Napi::Value ContextLength(const Napi::CallbackInfo & info);
    Napi::Value ChatTemplate(const Napi::CallbackInfo & info);
    Napi::Value ApplyChatTemplate(const Napi::CallbackInfo & info);

    // Internal helpers
    bool EnsureContext(uint32_t n_ctx, std::string & error_out);

    static bool AbortCallback(void * data);

    llama_model *        model_   = nullptr;
    llama_context *      ctx_     = nullptr;
    const llama_vocab *  vocab_   = nullptr;
    uint32_t             n_ctx_   = 0;

    std::mutex           ctx_mutex_;
    std::atomic<bool>    cancel_{false};
    std::atomic<bool>    generating_{false};

    bool disposed_ = false;
};
