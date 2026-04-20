#pragma once

#include <napi.h>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "llama.h"

// Forward-declared here to keep common/chat.h (and its heavy Jinja / JSON
// transitive includes) out of this header. Defined in common/chat.h; freed
// via common_chat_templates_free() from the .cpp.
struct common_chat_templates;

// Per-request cancellation state. Shared between LlamaModel's request map
// and the GenerateWorker running the request, so either side can signal
// cancellation without touching the other's lifetime.
struct RequestState {
    uint32_t id = 0;
    std::atomic<bool> cancel{false};
};

class LlamaModel : public Napi::ObjectWrap<LlamaModel> {
public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);

    explicit LlamaModel(const Napi::CallbackInfo & info);
    ~LlamaModel();

    // Accessors used by GenerateWorker.
    std::mutex &                         ctx_mutex()       { return ctx_mutex_; }
    bool                                 disposed() const  { return disposed_; }
    llama_context *                      ctx() const       { return ctx_; }
    const llama_vocab *                  vocab() const     { return vocab_; }
    void set_active_request(std::shared_ptr<RequestState> s) {
        active_request_ = std::move(s);
    }
    void clear_active_request() { active_request_.reset(); }

    // Remove a completed request from the map (called from wrapped_done).
    void UnregisterRequest(uint32_t req_id);

    // Prepare (reset + ensure) the llama_context for the next decode.
    // Must be called while ctx_mutex_ is held — the worker thread does this.
    // Moving this out of the JS main thread is critical: acquiring ctx_mutex_
    // on the main thread would block Node's event loop while any other
    // generation is decoding, stalling timers, abort signals, and I/O.
    bool PrepareContextLocked(uint32_t n_ctx, bool reset, std::string & error_out);

private:
    // JS-facing methods
    Napi::Value Generate(const Napi::CallbackInfo & info);
    void        Abort(const Napi::CallbackInfo & info);
    void        AbortRequest(const Napi::CallbackInfo & info);
    void        Dispose(const Napi::CallbackInfo & info);
    Napi::Value ContextLength(const Napi::CallbackInfo & info);
    Napi::Value ChatTemplate(const Napi::CallbackInfo & info);
    Napi::Value ApplyChatTemplate(const Napi::CallbackInfo & info);
    // Both implemented in chat_templates.cpp.
    //
    // ApplyChatTemplateJinja uses libcommon's Jinja renderer so it can take
    // tools / json_schema / enable_thinking and return the auto-generated
    // grammar + triggers + parser format name. Auto-falls back to the legacy
    // named-template path when the embedded template is an alias string.
    //
    // ParseChatResponse round-trips the opaque `parser` blob (from
    // common_peg_arena::save()) and extracts { content, reasoning, toolCalls }.
    Napi::Value ApplyChatTemplateJinja(const Napi::CallbackInfo & info);
    Napi::Value ParseChatResponse(const Napi::CallbackInfo & info);
    Napi::Value Tokenize(const Napi::CallbackInfo & info);
    Napi::Value Detokenize(const Napi::CallbackInfo & info);
    Napi::Value GetModelInfo(const Napi::CallbackInfo & info);

    // Internal helpers
    bool EnsureContext(uint32_t n_ctx, std::string & error_out);
    void CancelAll();

    static bool AbortCallback(void * data);

    llama_model *        model_   = nullptr;
    llama_context *      ctx_     = nullptr;
    const llama_vocab *  vocab_   = nullptr;
    uint32_t             n_ctx_   = 0;

    std::mutex           ctx_mutex_;

    // Request map: id → per-request state. Guarded by req_mutex_.
    std::mutex                                                   req_mutex_;
    std::unordered_map<uint32_t, std::shared_ptr<RequestState>>  requests_;
    uint32_t                                                     next_req_id_ = 1;

    // The request currently holding ctx_mutex_ and decoding. Only touched
    // by the worker thread under ctx_mutex_, and by AbortCallback which is
    // invoked from llama_decode on that same thread — no external race.
    std::shared_ptr<RequestState>                                active_request_;

    // Lazily initialised on first ApplyChatTemplateJinja call and reused for
    // the lifetime of the model. Freed in Dispose() / dtor.
    common_chat_templates * chat_templates_ = nullptr;

    bool disposed_ = false;
};
