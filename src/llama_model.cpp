#include "llama_model.h"
#include "generate_worker.h"
#include "embed_worker.h"
#include "addon_state.h"
#include "load_worker.h"  // LoadHandle (External payload)

#include "chat.h"  // libcommon: common_chat_templates_free

#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Static init
// ---------------------------------------------------------------------------

Napi::Object LlamaModel::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "LlamaModel", {
        InstanceMethod<&LlamaModel::Generate>("generate"),
        InstanceMethod<&LlamaModel::Abort>("abort"),
        InstanceMethod<&LlamaModel::AbortRequest>("abortRequest"),
        InstanceMethod<&LlamaModel::Dispose>("dispose"),
        InstanceAccessor<&LlamaModel::ContextLength>("contextLength"),
        InstanceAccessor<&LlamaModel::ChatTemplate>("chatTemplate"),
        InstanceMethod<&LlamaModel::ApplyChatTemplate>("applyChatTemplate"),
        InstanceMethod<&LlamaModel::ApplyChatTemplateJinja>("applyChatTemplateJinja"),
        InstanceMethod<&LlamaModel::ParseChatResponse>("parseChatResponse"),
        InstanceMethod<&LlamaModel::Tokenize>("tokenize"),
        InstanceMethod<&LlamaModel::Detokenize>("detokenize"),
        InstanceMethod<&LlamaModel::GetModelInfo>("getModelInfo"),
        InstanceMethod<&LlamaModel::Embed>("embed"),
    });

    // Note: no SetInstanceData here — the env's instance-data slot is reserved
    // for AddonState (set by InitModule). The exported function value rooted
    // via `exports.Set` keeps the constructor alive for the env's lifetime.
    exports.Set("LlamaModel", func);
    return exports;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

LlamaModel::LlamaModel(const Napi::CallbackInfo & info)
    : Napi::ObjectWrap<LlamaModel>(info)
{
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "First argument must be the model path (string)")
            .ThrowAsJavaScriptException();
        return;
    }

    state_ = env.GetInstanceData<AddonState>();

    Napi::Object opts = (info.Length() >= 2 && info[1].IsObject())
        ? info[1].As<Napi::Object>()
        : Napi::Object::New(env);
    if (opts.Has("embeddings") && opts.Get("embeddings").IsBoolean()) {
        embeddings_ = opts.Get("embeddings").As<Napi::Boolean>().Value();
    }
    if (opts.Has("poolingType") && opts.Get("poolingType").IsNumber()) {
        pooling_type_ = opts.Get("poolingType").As<Napi::Number>().Int32Value();
    }

    // ----- Async path: 3rd arg is an External<LoadHandle> from LoadWorker.
    //       Take ownership of the already-loaded model; skip the sync load.
    if (info.Length() >= 3 && info[2].IsExternal()) {
        auto * handle = info[2].As<Napi::External<LoadHandle>>().Data();
        if (!handle || !handle->model) {
            Napi::Error::New(env,
                "LlamaModel: load handle is empty (already consumed?)")
                .ThrowAsJavaScriptException();
            return;
        }
        model_         = handle->model;
        handle->model  = nullptr;  // External finalizer must not double-free
        n_ctx_         = handle->n_ctx;
        embeddings_    = handle->embeddings;
        pooling_type_  = handle->pooling_type;
        vocab_         = llama_model_get_vocab(model_);
        return;
    }

    // ----- Sync path: load now, on the JS thread.
    std::string model_path = info[0].As<Napi::String>().Utf8Value();

    int32_t n_gpu_layers = 99;
    uint32_t n_ctx_opt   = 2048;

    if (opts.Has("nGpuLayers") && opts.Get("nGpuLayers").IsNumber()) {
        n_gpu_layers = opts.Get("nGpuLayers").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("nCtx") && opts.Get("nCtx").IsNumber()) {
        n_ctx_opt = opts.Get("nCtx").As<Napi::Number>().Uint32Value();
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;
    model_params.progress_callback = [](float, void *) { return true; };

    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        Napi::Error::New(env,
            std::string("Failed to load model: ") + model_path)
            .ThrowAsJavaScriptException();
        return;
    }

    vocab_ = llama_model_get_vocab(model_);
    n_ctx_ = n_ctx_opt;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

LlamaModel::~LlamaModel() {
    if (!disposed_) {
        CancelAll();
        std::lock_guard<std::mutex> lock(ctx_mutex_);
        if (ctx_) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
        if (model_) {
            llama_model_free(model_);
            model_ = nullptr;
        }
        vocab_ = nullptr;
    }
    if (chat_templates_) {
        common_chat_templates_free(chat_templates_);
        chat_templates_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// AbortCallback — invoked from inside llama_decode on the worker thread.
// Reads the active request's cancel flag. Only the worker holding ctx_mutex_
// writes active_request_, and llama_decode runs on that same thread, so no
// additional synchronisation is required here.
// ---------------------------------------------------------------------------

bool LlamaModel::AbortCallback(void * data) {
    auto * self = static_cast<LlamaModel *>(data);
    // Short-circuit on env shutdown so llama_decode returns promptly during
    // teardown rather than running to the next token boundary.
    if (self->state_ && self->state_->shutting_down.load(std::memory_order_acquire)) {
        return true;
    }
    auto & state = self->active_request_;
    return state && state->cancel.load(std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// EnsureContext — create/reuse ctx_
// ---------------------------------------------------------------------------

bool LlamaModel::PrepareContextLocked(uint32_t n_ctx, bool reset, std::string & error_out) {
    if (disposed_ || !model_) {
        error_out = "LlamaModel has been disposed";
        return false;
    }
    if (reset && ctx_) {
        llama_free(ctx_);
        ctx_   = nullptr;
        n_ctx_ = 0;
    }
    return EnsureContext(n_ctx, error_out);
}

bool LlamaModel::EnsureContext(uint32_t n_ctx, std::string & error_out) {
    if (ctx_ && n_ctx_ == n_ctx) {
        return true;  // reuse existing
    }

    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = n_ctx;
    ctx_params.n_batch  = n_ctx;
    ctx_params.abort_callback      = &LlamaModel::AbortCallback;
    ctx_params.abort_callback_data = this;
    // Embedding mode is set at construction; honored when creating the ctx.
    if (embeddings_) {
        ctx_params.embeddings = true;
    }
    if (pooling_type_ >= 0) {
        ctx_params.pooling_type = (enum llama_pooling_type) pooling_type_;
    }

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        error_out = "Failed to create llama_context";
        return false;
    }

    n_ctx_ = n_ctx;
    return true;
}

// ---------------------------------------------------------------------------
// CancelAll — flip the cancel flag on every tracked request.
// Used from abort() and Dispose() to unblock queued and running workers.
// ---------------------------------------------------------------------------

void LlamaModel::CancelAll() {
    std::lock_guard<std::mutex> lk(req_mutex_);
    for (auto & [id, state] : requests_) {
        (void)id;
        state->cancel.store(true, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// UnregisterRequest — called from the JS-side wrapped_done after completion.
// ---------------------------------------------------------------------------

void LlamaModel::UnregisterRequest(uint32_t req_id) {
    std::lock_guard<std::mutex> lk(req_mutex_);
    requests_.erase(req_id);
}

// ---------------------------------------------------------------------------
// generate(prompt, opts, onToken, onDone) → request id
// ---------------------------------------------------------------------------

Napi::Value LlamaModel::Generate(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();

    if (disposed_) {
        Napi::Error::New(env, "LlamaModel has been disposed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (info.Length() < 4 ||
        !info[0].IsString() ||
        !info[1].IsObject() ||
        !info[2].IsFunction() ||
        !info[3].IsFunction()) {
        Napi::TypeError::New(env,
            "generate(prompt: string, opts: object, onToken: fn, onDone: fn)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string prompt = info[0].As<Napi::String>().Utf8Value();
    Napi::Object opts  = info[1].As<Napi::Object>();
    Napi::Function token_cb = info[2].As<Napi::Function>();
    Napi::Function done_cb  = info[3].As<Napi::Function>();

    // Read options
    int32_t  n_predict      = 256;
    float    temperature    = 0.8f;
    float    top_p          = 0.95f;
    int32_t  top_k          = 40;
    float    min_p          = 0.0f;
    float    repeat_penalty = 1.0f;
    int32_t  repeat_last_n  = 64;
    uint32_t n_ctx          = n_ctx_;
    std::string              grammar_str;
    std::vector<std::string> stop_sequences;
    std::vector<std::string> grammar_trigger_patterns;
    std::vector<int32_t>     grammar_trigger_tokens;
    std::vector<std::string> preserved_tokens;

    if (opts.Has("nPredict") && opts.Get("nPredict").IsNumber()) {
        n_predict = opts.Get("nPredict").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("temperature") && opts.Get("temperature").IsNumber()) {
        temperature = opts.Get("temperature").As<Napi::Number>().FloatValue();
    }
    if (opts.Has("topP") && opts.Get("topP").IsNumber()) {
        top_p = opts.Get("topP").As<Napi::Number>().FloatValue();
    }
    if (opts.Has("topK") && opts.Get("topK").IsNumber()) {
        top_k = opts.Get("topK").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("minP") && opts.Get("minP").IsNumber()) {
        min_p = opts.Get("minP").As<Napi::Number>().FloatValue();
    }
    if (opts.Has("repeatPenalty") && opts.Get("repeatPenalty").IsNumber()) {
        repeat_penalty = opts.Get("repeatPenalty").As<Napi::Number>().FloatValue();
    }
    if (opts.Has("repeatLastN") && opts.Get("repeatLastN").IsNumber()) {
        repeat_last_n = opts.Get("repeatLastN").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("nCtx") && opts.Get("nCtx").IsNumber()) {
        n_ctx = opts.Get("nCtx").As<Napi::Number>().Uint32Value();
    }
    if (opts.Has("grammar") && opts.Get("grammar").IsString()) {
        grammar_str = opts.Get("grammar").As<Napi::String>().Utf8Value();
    }
    if (opts.Has("stop") && opts.Get("stop").IsArray()) {
        Napi::Array arr = opts.Get("stop").As<Napi::Array>();
        for (uint32_t i = 0; i < arr.Length(); ++i) {
            Napi::Value v = arr.Get(i);
            if (v.IsString()) {
                stop_sequences.push_back(v.As<Napi::String>().Utf8Value());
            }
        }
    }
    if (opts.Has("grammarTriggerPatterns") && opts.Get("grammarTriggerPatterns").IsArray()) {
        Napi::Array arr = opts.Get("grammarTriggerPatterns").As<Napi::Array>();
        for (uint32_t i = 0; i < arr.Length(); ++i) {
            Napi::Value v = arr.Get(i);
            if (v.IsString()) {
                grammar_trigger_patterns.push_back(v.As<Napi::String>().Utf8Value());
            }
        }
    }
    if (opts.Has("grammarTriggerTokens") && opts.Get("grammarTriggerTokens").IsArray()) {
        Napi::Array arr = opts.Get("grammarTriggerTokens").As<Napi::Array>();
        for (uint32_t i = 0; i < arr.Length(); ++i) {
            Napi::Value v = arr.Get(i);
            if (v.IsNumber()) {
                grammar_trigger_tokens.push_back(v.As<Napi::Number>().Int32Value());
            }
        }
    }
    if (opts.Has("preservedTokens") && opts.Get("preservedTokens").IsArray()) {
        Napi::Array arr = opts.Get("preservedTokens").As<Napi::Array>();
        for (uint32_t i = 0; i < arr.Length(); ++i) {
            Napi::Value v = arr.Get(i);
            if (v.IsString()) {
                preserved_tokens.push_back(v.As<Napi::String>().Utf8Value());
            }
        }
    }
    bool reset_context = false;
    if (opts.Has("resetContext") && opts.Get("resetContext").IsBoolean()) {
        reset_context = opts.Get("resetContext").As<Napi::Boolean>().Value();
    }
    bool    want_logprobs   = false;
    int32_t top_logprobs_n  = 0;
    if (opts.Has("logprobs") && opts.Get("logprobs").IsBoolean()) {
        want_logprobs = opts.Get("logprobs").As<Napi::Boolean>().Value();
    }
    if (opts.Has("topLogprobs") && opts.Get("topLogprobs").IsNumber()) {
        top_logprobs_n = opts.Get("topLogprobs").As<Napi::Number>().Int32Value();
        if (top_logprobs_n > 0) want_logprobs = true;  // imply logprobs
    }

    // Context setup (possibly involving llama_free / llama_init_from_model)
    // is done by the worker under ctx_mutex_ — never on the JS main thread,
    // which would block Node's event loop while other generations decode.

    // Allocate a request state + id and register it so abort() can find it.
    auto state = std::make_shared<RequestState>();
    uint32_t req_id;
    {
        std::lock_guard<std::mutex> lk(req_mutex_);
        req_id = next_req_id_++;
        state->id = req_id;
        requests_[req_id] = state;
    }

    // Hold a strong reference to this JS object so the GC cannot collect it
    // while the worker thread is running.
    this->Ref();

    // Wrap done_cb so it:
    //   1. Removes the request from the map.
    //   2. Drops the strong ref.
    //   3. Forwards to the caller's done callback.
    LlamaModel * self = this;
    Napi::Function wrapped_done = Napi::Function::New(env,
        [done_cb_ref = Napi::Persistent(done_cb), self, req_id]
        (const Napi::CallbackInfo & ci) mutable {
            self->UnregisterRequest(req_id);
            self->Unref();
            Napi::Value arg = ci.Length() > 0 ? ci[0] : ci.Env().Null();
            done_cb_ref.Call({arg});
        }, "wrappedDone");

    auto * worker = new GenerateWorker(
        token_cb, wrapped_done,
        this, state,
        prompt,
        n_ctx, reset_context,
        n_predict, temperature, top_p, top_k,
        min_p, repeat_penalty, repeat_last_n,
        std::move(grammar_str), std::move(stop_sequences),
        std::move(grammar_trigger_patterns),
        std::move(grammar_trigger_tokens),
        std::move(preserved_tokens),
        want_logprobs, top_logprobs_n
    );

    worker->Queue();

    return Napi::Number::New(env, (double)req_id);
}

// ---------------------------------------------------------------------------
// abort() — cancel current + all queued.
// ---------------------------------------------------------------------------

void LlamaModel::Abort(const Napi::CallbackInfo & /*info*/) {
    CancelAll();
}

// ---------------------------------------------------------------------------
// abortRequest(id) — cancel a specific in-flight or queued request.
// ---------------------------------------------------------------------------

void LlamaModel::AbortRequest(const Napi::CallbackInfo & info) {
    if (info.Length() < 1 || !info[0].IsNumber()) {
        return;
    }
    uint32_t id = info[0].As<Napi::Number>().Uint32Value();
    std::lock_guard<std::mutex> lk(req_mutex_);
    auto it = requests_.find(id);
    if (it != requests_.end()) {
        it->second->cancel.store(true, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// embed(text, done) — async embed via EmbedWorker (model must be in
// embedding mode, i.e. constructed with `embeddings: true`).
// ---------------------------------------------------------------------------

Napi::Value LlamaModel::Embed(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (!model_) {
        Napi::Error::New(env, "LlamaModel has been disposed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (!embeddings_) {
        Napi::Error::New(env,
            "embed() requires the model to be loaded with { embeddings: true }")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "embed(text: string, done: fn)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string text = info[0].As<Napi::String>().Utf8Value();
    Napi::Function done = info[1].As<Napi::Function>();

    auto * worker = new EmbedWorker(done, this, std::move(text));
    worker->Queue();
    return env.Undefined();
}

// ---------------------------------------------------------------------------
// dispose() — cancel all, wait for current to finish, free resources.
// ---------------------------------------------------------------------------

void LlamaModel::Dispose(const Napi::CallbackInfo & /*info*/) {
    if (disposed_) {
        return;
    }

    // Signal every tracked request to stop; queued workers will fast-reject.
    CancelAll();

    // Wait for the currently running worker (if any) to release the ctx.
    std::lock_guard<std::mutex> lock(ctx_mutex_);

    disposed_ = true;

    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    vocab_ = nullptr;
    if (chat_templates_) {
        common_chat_templates_free(chat_templates_);
        chat_templates_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// contextLength getter
// ---------------------------------------------------------------------------

Napi::Value LlamaModel::ContextLength(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (!ctx_) {
        return Napi::Number::New(env, 0);
    }
    return Napi::Number::New(env, (double)llama_n_ctx(ctx_));
}

// ---------------------------------------------------------------------------
// chatTemplate getter
// ---------------------------------------------------------------------------

Napi::Value LlamaModel::ChatTemplate(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (!model_) {
        return env.Null();
    }
    const char * tmpl = llama_model_chat_template(model_, nullptr);
    if (!tmpl) {
        return env.Null();
    }
    return Napi::String::New(env, tmpl);
}

// ---------------------------------------------------------------------------
// applyChatTemplate(messages, opts?)
// ---------------------------------------------------------------------------

Napi::Value LlamaModel::ApplyChatTemplate(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();

    if (!model_) {
        Napi::Error::New(env, "LlamaModel has been disposed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env,
            "applyChatTemplate(messages: Array<{role, content}>, opts?: {addAssistant?: boolean})")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array msgs = info[0].As<Napi::Array>();
    uint32_t n_msg = msgs.Length();

    // Parse options
    bool add_assistant = true;
    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("addAssistant") && opts.Get("addAssistant").IsBoolean()) {
            add_assistant = opts.Get("addAssistant").As<Napi::Boolean>().Value();
        }
    }

    // Convert JS messages to llama_chat_message array.
    // Keep std::string values alive until after the call.
    std::vector<std::string> roles(n_msg);
    std::vector<std::string> contents(n_msg);
    std::vector<llama_chat_message> chat(n_msg);

    for (uint32_t i = 0; i < n_msg; ++i) {
        Napi::Value item = msgs.Get(i);
        if (!item.IsObject()) {
            Napi::TypeError::New(env, "Each message must be an object with 'role' and 'content'")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }
        Napi::Object msg = item.As<Napi::Object>();
        Napi::Value role_v    = msg.Get("role");
        Napi::Value content_v = msg.Get("content");
        if (!role_v.IsString() || !content_v.IsString()) {
            Napi::TypeError::New(env,
                "Each message must have string 'role' and string 'content'")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }
        roles[i]    = role_v.As<Napi::String>().Utf8Value();
        contents[i] = content_v.As<Napi::String>().Utf8Value();
        chat[i].role    = roles[i].c_str();
        chat[i].content = contents[i].c_str();
    }

    // Get model's built-in template
    const char * tmpl = llama_model_chat_template(model_, nullptr);

    // First call: determine required buffer size
    int32_t len = llama_chat_apply_template(tmpl, chat.data(), n_msg, add_assistant, nullptr, 0);
    if (len < 0) {
        Napi::Error::New(env, "Failed to apply chat template — template may be unsupported")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Second call: render into buffer
    std::vector<char> buf(len + 1);
    llama_chat_apply_template(tmpl, chat.data(), n_msg, add_assistant, buf.data(), buf.size());

    return Napi::String::New(env, buf.data(), len);
}

// ---------------------------------------------------------------------------
// tokenize(text, opts?)
// ---------------------------------------------------------------------------

Napi::Value LlamaModel::Tokenize(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();

    if (!model_) {
        Napi::Error::New(env, "LlamaModel has been disposed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "tokenize(text: string, opts?: {addSpecial?: boolean, parseSpecial?: boolean})")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string text = info[0].As<Napi::String>().Utf8Value();

    bool add_special   = true;
    bool parse_special  = false;
    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("addSpecial") && opts.Get("addSpecial").IsBoolean())
            add_special = opts.Get("addSpecial").As<Napi::Boolean>().Value();
        if (opts.Has("parseSpecial") && opts.Get("parseSpecial").IsBoolean())
            parse_special = opts.Get("parseSpecial").As<Napi::Boolean>().Value();
    }

    // First call with negative n_tokens_max to get the required count
    int32_t n = llama_tokenize(vocab_, text.c_str(), text.size(), nullptr, 0, add_special, parse_special);
    if (n < 0) n = -n;

    std::vector<llama_token> tokens(n);
    int32_t actual = llama_tokenize(vocab_, text.c_str(), text.size(), tokens.data(), tokens.size(), add_special, parse_special);
    if (actual < 0) {
        Napi::Error::New(env, "Tokenization failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    tokens.resize(actual);

    Napi::Array result = Napi::Array::New(env, tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        result.Set(i, Napi::Number::New(env, tokens[i]));
    }
    return result;
}

// ---------------------------------------------------------------------------
// detokenize(tokens, opts?)
// ---------------------------------------------------------------------------

Napi::Value LlamaModel::Detokenize(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();

    if (!model_) {
        Napi::Error::New(env, "LlamaModel has been disposed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (info.Length() < 1 || !info[0].IsArray()) {
        Napi::TypeError::New(env, "detokenize(tokens: number[], opts?: {removeSpecial?: boolean, unparseSpecial?: boolean})")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array arr = info[0].As<Napi::Array>();
    uint32_t n = arr.Length();
    std::vector<llama_token> tokens(n);
    for (uint32_t i = 0; i < n; ++i) {
        tokens[i] = arr.Get(i).As<Napi::Number>().Int32Value();
    }

    bool remove_special = false;
    bool unparse_special = false;
    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("removeSpecial") && opts.Get("removeSpecial").IsBoolean())
            remove_special = opts.Get("removeSpecial").As<Napi::Boolean>().Value();
        if (opts.Has("unparseSpecial") && opts.Get("unparseSpecial").IsBoolean())
            unparse_special = opts.Get("unparseSpecial").As<Napi::Boolean>().Value();
    }

    // First call to get required size
    int32_t len = llama_detokenize(vocab_, tokens.data(), tokens.size(), nullptr, 0, remove_special, unparse_special);
    if (len < 0) len = -len;

    std::vector<char> buf(len + 1);
    int32_t actual = llama_detokenize(vocab_, tokens.data(), tokens.size(), buf.data(), buf.size(), remove_special, unparse_special);
    if (actual < 0) {
        Napi::Error::New(env, "Detokenization failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return Napi::String::New(env, buf.data(), actual);
}

// ---------------------------------------------------------------------------
// getModelInfo()
// ---------------------------------------------------------------------------

Napi::Value LlamaModel::GetModelInfo(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();

    if (!model_) {
        Napi::Error::New(env, "LlamaModel has been disposed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object obj = Napi::Object::New(env);

    // Description. 512 is comfortably above the typical "<arch> <type> <params>
    // <ftype>" line llama.cpp produces, but llama_model_desc returns the bytes
    // it actually wrote so we use that to length-bound the JS string.
    char desc_buf[512];
    int desc_n = llama_model_desc(model_, desc_buf, sizeof(desc_buf));
    if (desc_n < 0) desc_n = 0;
    if ((size_t) desc_n > sizeof(desc_buf)) desc_n = (int) sizeof(desc_buf);
    obj.Set("description",
        Napi::String::New(env, desc_buf, (size_t) desc_n));

    // Numeric properties
    obj.Set("nParams",           Napi::Number::New(env, (double)llama_model_n_params(model_)));
    obj.Set("modelSize",         Napi::Number::New(env, (double)llama_model_size(model_)));
    obj.Set("trainContextLength", Napi::Number::New(env, llama_model_n_ctx_train(model_)));
    obj.Set("embeddingSize",     Napi::Number::New(env, llama_model_n_embd(model_)));
    obj.Set("nLayer",            Napi::Number::New(env, llama_model_n_layer(model_)));
    obj.Set("vocabSize",         Napi::Number::New(env, llama_vocab_n_tokens(vocab_)));

    // Special tokens
    Napi::Object tokens = Napi::Object::New(env);
    tokens.Set("bos", Napi::Number::New(env, llama_vocab_bos(vocab_)));
    tokens.Set("eos", Napi::Number::New(env, llama_vocab_eos(vocab_)));
    tokens.Set("eot", Napi::Number::New(env, llama_vocab_eot(vocab_)));
    obj.Set("specialTokens", tokens);

    return obj;
}
