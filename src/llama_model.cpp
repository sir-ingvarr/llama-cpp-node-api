#include "llama_model.h"
#include "generate_worker.h"
#include "embed_worker.h"
#include "addon_state.h"
#include "load_worker.h"  // LoadHandle (External payload)

#include "chat.h"  // libcommon: common_chat_templates_free

#include <algorithm>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Static init
// ---------------------------------------------------------------------------

Napi::Object LlamaModel::Init(Napi::Env env, Napi::Object exports) {
    // NOTE: use the runtime (non-template) InstanceMethod/InstanceAccessor
    // overloads — pass the method pointer as an argument, not as a template
    // parameter. The templated `InstanceMethod<&Method>` form triggers an MSVC
    // Internal Compiler Error (C1001) in node-addon-api's constexpr property
    // descriptor path on recent VS 2022 toolsets. Do NOT "modernize" these back
    // to the <&Method> form — it breaks the Windows build.
    Napi::Function func = DefineClass(env, "LlamaModel", {
        InstanceMethod("generate", &LlamaModel::Generate),
        InstanceMethod("abort", &LlamaModel::Abort),
        InstanceMethod("abortRequest", &LlamaModel::AbortRequest),
        InstanceMethod("dispose", &LlamaModel::Dispose),
        InstanceAccessor("contextLength", &LlamaModel::ContextLength, nullptr),
        InstanceAccessor("chatTemplate", &LlamaModel::ChatTemplate, nullptr),
        InstanceMethod("applyChatTemplate", &LlamaModel::ApplyChatTemplate),
        InstanceMethod("applyChatTemplateJinja", &LlamaModel::ApplyChatTemplateJinja),
        InstanceMethod("parseChatResponse", &LlamaModel::ParseChatResponse),
        InstanceMethod("tokenize", &LlamaModel::Tokenize),
        InstanceMethod("detokenize", &LlamaModel::Detokenize),
        InstanceMethod("getModelInfo", &LlamaModel::GetModelInfo),
        InstanceMethod("embed", &LlamaModel::Embed),
        InstanceMethod("saveState", &LlamaModel::SaveState),
        InstanceMethod("loadState", &LlamaModel::LoadState),
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
    if (opts.Has("cacheTypeK") && opts.Get("cacheTypeK").IsNumber()) {
        cache_type_k_ = opts.Get("cacheTypeK").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("cacheTypeV") && opts.Get("cacheTypeV").IsNumber()) {
        cache_type_v_ = opts.Get("cacheTypeV").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("flashAttention") && opts.Get("flashAttention").IsNumber()) {
        flash_attn_type_ = opts.Get("flashAttention").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("nThreads") && opts.Get("nThreads").IsNumber()) {
        n_threads_ = opts.Get("nThreads").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("nThreadsBatch") && opts.Get("nThreadsBatch").IsNumber()) {
        n_threads_batch_ = opts.Get("nThreadsBatch").As<Napi::Number>().Int32Value();
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
        cache_type_k_     = handle->cache_type_k;
        cache_type_v_     = handle->cache_type_v;
        flash_attn_type_  = handle->flash_attn_type;
        n_threads_        = handle->n_threads;
        n_threads_batch_  = handle->n_threads_batch;
        loras_            = std::move(handle->loras);
        lora_scales_      = std::move(handle->lora_scales);
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
    model_params.load_mode = ResolveLoadModeFromOpts(opts);

    model_ = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model_) {
        Napi::Error::New(env,
            std::string("Failed to load model: ") + model_path)
            .ThrowAsJavaScriptException();
        return;
    }

    // LoRA adapters: load now (sync path mirrors the sync model load).
    // Adapters are owned by the model — llama frees them with the model.
    for (const auto & [lpath, scale] : ParseLoraSpecs(opts)) {
        llama_adapter_lora * adapter =
            llama_adapter_lora_init(model_, lpath.c_str());
        if (!adapter) {
            llama_model_free(model_);
            model_ = nullptr;
            Napi::Error::New(env,
                std::string("Failed to load LoRA adapter: ") + lpath)
                .ThrowAsJavaScriptException();
            return;
        }
        loras_.push_back(adapter);
        lora_scales_.push_back(scale);
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
    if (cache_type_k_ >= 0) {
        ctx_params.type_k = (enum ggml_type) cache_type_k_;
    }
    if (cache_type_v_ >= 0) {
        ctx_params.type_v = (enum ggml_type) cache_type_v_;
    }
    if (flash_attn_type_ >= -1 && flash_attn_type_ <= 1) {
        // Valid enum range: -1 (AUTO), 0 (DISABLED), 1 (ENABLED). -1 is itself a
        // valid value, so assign whenever in range. Out-of-range ints (e.g. from
        // a direct native caller bypassing JS normalization) are ignored, leaving
        // the default AUTO rather than casting garbage to the enum (UB).
        ctx_params.flash_attn_type = (enum llama_flash_attn_type) flash_attn_type_;
    }
    if (n_threads_ > 0) {
        ctx_params.n_threads = n_threads_;
    }
    if (n_threads_batch_ > 0) {
        ctx_params.n_threads_batch = n_threads_batch_;
    }

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        error_out = "Failed to create llama_context";
        return false;
    }

    // Attach LoRA adapters (if any) to the fresh context.
    if (!loras_.empty()) {
        if (llama_set_adapters_lora(ctx_, loras_.data(), loras_.size(),
                                    lora_scales_.data()) != 0) {
            llama_free(ctx_);
            ctx_ = nullptr;
            error_out = "llama_set_adapters_lora failed";
            return false;
        }
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

    // Read options into a GenerateParams snapshot (defaults in the struct).
    GenerateParams gp;
    gp.prompt = std::move(prompt);
    gp.n_ctx  = n_ctx_;

    auto get_f32 = [&](const char * key, float & out) {
        if (opts.Has(key) && opts.Get(key).IsNumber()) {
            out = opts.Get(key).As<Napi::Number>().FloatValue();
        }
    };
    auto get_i32 = [&](const char * key, int32_t & out) {
        if (opts.Has(key) && opts.Get(key).IsNumber()) {
            out = opts.Get(key).As<Napi::Number>().Int32Value();
        }
    };
    auto get_str = [&](const char * key, std::string & out) {
        if (opts.Has(key) && opts.Get(key).IsString()) {
            out = opts.Get(key).As<Napi::String>().Utf8Value();
        }
    };
    auto get_str_array = [&](const char * key, std::vector<std::string> & out) {
        if (!opts.Has(key) || !opts.Get(key).IsArray()) return;
        Napi::Array arr = opts.Get(key).As<Napi::Array>();
        out.clear();
        for (uint32_t i = 0; i < arr.Length(); ++i) {
            Napi::Value v = arr.Get(i);
            if (v.IsString()) {
                out.push_back(v.As<Napi::String>().Utf8Value());
            }
        }
    };

    get_i32("nPredict",         gp.n_predict);
    get_f32("temperature",      gp.temperature);
    get_f32("dynatempRange",    gp.dynatemp_range);
    get_f32("dynatempExponent", gp.dynatemp_exponent);
    get_f32("topP",             gp.top_p);
    get_i32("topK",             gp.top_k);
    get_f32("minP",             gp.min_p);
    get_f32("typicalP",         gp.typical_p);
    get_f32("xtcProbability",   gp.xtc_probability);
    get_f32("xtcThreshold",     gp.xtc_threshold);
    get_f32("topNSigma",        gp.top_n_sigma);
    get_f32("repeatPenalty",    gp.repeat_penalty);
    get_i32("repeatLastN",      gp.repeat_last_n);
    get_f32("dryMultiplier",    gp.dry_multiplier);
    get_f32("dryBase",          gp.dry_base);
    get_i32("dryAllowedLength", gp.dry_allowed_length);
    get_i32("dryPenaltyLastN",  gp.dry_penalty_last_n);
    get_str_array("drySequenceBreakers", gp.dry_sequence_breakers);
    get_i32("mirostat",         gp.mirostat);
    get_f32("mirostatTau",      gp.mirostat_tau);
    get_f32("mirostatEta",      gp.mirostat_eta);
    get_f32("adaptivePTarget",  gp.adaptive_p_target);
    get_f32("adaptivePDecay",   gp.adaptive_p_decay);
    if (gp.mirostat < 0 || gp.mirostat > 2) {
        Napi::RangeError::New(env, "generate: mirostat must be 0, 1 or 2")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    if (opts.Has("nCtx") && opts.Get("nCtx").IsNumber()) {
        gp.n_ctx = opts.Get("nCtx").As<Napi::Number>().Uint32Value();
    }
    get_str("grammar", gp.grammar);
    get_str_array("stop",                   gp.stop_sequences);
    get_str_array("grammarTriggerPatterns", gp.grammar_trigger_patterns);
    get_str_array("preservedTokens",        gp.preserved_tokens);
    if (opts.Has("grammarTriggerTokens") && opts.Get("grammarTriggerTokens").IsArray()) {
        Napi::Array arr = opts.Get("grammarTriggerTokens").As<Napi::Array>();
        for (uint32_t i = 0; i < arr.Length(); ++i) {
            Napi::Value v = arr.Get(i);
            if (v.IsNumber()) {
                gp.grammar_trigger_tokens.push_back(v.As<Napi::Number>().Int32Value());
            }
        }
    }
    if (opts.Has("resetContext") && opts.Get("resetContext").IsBoolean()) {
        gp.reset_context = opts.Get("resetContext").As<Napi::Boolean>().Value();
    }
    if (opts.Has("logprobs") && opts.Get("logprobs").IsBoolean()) {
        gp.want_logprobs = opts.Get("logprobs").As<Napi::Boolean>().Value();
    }
    if (opts.Has("topLogprobs") && opts.Get("topLogprobs").IsNumber()) {
        gp.top_logprobs_n = opts.Get("topLogprobs").As<Napi::Number>().Int32Value();
        if (gp.top_logprobs_n > 0) gp.want_logprobs = true;  // imply logprobs
    }
    if (opts.Has("seed") && opts.Get("seed").IsNumber()) {
        gp.seed = opts.Get("seed").As<Napi::Number>().Int64Value();
    }
    if (opts.Has("logitBias") && opts.Get("logitBias").IsObject()) {
        Napi::Object lb = opts.Get("logitBias").As<Napi::Object>();
        Napi::Array  keys = lb.GetPropertyNames();
        gp.logit_bias.reserve(keys.Length());
        for (uint32_t i = 0; i < keys.Length(); ++i) {
            Napi::Value k = keys.Get(i);
            // Numeric-string keys (JS object keys are always strings) →
            // parse to int. Skip non-numeric keys and non-finite values.
            std::string ks = k.ToString().Utf8Value();
            char * end = nullptr;
            long tok_id = std::strtol(ks.c_str(), &end, 10);
            if (end == ks.c_str() || *end != '\0') continue;
            Napi::Value v = lb.Get(k);
            if (!v.IsNumber()) continue;
            float bias = v.As<Napi::Number>().FloatValue();
            gp.logit_bias.emplace_back((int32_t) tok_id, bias);
        }
    }

    // Reasoning budget (tokens inside the thinking block). The tags are
    // tokenized on the worker thread; chat() forwards the template's
    // thinkingStartTag / thinkingEndTags automatically.
    get_i32("reasoningBudget",         gp.reasoning_budget);
    get_str("reasoningBudgetStartTag", gp.reasoning_budget_start_tag);
    get_str_array("reasoningBudgetEndTags", gp.reasoning_budget_end_tags);
    get_str("reasoningBudgetMessage",  gp.reasoning_budget_message);
    get_str("reasoningBudgetPrefill",  gp.reasoning_budget_prefill);

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
        std::move(gp)
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
        Napi::Value el = arr.Get(i);
        if (!el.IsNumber()) {
            Napi::TypeError::New(env, "detokenize: tokens must be an array of numbers")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }
        tokens[i] = el.As<Napi::Number>().Int32Value();
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

    // Whisper-style suppress list baked into the vocab metadata (empty for
    // most text models). Useful for seeding logitBias with -Infinity bans.
    {
        int32_t n_suppress = 0;
        const llama_token * suppress =
            llama_vocab_get_suppress_tokens(vocab_, &n_suppress);
        Napi::Array arr = Napi::Array::New(env, (size_t) std::max(n_suppress, 0));
        for (int32_t i = 0; i < n_suppress; ++i) {
            arr.Set((uint32_t) i, Napi::Number::New(env, suppress[i]));
        }
        obj.Set("suppressTokens", arr);
    }

    return obj;
}
