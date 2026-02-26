#include "llama_model.h"
#include "generate_worker.h"

#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Static init
// ---------------------------------------------------------------------------

Napi::Object LlamaModel::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass(env, "LlamaModel", {
        InstanceMethod<&LlamaModel::Generate>("generate"),
        InstanceMethod<&LlamaModel::Abort>("abort"),
        InstanceMethod<&LlamaModel::Dispose>("dispose"),
        InstanceAccessor<&LlamaModel::ContextLength>("contextLength"),
    });

    Napi::FunctionReference * ctor = new Napi::FunctionReference();
    *ctor = Napi::Persistent(func);
    env.SetInstanceData(ctor);

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

    std::string model_path = info[0].As<Napi::String>().Utf8Value();

    int32_t n_gpu_layers = 99;
    uint32_t n_ctx_opt   = 2048;

    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("nGpuLayers") && opts.Get("nGpuLayers").IsNumber()) {
            n_gpu_layers = opts.Get("nGpuLayers").As<Napi::Number>().Int32Value();
        }
        if (opts.Has("nCtx") && opts.Get("nCtx").IsNumber()) {
            n_ctx_opt = opts.Get("nCtx").As<Napi::Number>().Uint32Value();
        }
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
        if (ctx_) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
        if (model_) {
            llama_model_free(model_);
            model_ = nullptr;
        }
    }
}

// ---------------------------------------------------------------------------
// AbortCallback (static) — called from llama_decode on the worker thread
// ---------------------------------------------------------------------------

bool LlamaModel::AbortCallback(void * data) {
    return static_cast<LlamaModel *>(data)->cancel_.load(
        std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// EnsureContext — create/reuse ctx_
// ---------------------------------------------------------------------------

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

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        error_out = "Failed to create llama_context";
        return false;
    }

    n_ctx_ = n_ctx;
    return true;
}

// ---------------------------------------------------------------------------
// generate(prompt, opts, onToken, onDone)
// ---------------------------------------------------------------------------

void LlamaModel::Generate(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();

    if (disposed_) {
        Napi::Error::New(env, "LlamaModel has been disposed")
            .ThrowAsJavaScriptException();
        return;
    }

    if (info.Length() < 4 ||
        !info[0].IsString() ||
        !info[1].IsObject() ||
        !info[2].IsFunction() ||
        !info[3].IsFunction()) {
        Napi::TypeError::New(env,
            "generate(prompt: string, opts: object, onToken: fn, onDone: fn)")
            .ThrowAsJavaScriptException();
        return;
    }

    if (generating_.exchange(true)) {
        Napi::Error::New(env, "Already generating — call abort() first")
            .ThrowAsJavaScriptException();
        return;
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
    bool reset_context = false;
    if (opts.Has("resetContext") && opts.Get("resetContext").IsBoolean()) {
        reset_context = opts.Get("resetContext").As<Napi::Boolean>().Value();
    }

    // Ensure context exists (may recreate if n_ctx changed)
    if (reset_context && ctx_) {
        llama_free(ctx_);
        ctx_  = nullptr;
        n_ctx_ = 0;
    }

    std::string ctx_error;
    if (!EnsureContext(n_ctx, ctx_error)) {
        generating_.store(false);
        Napi::Error::New(env, ctx_error).ThrowAsJavaScriptException();
        return;
    }

    // Reset cancel flag before starting
    cancel_.store(false);

    // Hold a strong reference to this JS object so the GC cannot collect it
    // while the worker thread is running.
    this->Ref();

    // Wrap done_cb in a thin function that:
    //   1. Drops the strong ref so the model can be GC'd once generation ends.
    //   2. Clears the generating_ flag.
    //   3. Forwards to the caller's done callback.
    std::atomic<bool> * gen_flag = &generating_;
    LlamaModel * self = this;
    Napi::Function wrapped_done = Napi::Function::New(env,
        [done_cb_ref = Napi::Persistent(done_cb), gen_flag, self]
        (const Napi::CallbackInfo & ci) mutable {
            self->Unref();
            gen_flag->store(false);
            Napi::Value arg = ci.Length() > 0 ? ci[0] : ci.Env().Null();
            done_cb_ref.Call({arg});
        }, "wrappedDone");

    auto * worker = new GenerateWorker(
        token_cb, wrapped_done,
        ctx_, vocab_,
        ctx_mutex_, cancel_,
        prompt,
        n_predict, temperature, top_p, top_k,
        min_p, repeat_penalty, repeat_last_n,
        std::move(grammar_str), std::move(stop_sequences)
    );

    worker->Queue();
}

// ---------------------------------------------------------------------------
// abort()
// ---------------------------------------------------------------------------

void LlamaModel::Abort(const Napi::CallbackInfo & /*info*/) {
    cancel_.store(true, std::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// dispose()
// ---------------------------------------------------------------------------

void LlamaModel::Dispose(const Napi::CallbackInfo & info) {
    if (disposed_) {
        return;
    }

    // Wait for any ongoing generation to finish before freeing resources
    std::lock_guard<std::mutex> lock(ctx_mutex_);

    disposed_ = true;
    cancel_.store(true);

    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
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
