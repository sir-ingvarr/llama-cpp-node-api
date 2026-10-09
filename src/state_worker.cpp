#include "state_worker.h"
#include "addon_state.h"

#include <mutex>
#include <utility>

// ---------------------------------------------------------------------------
// SaveStateWorker
// ---------------------------------------------------------------------------

SaveStateWorker::SaveStateWorker(Napi::Function &         done_cb,
                                 LlamaModel *             owner,
                                 std::string              path,
                                 std::vector<llama_token> tokens)
    : Napi::AsyncWorker(done_cb.Env()),
      owner_(owner),
      path_(std::move(path)),
      tokens_(std::move(tokens))
{
    done_cb_ = Napi::Persistent(done_cb);
}

void SaveStateWorker::Execute() {
    WorkerGuard guard(owner_->addon_state());
    if (guard.shutting_down()) {
        SetError("env is shutting down");
        return;
    }

    std::unique_lock<std::mutex> lock(owner_->ctx_mutex());
    if (owner_->disposed()) {
        SetError("model is disposed");
        return;
    }
    llama_context * ctx = owner_->ctx();
    if (!ctx) {
        SetError("saveState: no context to save — run a generation first");
        return;
    }

    if (!llama_state_save_file(ctx, path_.c_str(),
                               tokens_.data(), tokens_.size())) {
        SetError("llama_state_save_file failed: " + path_);
    }
}

void SaveStateWorker::OnOK() {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    done_cb_.Call({env.Null()});
}

void SaveStateWorker::OnError(const Napi::Error & error) {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    done_cb_.Call({error.Value()});
}

// ---------------------------------------------------------------------------
// LoadStateWorker
// ---------------------------------------------------------------------------

LoadStateWorker::LoadStateWorker(Napi::Function & done_cb,
                                 LlamaModel *     owner,
                                 std::string      path)
    : Napi::AsyncWorker(done_cb.Env()),
      owner_(owner),
      path_(std::move(path))
{
    done_cb_ = Napi::Persistent(done_cb);
}

void LoadStateWorker::Execute() {
    WorkerGuard guard(owner_->addon_state());
    if (guard.shutting_down()) {
        SetError("env is shutting down");
        return;
    }

    std::unique_lock<std::mutex> lock(owner_->ctx_mutex());
    if (owner_->disposed()) {
        SetError("model is disposed");
        return;
    }

    // The state must be restored into a context with the same configuration
    // it was saved from; create one with the model's current settings if a
    // generation hasn't run yet. No KV reset — the load overwrites it anyway.
    std::string ctx_err;
    if (!owner_->PrepareContextLocked(owner_->n_ctx_default(),
                                      /*reset=*/false, ctx_err)) {
        SetError(ctx_err);
        return;
    }
    llama_context * ctx = owner_->ctx();

    tokens_out_.resize(llama_n_ctx(ctx));
    size_t n_tokens = 0;
    if (!llama_state_load_file(ctx, path_.c_str(),
                               tokens_out_.data(), tokens_out_.size(),
                               &n_tokens)) {
        SetError("llama_state_load_file failed: " + path_ +
                 " (wrong model, context config, or state version?)");
        return;
    }
    tokens_out_.resize(n_tokens);
}

void LoadStateWorker::OnOK() {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    Napi::Array tokens = Napi::Array::New(env, tokens_out_.size());
    for (size_t i = 0; i < tokens_out_.size(); ++i) {
        tokens.Set((uint32_t) i, Napi::Number::New(env, tokens_out_[i]));
    }
    done_cb_.Call({env.Null(), tokens});
}

void LoadStateWorker::OnError(const Napi::Error & error) {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    done_cb_.Call({error.Value()});
}

// ---------------------------------------------------------------------------
// LlamaModel::SaveState / LoadState — JS entry points
// ---------------------------------------------------------------------------

// saveState(path, tokens: number[], done)
Napi::Value LlamaModel::SaveState(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (disposed_ || !model_) {
        Napi::Error::New(env, "LlamaModel has been disposed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 3 ||
        !info[0].IsString() ||
        !info[1].IsArray() ||
        !info[2].IsFunction()) {
        Napi::TypeError::New(env,
            "saveState(path: string, tokens: number[], done: fn)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string path = info[0].As<Napi::String>().Utf8Value();
    Napi::Array arr  = info[1].As<Napi::Array>();
    std::vector<llama_token> tokens;
    tokens.reserve(arr.Length());
    for (uint32_t i = 0; i < arr.Length(); ++i) {
        Napi::Value v = arr.Get(i);
        if (v.IsNumber()) {
            tokens.push_back((llama_token) v.As<Napi::Number>().Int32Value());
        }
    }
    Napi::Function done = info[2].As<Napi::Function>();

    auto * worker = new SaveStateWorker(done, this, std::move(path), std::move(tokens));
    worker->Queue();
    return env.Undefined();
}

// loadState(path, done) → done(err, tokens: number[])
Napi::Value LlamaModel::LoadState(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (disposed_ || !model_) {
        Napi::Error::New(env, "LlamaModel has been disposed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "loadState(path: string, done: fn)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string    path = info[0].As<Napi::String>().Utf8Value();
    Napi::Function done = info[1].As<Napi::Function>();

    auto * worker = new LoadStateWorker(done, this, std::move(path));
    worker->Queue();
    return env.Undefined();
}
