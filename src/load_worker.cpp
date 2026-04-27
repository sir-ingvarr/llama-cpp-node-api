#include "load_worker.h"
#include "addon_state.h"

LoadWorker::LoadWorker(Napi::Function & done_cb,
                       std::string      path,
                       LoadHandle *     handle)
    : Napi::AsyncWorker(done_cb.Env()),
      path_(std::move(path)),
      handle_(handle)
{
    done_cb_ = Napi::Persistent(done_cb);
    state_   = done_cb.Env().GetInstanceData<AddonState>();
}

void LoadWorker::Execute() {
    WorkerGuard guard(state_);
    if (guard.shutting_down()) {
        SetError("env is shutting down");
        return;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = handle_->n_gpu_layers;
    model_params.progress_callback  = [](float, void *) { return true; };

    handle_->model = llama_model_load_from_file(path_.c_str(), model_params);
    if (!handle_->model) {
        SetError(std::string("Failed to load model: ") + path_);
        return;
    }
}

void LoadWorker::OnOK() {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);

    // Hand off ownership of `handle_` to V8. The finalizer frees
    // `handle_->model` if the LlamaModel constructor never consumed it
    // (e.g. the caller dropped the Promise before constructing).
    auto external = Napi::External<LoadHandle>::New(
        env, handle_,
        [](Napi::Env, LoadHandle * h) {
            if (h && h->model) {
                llama_model_free(h->model);
            }
            delete h;
        });
    handle_ = nullptr;  // transferred to External
    done_cb_.Call({env.Null(), external});
}

void LoadWorker::OnError(const Napi::Error & error) {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    delete handle_;  // no model to free; it never loaded
    handle_ = nullptr;
    done_cb_.Call({error.Value()});
}

// loadModel(path, opts, done)
static Napi::Value LoadModel(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 ||
        !info[0].IsString() ||
        !info[1].IsObject() ||
        !info[2].IsFunction()) {
        Napi::TypeError::New(env, "loadModel(path: string, opts: object, done: fn)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string  path  = info[0].As<Napi::String>().Utf8Value();
    Napi::Object opts  = info[1].As<Napi::Object>();
    Napi::Function done = info[2].As<Napi::Function>();

    auto * handle = new LoadHandle();
    if (opts.Has("nGpuLayers") && opts.Get("nGpuLayers").IsNumber()) {
        handle->n_gpu_layers = opts.Get("nGpuLayers").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("nCtx") && opts.Get("nCtx").IsNumber()) {
        handle->n_ctx = opts.Get("nCtx").As<Napi::Number>().Uint32Value();
    }
    if (opts.Has("embeddings") && opts.Get("embeddings").IsBoolean()) {
        handle->embeddings = opts.Get("embeddings").As<Napi::Boolean>().Value();
    }
    if (opts.Has("poolingType") && opts.Get("poolingType").IsNumber()) {
        handle->pooling_type = opts.Get("poolingType").As<Napi::Number>().Int32Value();
    }

    auto * worker = new LoadWorker(done, std::move(path), handle);
    worker->Queue();
    return env.Undefined();
}

void RegisterLoadModel(Napi::Env env, Napi::Object exports) {
    exports.Set("loadModel", Napi::Function::New(env, LoadModel, "loadModel"));
}
