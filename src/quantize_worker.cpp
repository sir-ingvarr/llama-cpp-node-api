#include "quantize_worker.h"
#include "addon_state.h"

#include <string>
#include <unordered_map>

QuantizeWorker::QuantizeWorker(
    Napi::Function &                done_cb,
    std::string                     input_path,
    std::string                     output_path,
    llama_model_quantize_params     params
)
    : Napi::AsyncWorker(done_cb.Env()),
      input_path_(std::move(input_path)),
      output_path_(std::move(output_path)),
      params_(params)
{
    done_cb_ = Napi::Persistent(done_cb);
    state_   = done_cb.Env().GetInstanceData<AddonState>();
}

void QuantizeWorker::Execute() {
    WorkerGuard guard(state_);
    if (guard.shutting_down()) {
        SetError("env is shutting down");
        return;
    }
    const uint32_t ret =
        llama_model_quantize(input_path_.c_str(), output_path_.c_str(), &params_);
    if (ret != 0) {
        SetError("llama_model_quantize failed (code " + std::to_string(ret) + ")");
    }
}

void QuantizeWorker::OnOK() {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    done_cb_.Call({env.Null()});
}

void QuantizeWorker::OnError(const Napi::Error & error) {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    done_cb_.Call({error.Value()});
}

// ---------------------------------------------------------------------------
// ftype string → enum mapping.
// The names mirror llama.cpp's CLI convention (the `LLAMA_FTYPE_MOSTLY_`
// prefix is stripped). Case-insensitive on the JS side would be nice but is
// kept strict here — JS normalises before passing through.
// ---------------------------------------------------------------------------

static const std::unordered_map<std::string, llama_ftype> & ftype_map() {
    static const std::unordered_map<std::string, llama_ftype> m = {
        {"F32",       LLAMA_FTYPE_ALL_F32},
        {"F16",       LLAMA_FTYPE_MOSTLY_F16},
        {"BF16",      LLAMA_FTYPE_MOSTLY_BF16},
        {"Q4_0",      LLAMA_FTYPE_MOSTLY_Q4_0},
        {"Q4_1",      LLAMA_FTYPE_MOSTLY_Q4_1},
        {"Q5_0",      LLAMA_FTYPE_MOSTLY_Q5_0},
        {"Q5_1",      LLAMA_FTYPE_MOSTLY_Q5_1},
        {"Q8_0",      LLAMA_FTYPE_MOSTLY_Q8_0},
        {"Q2_K",      LLAMA_FTYPE_MOSTLY_Q2_K},
        {"Q2_K_S",    LLAMA_FTYPE_MOSTLY_Q2_K_S},
        {"Q3_K_S",    LLAMA_FTYPE_MOSTLY_Q3_K_S},
        {"Q3_K_M",    LLAMA_FTYPE_MOSTLY_Q3_K_M},
        {"Q3_K_L",    LLAMA_FTYPE_MOSTLY_Q3_K_L},
        {"Q4_K_S",    LLAMA_FTYPE_MOSTLY_Q4_K_S},
        {"Q4_K_M",    LLAMA_FTYPE_MOSTLY_Q4_K_M},
        {"Q5_K_S",    LLAMA_FTYPE_MOSTLY_Q5_K_S},
        {"Q5_K_M",    LLAMA_FTYPE_MOSTLY_Q5_K_M},
        {"Q6_K",      LLAMA_FTYPE_MOSTLY_Q6_K},
        {"IQ2_XXS",   LLAMA_FTYPE_MOSTLY_IQ2_XXS},
        {"IQ2_XS",    LLAMA_FTYPE_MOSTLY_IQ2_XS},
        {"IQ2_S",     LLAMA_FTYPE_MOSTLY_IQ2_S},
        {"IQ2_M",     LLAMA_FTYPE_MOSTLY_IQ2_M},
        {"IQ3_XXS",   LLAMA_FTYPE_MOSTLY_IQ3_XXS},
        {"IQ3_XS",    LLAMA_FTYPE_MOSTLY_IQ3_XS},
        {"IQ3_S",     LLAMA_FTYPE_MOSTLY_IQ3_S},
        {"IQ3_M",     LLAMA_FTYPE_MOSTLY_IQ3_M},
        {"IQ1_S",     LLAMA_FTYPE_MOSTLY_IQ1_S},
        {"IQ1_M",     LLAMA_FTYPE_MOSTLY_IQ1_M},
        {"IQ4_NL",    LLAMA_FTYPE_MOSTLY_IQ4_NL},
        {"IQ4_XS",    LLAMA_FTYPE_MOSTLY_IQ4_XS},
        {"TQ1_0",     LLAMA_FTYPE_MOSTLY_TQ1_0},
        {"TQ2_0",     LLAMA_FTYPE_MOSTLY_TQ2_0},
        {"MXFP4_MOE", LLAMA_FTYPE_MOSTLY_MXFP4_MOE},
        {"NVFP4",     LLAMA_FTYPE_MOSTLY_NVFP4},
        {"Q1_0",      LLAMA_FTYPE_MOSTLY_Q1_0},
    };
    return m;
}

// quantize(inputPath, outputPath, opts, done)
static Napi::Value Quantize(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();

    if (info.Length() < 4 ||
        !info[0].IsString() ||
        !info[1].IsString() ||
        !info[2].IsObject() ||
        !info[3].IsFunction()) {
        Napi::TypeError::New(env,
            "quantize(input: string, output: string, opts: object, done: fn)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string input  = info[0].As<Napi::String>().Utf8Value();
    std::string output = info[1].As<Napi::String>().Utf8Value();
    Napi::Object opts  = info[2].As<Napi::Object>();
    Napi::Function done = info[3].As<Napi::Function>();

    llama_model_quantize_params params = llama_model_quantize_default_params();

    // ftype — accepted as string name or raw enum value.
    if (opts.Has("ftype")) {
        Napi::Value ft = opts.Get("ftype");
        if (ft.IsNumber()) {
            params.ftype = (llama_ftype)ft.As<Napi::Number>().Int32Value();
        } else if (ft.IsString()) {
            std::string name = ft.As<Napi::String>().Utf8Value();
            const auto & m = ftype_map();
            auto it = m.find(name);
            if (it == m.end()) {
                Napi::Error::New(env, "Unknown ftype: " + name)
                    .ThrowAsJavaScriptException();
                return env.Undefined();
            }
            params.ftype = it->second;
        } else {
            Napi::TypeError::New(env, "ftype must be a string or number")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }
    }

    if (opts.Has("nthread") && opts.Get("nthread").IsNumber()) {
        params.nthread = opts.Get("nthread").As<Napi::Number>().Int32Value();
    }
    if (opts.Has("allowRequantize") && opts.Get("allowRequantize").IsBoolean()) {
        params.allow_requantize = opts.Get("allowRequantize").As<Napi::Boolean>().Value();
    }
    if (opts.Has("quantizeOutputTensor") && opts.Get("quantizeOutputTensor").IsBoolean()) {
        params.quantize_output_tensor = opts.Get("quantizeOutputTensor").As<Napi::Boolean>().Value();
    }
    if (opts.Has("onlyCopy") && opts.Get("onlyCopy").IsBoolean()) {
        params.only_copy = opts.Get("onlyCopy").As<Napi::Boolean>().Value();
    }
    if (opts.Has("pure") && opts.Get("pure").IsBoolean()) {
        params.pure = opts.Get("pure").As<Napi::Boolean>().Value();
    }
    if (opts.Has("keepSplit") && opts.Get("keepSplit").IsBoolean()) {
        params.keep_split = opts.Get("keepSplit").As<Napi::Boolean>().Value();
    }
    if (opts.Has("dryRun") && opts.Get("dryRun").IsBoolean()) {
        params.dry_run = opts.Get("dryRun").As<Napi::Boolean>().Value();
    }

    auto * worker = new QuantizeWorker(done, std::move(input), std::move(output), params);
    worker->Queue();

    return env.Undefined();
}

// quantizeFtypes() → string[] — names accepted by quantize().
static Napi::Value QuantizeFtypes(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    const auto & m = ftype_map();
    Napi::Array arr = Napi::Array::New(env, m.size());
    uint32_t i = 0;
    for (const auto & [name, _ft] : m) {
        (void)_ft;
        arr.Set(i++, Napi::String::New(env, name));
    }
    return arr;
}

void RegisterQuantize(Napi::Env env, Napi::Object exports) {
    exports.Set("quantize",        Napi::Function::New(env, Quantize,        "quantize"));
    exports.Set("quantizeFtypes",  Napi::Function::New(env, QuantizeFtypes,  "quantizeFtypes"));
}
