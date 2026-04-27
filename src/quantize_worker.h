#pragma once

#include <napi.h>
#include <string>

#include "llama.h"

struct AddonState;

// QuantizeWorker runs llama_model_quantize on a libuv worker thread.
// Wraps the full input→output file conversion; there is no streaming
// progress (llama.cpp prints its own progress to stderr).
class QuantizeWorker : public Napi::AsyncWorker {
public:
    QuantizeWorker(
        Napi::Function &                 done_cb,
        std::string                      input_path,
        std::string                      output_path,
        llama_model_quantize_params      params
    );

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error & error) override;

private:
    Napi::FunctionReference     done_cb_;
    std::string                 input_path_;
    std::string                 output_path_;
    llama_model_quantize_params params_;
    AddonState *                state_ = nullptr;
};

// Registers the `quantize(input, output, opts, done)` function on `exports`.
void RegisterQuantize(Napi::Env env, Napi::Object exports);
