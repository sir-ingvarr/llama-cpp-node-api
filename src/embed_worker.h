#pragma once

#include <napi.h>
#include <string>
#include <vector>

class LlamaModel;
struct AddonState;

// EmbedWorker runs llama_decode in embedding mode on a libuv worker thread
// and copies out the pooled (or last-token) embedding for the input string.
// Acquires `LlamaModel::ctx_mutex_` so it serialises with any concurrent
// generate() — but a model in embedding mode rejects generate() anyway, so
// in practice the contention is just with other concurrent embed() calls.
class EmbedWorker : public Napi::AsyncWorker {
public:
    EmbedWorker(Napi::Function & done_cb,
                LlamaModel *     owner,
                std::string      text);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error & error) override;

private:
    Napi::FunctionReference  done_cb_;
    LlamaModel *             owner_;
    std::string              text_;
    AddonState *             state_     = nullptr;
    std::vector<float>       embedding_;
};
