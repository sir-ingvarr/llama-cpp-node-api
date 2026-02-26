#pragma once

#include <napi.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

#include "llama.h"

struct TokenChunk {
    std::string text;
};

// Forward declaration
class LlamaModel;

// GenerateWorker runs llama_decode on a libuv worker thread and streams
// token text back to JS via AsyncProgressQueueWorker.
class GenerateWorker : public Napi::AsyncProgressQueueWorker<TokenChunk> {
public:
    GenerateWorker(
        Napi::Function &         token_cb,
        Napi::Function &         done_cb,
        llama_context *          ctx,
        const llama_vocab *      vocab,
        std::mutex &             ctx_mutex,
        std::atomic<bool> &      cancel,
        const std::string &      prompt,
        int32_t                  n_predict,
        float                    temperature,
        float                    top_p,
        int32_t                  top_k,
        float                    min_p,
        float                    repeat_penalty,
        int32_t                  repeat_last_n,
        std::string              grammar_str,
        std::vector<std::string> stop_sequences
    );

    void Execute(const ExecutionProgress & progress) override;
    void OnProgress(const TokenChunk * chunks, size_t count) override;
    void OnOK() override;
    void OnError(const Napi::Error & error) override;

private:
    Napi::FunctionReference  token_cb_;
    Napi::FunctionReference  done_cb_;
    llama_context *          ctx_;
    const llama_vocab *      vocab_;
    std::mutex &             ctx_mutex_;
    std::atomic<bool> &      cancel_;
    std::string              prompt_;
    int32_t                  n_predict_;
    float                    temperature_;
    float                    top_p_;
    int32_t                  top_k_;
    float                    min_p_;
    float                    repeat_penalty_;
    int32_t                  repeat_last_n_;
    std::string              grammar_str_;
    std::vector<std::string> stop_sequences_;
};
