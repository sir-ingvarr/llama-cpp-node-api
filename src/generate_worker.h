#pragma once

#include <napi.h>
#include <memory>
#include <string>
#include <vector>

#include "llama_model.h"

struct TokenChunk {
    std::string text;
};

// GenerateWorker runs llama_decode on a libuv worker thread and streams
// token text back to JS via AsyncProgressQueueWorker.
class GenerateWorker : public Napi::AsyncProgressQueueWorker<TokenChunk> {
public:
    GenerateWorker(
        Napi::Function &                  token_cb,
        Napi::Function &                  done_cb,
        LlamaModel *                      owner,
        std::shared_ptr<RequestState>     state,
        const std::string &               prompt,
        uint32_t                          n_ctx,
        bool                              reset_context,
        int32_t                           n_predict,
        float                             temperature,
        float                             top_p,
        int32_t                           top_k,
        float                             min_p,
        float                             repeat_penalty,
        int32_t                           repeat_last_n,
        std::string                       grammar_str,
        std::vector<std::string>          stop_sequences,
        std::vector<std::string>          grammar_trigger_patterns,
        std::vector<int32_t>              grammar_trigger_tokens,
        std::vector<std::string>          preserved_tokens
    );

    void Execute(const ExecutionProgress & progress) override;
    void OnProgress(const TokenChunk * chunks, size_t count) override;
    void OnOK() override;
    void OnError(const Napi::Error & error) override;

private:
    Napi::FunctionReference       token_cb_;
    Napi::FunctionReference       done_cb_;
    LlamaModel *                  owner_;
    std::shared_ptr<RequestState> state_;
    std::string                   prompt_;
    uint32_t                      n_ctx_;
    bool                          reset_context_;
    int32_t                       n_predict_;
    float                         temperature_;
    float                         top_p_;
    int32_t                       top_k_;
    float                         min_p_;
    float                         repeat_penalty_;
    int32_t                       repeat_last_n_;
    std::string                   grammar_str_;
    std::vector<std::string>      stop_sequences_;
    // Lazy-grammar inputs — when grammar_trigger_patterns_ and
    // grammar_trigger_tokens_ are both empty, grammar is applied eagerly.
    std::vector<std::string>      grammar_trigger_patterns_;
    std::vector<int32_t>          grammar_trigger_tokens_;
    std::vector<std::string>      preserved_tokens_;  // reserved for future use
};
