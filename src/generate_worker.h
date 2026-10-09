#pragma once

#include <napi.h>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llama_model.h"

struct TokenChunk {
    std::string text;
    // Logprob fields are populated only when GenerateWorker was constructed
    // with want_logprobs=true. With stop-sequence lookahead a single chunk
    // can span multiple sampled tokens; in that case `logprob` is the most
    // recent token's value (documented caveat).
    bool                                     has_logprobs = false;
    float                                    logprob      = 0.0f;
    std::vector<std::pair<std::string, float>> top_logprobs;
};

// All per-request generation parameters, snapshotted from the JS opts object
// on the main thread and moved into the worker. Defaults mirror llama.cpp's
// common_params_sampling where a sampler exists upstream; a default that
// means "disabled" keeps the sampler out of the chain entirely.
struct GenerateParams {
    std::string prompt;
    uint32_t    n_ctx         = 2048;
    bool        reset_context = false;
    int32_t     n_predict     = 256;

    // Probability filters / transforms (chain order documented in CLAUDE.md).
    float   temperature        = 0.8f;
    float   dynatemp_range     = 0.0f;   // > 0 switches temp → temp_ext (dynamic temperature)
    float   dynatemp_exponent  = 1.0f;
    float   top_p              = 0.95f;  // 1.0 = disabled
    int32_t top_k              = 40;     // <= 0 = disabled
    float   min_p              = 0.0f;   // 0.0 = disabled
    float   typical_p          = 1.0f;   // 1.0 = disabled
    float   xtc_probability    = 0.0f;   // 0.0 = disabled
    float   xtc_threshold      = 0.1f;
    float   top_n_sigma        = -1.0f;  // < 0 = disabled
    float   repeat_penalty     = 1.0f;   // 1.0 = disabled
    int32_t repeat_last_n      = 64;     // 0 = disabled
    float   dry_multiplier     = 0.0f;   // 0.0 = disabled
    float   dry_base           = 1.75f;
    int32_t dry_allowed_length = 2;
    int32_t dry_penalty_last_n = 64;     // 0 = disabled
    std::vector<std::string> dry_sequence_breakers{"\n", ":", "\"", "*"};

    // Mirostat replaces the filter stack entirely (mirrors upstream): the
    // chain becomes logit_bias → grammar → temp → mirostat.
    int32_t mirostat     = 0;      // 0 = off, 1 = v1, 2 = v2
    float   mirostat_tau = 5.0f;
    float   mirostat_eta = 0.1f;

    // adaptive-p replaces the final dist sampler when target >= 0.
    float adaptive_p_target = -1.0f;  // < 0 = disabled
    float adaptive_p_decay  = 0.90f;

    // -1 = use LLAMA_DEFAULT_SEED (non-deterministic). Seeds dist/xtc/
    // mirostat/adaptive-p.
    int64_t seed = -1;
    std::vector<std::pair<int32_t, float>> logit_bias;

    // Grammar. When either trigger list is non-empty the grammar is lazy.
    std::string              grammar;
    std::vector<std::string> grammar_trigger_patterns;
    std::vector<int32_t>     grammar_trigger_tokens;
    std::vector<std::string> preserved_tokens;  // reserved for future use

    std::vector<std::string> stop_sequences;

    bool    want_logprobs  = false;
    int32_t top_logprobs_n = 0;

    // Reasoning budget (libcommon's common_reasoning_budget sampler): caps
    // the number of tokens generated between a thinking start tag and any of
    // the end tags, force-closing the block (message + first end tag) when
    // the budget runs out. Disabled unless budget >= 0 and both tags are set.
    int32_t                  reasoning_budget = -1;
    std::string              reasoning_budget_start_tag;
    std::vector<std::string> reasoning_budget_end_tags;
    std::string              reasoning_budget_message;
    // Assistant-turn prefix already present in the prompt (e.g. a template
    // that pre-opens "<think>"). Fed to the budget sampler so its state
    // machine starts in the right place. From applyChatTemplateJinja's
    // generationPrompt.
    std::string              reasoning_budget_prefill;
};

// GenerateWorker runs llama_decode on a libuv worker thread and streams
// token text back to JS via AsyncProgressQueueWorker.
class GenerateWorker : public Napi::AsyncProgressQueueWorker<TokenChunk> {
public:
    GenerateWorker(
        Napi::Function &              token_cb,
        Napi::Function &              done_cb,
        LlamaModel *                  owner,
        std::shared_ptr<RequestState> state,
        GenerateParams                params
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
    GenerateParams                p_;
};
