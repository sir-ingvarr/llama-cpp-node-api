#include "generate_worker.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

GenerateWorker::GenerateWorker(
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
)
    : Napi::AsyncProgressQueueWorker<TokenChunk>(token_cb.Env()),
      ctx_(ctx),
      vocab_(vocab),
      ctx_mutex_(ctx_mutex),
      cancel_(cancel),
      prompt_(prompt),
      n_predict_(n_predict),
      temperature_(temperature),
      top_p_(top_p),
      top_k_(top_k),
      min_p_(min_p),
      repeat_penalty_(repeat_penalty),
      repeat_last_n_(repeat_last_n),
      grammar_str_(std::move(grammar_str)),
      stop_sequences_(std::move(stop_sequences))
{
    token_cb_ = Napi::Persistent(token_cb);
    done_cb_  = Napi::Persistent(done_cb);
}

// ---------------------------------------------------------------------------
// Stop-sequence helpers
// ---------------------------------------------------------------------------

// Returns how many trailing characters of `text` form a prefix of any stop
// sequence (i.e. we must hold them back because they might be the start of a
// stop sequence that the next token will complete).
static size_t trailing_stop_prefix(
    const std::string & text,
    const std::vector<std::string> & stops)
{
    size_t hold = 0;
    for (const auto & stop : stops) {
        if (stop.empty()) continue;
        size_t max_pfx = std::min(stop.size() - 1, text.size());
        for (size_t plen = max_pfx; plen >= 1; --plen) {
            if (text.compare(text.size() - plen, plen, stop, 0, plen) == 0) {
                hold = std::max(hold, plen);
                break;
            }
        }
    }
    return hold;
}

// ---------------------------------------------------------------------------
// Execute — runs on a libuv worker thread
// ---------------------------------------------------------------------------

void GenerateWorker::Execute(const ExecutionProgress & progress) {
    std::unique_lock<std::mutex> lock(ctx_mutex_);

    const bool is_first =
        llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) == -1;

    // Tokenize
    const int n_prompt_tokens_neg =
        -llama_tokenize(vocab_, prompt_.c_str(), (int32_t)prompt_.size(),
                        nullptr, 0, is_first, true);
    if (n_prompt_tokens_neg <= 0) {
        SetError("llama_tokenize: empty result");
        return;
    }

    std::vector<llama_token> prompt_tokens((size_t)n_prompt_tokens_neg);
    if (llama_tokenize(vocab_, prompt_.c_str(), (int32_t)prompt_.size(),
                       prompt_tokens.data(), (int32_t)prompt_tokens.size(),
                       is_first, true) < 0) {
        SetError("llama_tokenize: failed");
        return;
    }

    // Check context capacity
    const int n_ctx      = llama_n_ctx(ctx_);
    const int n_ctx_used =
        (int)(llama_memory_seq_pos_max(llama_get_memory(ctx_), 0) + 1);
    if (n_ctx_used + (int)prompt_tokens.size() > n_ctx) {
        SetError("context size exceeded");
        return;
    }

    // Build sampler chain (order mirrors llama.cpp's common_sampler)
    llama_sampler * smpl =
        llama_sampler_chain_init(llama_sampler_chain_default_params());

    // 1. Repeat penalty
    if (repeat_penalty_ != 1.0f && repeat_last_n_ != 0) {
        llama_sampler_chain_add(smpl,
            llama_sampler_init_penalties(repeat_last_n_, repeat_penalty_,
                                         0.0f, 0.0f));
    }
    // 2. Top-k
    if (top_k_ > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k_));
    }
    // 3. Top-p
    if (top_p_ < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p_, 1));
    }
    // 4. Min-p
    if (min_p_ > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p_, 1));
    }
    // 5. Temperature
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature_));
    // 6. Grammar (constrain the distribution before final sampling)
    if (!grammar_str_.empty()) {
        llama_sampler_chain_add(smpl,
            llama_sampler_init_grammar(vocab_, grammar_str_.c_str(), "root"));
    }
    // 7. Final stochastic sampler
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    llama_batch batch =
        llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
    llama_token new_token_id = 0;
    int32_t n_decoded = 0;

    // Buffer for stop-sequence lookahead. Characters here have been decoded
    // but not yet forwarded to JS; they are flushed once we know they are
    // not the beginning of a stop sequence.
    std::string pending;

    while (true) {
        if (cancel_.load(std::memory_order_relaxed)) {
            break;
        }

        int ret = llama_decode(ctx_, batch);
        if (ret == 2) { break; } // aborted via abort_callback
        if (ret != 0) {
            llama_sampler_free(smpl);
            SetError(std::string("llama_decode failed, ret=") +
                     std::to_string(ret));
            return;
        }

        new_token_id = llama_sampler_sample(smpl, ctx_, -1);

        if (llama_vocab_is_eog(vocab_, new_token_id)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab_, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            llama_sampler_free(smpl);
            SetError("llama_token_to_piece failed");
            return;
        }

        pending.append(buf, (size_t)n);

        // --- Stop sequence detection ---
        if (!stop_sequences_.empty()) {
            bool stopped = false;
            for (const auto & stop : stop_sequences_) {
                if (stop.empty()) continue;
                if (pending.size() >= stop.size()) {
                    size_t off = pending.size() - stop.size();
                    if (pending.compare(off, stop.size(), stop) == 0) {
                        // Flush everything before the stop sequence, then halt.
                        if (off > 0) {
                            TokenChunk ch;
                            ch.text = pending.substr(0, off);
                            progress.Send(&ch, 1);
                        }
                        pending.clear();
                        stopped = true;
                        break;
                    }
                }
            }
            if (stopped) break;

            // Flush only the portion that cannot be the start of a stop sequence.
            size_t hold = trailing_stop_prefix(pending, stop_sequences_);
            size_t safe  = pending.size() - hold;
            if (safe > 0) {
                TokenChunk ch;
                ch.text = pending.substr(0, safe);
                progress.Send(&ch, 1);
                pending.erase(0, safe);
            }
        } else {
            // No stop sequences — emit immediately.
            TokenChunk chunk;
            chunk.text = std::move(pending);
            pending.clear();
            progress.Send(&chunk, 1);
        }

        batch = llama_batch_get_one(&new_token_id, 1);
        ++n_decoded;

        if (n_predict_ > 0 && n_decoded >= n_predict_) {
            break;
        }
    }

    // Flush any text held back for stop-sequence lookahead that never matched.
    if (!pending.empty()) {
        TokenChunk ch;
        ch.text = std::move(pending);
        progress.Send(&ch, 1);
    }

    llama_sampler_free(smpl);
}

void GenerateWorker::OnProgress(const TokenChunk * chunks, size_t count) {
    Napi::Env env = token_cb_.Env();
    Napi::HandleScope scope(env);
    for (size_t i = 0; i < count; ++i) {
        token_cb_.Call({Napi::String::New(env, chunks[i].text)});
    }
}

void GenerateWorker::OnOK() {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    done_cb_.Call({env.Null()});
}

void GenerateWorker::OnError(const Napi::Error & error) {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    done_cb_.Call({error.Value()});
}
