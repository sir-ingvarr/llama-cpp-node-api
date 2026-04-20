#include "generate_worker.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

// Returns the index past the last *complete* UTF-8 sequence in s.
// Bytes in [result, s.size()) form an incomplete trailing sequence and
// must be carried to the next chunk.
static size_t complete_utf8_boundary(const std::string & s) {
    size_t n = s.size();
    if (n == 0) return 0;
    size_t i = n;
    while (i > 0) {
        unsigned char c = (unsigned char)s[--i];
        if ((c & 0x80) == 0)        return n;          // ASCII — complete
        if ((c & 0xC0) == 0x80)     continue;          // continuation byte
        // Leading byte: determine expected total length
        size_t expected;
        if      ((c & 0xE0) == 0xC0) expected = 2;
        else if ((c & 0xF0) == 0xE0) expected = 3;
        else if ((c & 0xF8) == 0xF0) expected = 4;
        else                          return n;         // invalid — emit as-is
        return (n - i >= expected) ? n : i;
    }
    return 0;
}

GenerateWorker::GenerateWorker(
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
)
    : Napi::AsyncProgressQueueWorker<TokenChunk>(token_cb.Env()),
      owner_(owner),
      state_(std::move(state)),
      prompt_(prompt),
      n_ctx_(n_ctx),
      reset_context_(reset_context),
      n_predict_(n_predict),
      temperature_(temperature),
      top_p_(top_p),
      top_k_(top_k),
      min_p_(min_p),
      repeat_penalty_(repeat_penalty),
      repeat_last_n_(repeat_last_n),
      grammar_str_(std::move(grammar_str)),
      stop_sequences_(std::move(stop_sequences)),
      grammar_trigger_patterns_(std::move(grammar_trigger_patterns)),
      grammar_trigger_tokens_(std::move(grammar_trigger_tokens)),
      preserved_tokens_(std::move(preserved_tokens))
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
    // Fast-reject if cancelled while queued, before contending for the ctx mutex.
    if (state_->cancel.load(std::memory_order_relaxed)) {
        return;
    }

    std::unique_lock<std::mutex> lock(owner_->ctx_mutex());

    // Another caller may have disposed the model while we waited.
    if (owner_->disposed()) {
        return;
    }
    // Recheck cancel after acquiring the mutex — may have been flipped while waiting.
    if (state_->cancel.load(std::memory_order_relaxed)) {
        return;
    }

    // Prepare the context now (on the worker thread) so the JS main thread
    // never blocks on ctx_mutex_. See LlamaModel::PrepareContextLocked.
    std::string ctx_err;
    if (!owner_->PrepareContextLocked(n_ctx_, reset_context_, ctx_err)) {
        SetError(ctx_err);
        return;
    }

    // Register ourselves as the active request so AbortCallback reads our cancel.
    owner_->set_active_request(state_);

    llama_context *     ctx   = owner_->ctx();
    const llama_vocab * vocab = owner_->vocab();

    const bool is_first =
        llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

    // Tokenize
    const int n_prompt_tokens_neg =
        -llama_tokenize(vocab, prompt_.c_str(), (int32_t)prompt_.size(),
                        nullptr, 0, is_first, true);
    if (n_prompt_tokens_neg <= 0) {
        owner_->clear_active_request();
        SetError("llama_tokenize: empty result");
        return;
    }

    std::vector<llama_token> prompt_tokens((size_t)n_prompt_tokens_neg);
    if (llama_tokenize(vocab, prompt_.c_str(), (int32_t)prompt_.size(),
                       prompt_tokens.data(), (int32_t)prompt_tokens.size(),
                       is_first, true) < 0) {
        owner_->clear_active_request();
        SetError("llama_tokenize: failed");
        return;
    }

    // Check context capacity
    const int n_ctx      = llama_n_ctx(ctx);
    const int n_ctx_used =
        (int)(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1);
    if (n_ctx_used + (int)prompt_tokens.size() > n_ctx) {
        owner_->clear_active_request();
        SetError("context size exceeded");
        return;
    }

    // Build sampler chain.
    // Grammar must come FIRST so it zeros out non-grammatical tokens from the
    // full logit distribution before top-k/top-p/min-p can discard them.
    // Order mirrors llama.cpp's common_sampler "grammar_first" path.
    llama_sampler * smpl =
        llama_sampler_chain_init(llama_sampler_chain_default_params());

    // 1. Grammar — constrain logits before any probability filtering.
    //    Triggers switch the sampler to "lazy" mode: grammar stays off until
    //    one of the patterns/tokens appears in the output, then activates for
    //    the remainder. Used e.g. for tool calls where the model writes prose
    //    first and only constrains once it emits <tool_call>.
    if (!grammar_str_.empty()) {
        const bool lazy =
            !grammar_trigger_patterns_.empty() || !grammar_trigger_tokens_.empty();
        if (lazy) {
            std::vector<const char *> pat_ptrs;
            pat_ptrs.reserve(grammar_trigger_patterns_.size());
            for (const auto & p : grammar_trigger_patterns_) {
                pat_ptrs.push_back(p.c_str());
            }
            std::vector<llama_token> tok_ids;
            tok_ids.reserve(grammar_trigger_tokens_.size());
            for (auto t : grammar_trigger_tokens_) {
                tok_ids.push_back((llama_token)t);
            }
            llama_sampler_chain_add(smpl,
                llama_sampler_init_grammar_lazy_patterns(
                    vocab, grammar_str_.c_str(), "root",
                    pat_ptrs.data(), pat_ptrs.size(),
                    tok_ids.data(),  tok_ids.size()));
        } else {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_grammar(vocab, grammar_str_.c_str(), "root"));
        }
    }
    // 2. Repeat penalty
    if (repeat_penalty_ != 1.0f && repeat_last_n_ != 0) {
        llama_sampler_chain_add(smpl,
            llama_sampler_init_penalties(repeat_last_n_, repeat_penalty_,
                                         0.0f, 0.0f));
    }
    // 3. Top-k
    if (top_k_ > 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k_));
    }
    // 4. Top-p
    if (top_p_ < 1.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p_, 1));
    }
    // 5. Min-p
    if (min_p_ > 0.0f) {
        llama_sampler_chain_add(smpl, llama_sampler_init_min_p(min_p_, 1));
    }
    // 6. Temperature
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature_));
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
    std::string utf8_carry;   // incomplete multi-byte sequence from previous chunk

    auto emit = [&](std::string text) {
        std::string full = utf8_carry + std::move(text);
        utf8_carry.clear();
        size_t boundary = complete_utf8_boundary(full);
        if (boundary < full.size()) {
            utf8_carry = full.substr(boundary);
            full.resize(boundary);
        }
        if (!full.empty()) {
            TokenChunk ch;
            ch.text = std::move(full);
            progress.Send(&ch, 1);
        }
    };

    while (true) {
        if (state_->cancel.load(std::memory_order_relaxed)) {
            break;
        }

        int ret = llama_decode(ctx, batch);
        if (ret == 2) { break; } // aborted via abort_callback
        if (ret != 0) {
            llama_sampler_free(smpl);
            owner_->clear_active_request();
            SetError(std::string("llama_decode failed, ret=") +
                     std::to_string(ret));
            return;
        }

        try {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);
        } catch (const std::runtime_error &) {
            // Grammar fully satisfied — stacks are empty, no further tokens
            // can be accepted. Treat as a clean end-of-generation.
            break;
        }

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            llama_sampler_free(smpl);
            owner_->clear_active_request();
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
                            emit(pending.substr(0, off));
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
                emit(pending.substr(0, safe));
                pending.erase(0, safe);
            }
        } else {
            // No stop sequences — emit immediately.
            emit(std::move(pending));
            pending.clear();
        }

        batch = llama_batch_get_one(&new_token_id, 1);
        ++n_decoded;

        if (n_predict_ > 0 && n_decoded >= n_predict_) {
            break;
        }
    }

    // Flush any text held back for stop-sequence lookahead that never matched.
    if (!pending.empty()) {
        emit(std::move(pending));
    }
    // Flush any incomplete UTF-8 sequence carried from the last chunk.
    if (!utf8_carry.empty()) {
        TokenChunk ch;
        ch.text = std::move(utf8_carry);
        progress.Send(&ch, 1);
    }

    llama_sampler_free(smpl);
    owner_->clear_active_request();
}

void GenerateWorker::OnProgress(const TokenChunk * chunks, size_t count) {
    Napi::Env env = token_cb_.Env();
    Napi::HandleScope scope(env);
    for (size_t i = 0; i < count; ++i) {
        const std::string & text = chunks[i].text;
        token_cb_.Call({Napi::Buffer<char>::Copy(env, text.data(), text.size())});
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
