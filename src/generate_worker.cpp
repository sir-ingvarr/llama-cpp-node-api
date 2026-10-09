#include "generate_worker.h"
#include "addon_state.h"

#include "reasoning-budget.h"  // libcommon: common_reasoning_budget_init

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

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

// Tokenize a tag/message with add_special=false, parse_special=true
// (equivalent to libcommon's common_tokenize for special sequences).
static std::vector<llama_token> tokenize_special(
    const llama_vocab * vocab, const std::string & text)
{
    if (text.empty()) return {};
    int32_t n = -llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                nullptr, 0, false, true);
    if (n <= 0) return {};
    std::vector<llama_token> toks((size_t)n);
    if (llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                       toks.data(), (int32_t)toks.size(), false, true) < 0) {
        return {};
    }
    return toks;
}

GenerateWorker::GenerateWorker(
    Napi::Function &              token_cb,
    Napi::Function &              done_cb,
    LlamaModel *                  owner,
    std::shared_ptr<RequestState> state,
    GenerateParams                params
)
    : Napi::AsyncProgressQueueWorker<TokenChunk>(token_cb.Env()),
      owner_(owner),
      state_(std::move(state)),
      p_(std::move(params))
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
    WorkerGuard guard(owner_->addon_state());
    // Fast-reject if cancelled while queued, before contending for the ctx mutex.
    if (state_->cancel.load(std::memory_order_relaxed) || guard.shutting_down()) {
        return;
    }

    std::unique_lock<std::mutex> lock(owner_->ctx_mutex());

    // Another caller may have disposed the model while we waited.
    if (owner_->disposed()) {
        return;
    }
    // Recheck cancel after acquiring the mutex — may have been flipped while waiting.
    if (state_->cancel.load(std::memory_order_relaxed) || guard.shutting_down()) {
        return;
    }

    // Prepare the context now (on the worker thread) so the JS main thread
    // never blocks on ctx_mutex_. See LlamaModel::PrepareContextLocked.
    std::string ctx_err;
    if (!owner_->PrepareContextLocked(p_.n_ctx, p_.reset_context, ctx_err)) {
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
        -llama_tokenize(vocab, p_.prompt.c_str(), (int32_t)p_.prompt.size(),
                        nullptr, 0, is_first, true);
    if (n_prompt_tokens_neg <= 0) {
        owner_->clear_active_request();
        SetError("llama_tokenize: empty result");
        return;
    }

    std::vector<llama_token> prompt_tokens((size_t)n_prompt_tokens_neg);
    if (llama_tokenize(vocab, p_.prompt.c_str(), (int32_t)p_.prompt.size(),
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
    // Grammar must come before the probability filters so it zeros out
    // non-grammatical tokens from the full logit distribution before
    // top-k/top-p/min-p can discard them. Order mirrors llama.cpp's
    // common_sampler "grammar_first" path.
    using SamplerPtr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;
    SamplerPtr smpl_owner{
        llama_sampler_chain_init(llama_sampler_chain_default_params()),
        &llama_sampler_free
    };
    llama_sampler * smpl = smpl_owner.get();

    const uint32_t seed = (p_.seed < 0) ? LLAMA_DEFAULT_SEED : (uint32_t) p_.seed;

    // 0. Reasoning budget — first in the chain so it can force the end tag by
    //    masking everything else before any other sampler runs. The budget
    //    sampler is a passthrough outside the thinking block. Combining it
    //    with an *eager* grammar is unsupported (the grammar could mask the
    //    forced end-tag tokens); lazy tool-call grammars trigger after the
    //    thinking block closes and are fine.
    if (p_.reasoning_budget >= 0 &&
        !p_.reasoning_budget_start_tag.empty() &&
        !p_.reasoning_budget_end_tags.empty()) {
        const llama_tokens start = tokenize_special(vocab, p_.reasoning_budget_start_tag);
        std::vector<llama_tokens> ends;
        ends.reserve(p_.reasoning_budget_end_tags.size());
        for (const auto & tag : p_.reasoning_budget_end_tags) {
            llama_tokens e = tokenize_special(vocab, tag);
            if (!e.empty()) ends.push_back(std::move(e));
        }
        if (!start.empty() && !ends.empty()) {
            llama_tokens forced = ends.front();
            if (!p_.reasoning_budget_message.empty()) {
                const llama_tokens msg =
                    tokenize_special(vocab, p_.reasoning_budget_message);
                forced.insert(forced.begin(), msg.begin(), msg.end());
            }
            llama_sampler * rbudget = common_reasoning_budget_init(
                vocab, {start}, ends, forced, p_.reasoning_budget);

            // Feed the assistant-turn prefix already present in the prompt
            // (e.g. a template that pre-opens "<think>") so the state machine
            // starts counting if the block is already open. Mirrors
            // common_sampler_init's prefill handling, including the skip of a
            // tokenizer-injected leading space.
            if (!p_.reasoning_budget_prefill.empty()) {
                const llama_tokens prefill =
                    tokenize_special(vocab, p_.reasoning_budget_prefill);
                for (size_t i = 0; i < prefill.size(); ++i) {
                    if (i == 0) {
                        char pbuf[64];
                        int pn = llama_token_to_piece(vocab, prefill[0],
                                                      pbuf, sizeof(pbuf), 0, true);
                        if (pn > 0 && std::isspace((unsigned char)pbuf[0]) &&
                            !std::isspace((unsigned char)p_.reasoning_budget_prefill[0])) {
                            continue;
                        }
                    }
                    llama_sampler_accept(rbudget, prefill[i]);
                }
            }
            llama_sampler_chain_add(smpl, rbudget);
        }
    }

    // 1. Logit bias — additive on raw logits, applied before grammar so that
    //    a positive bias on a grammar-disallowed token cannot resurrect it
    //    (grammar masks to -inf afterward and has the final say).
    if (!p_.logit_bias.empty()) {
        std::vector<llama_logit_bias> biases;
        biases.reserve(p_.logit_bias.size());
        for (const auto & [tok, b] : p_.logit_bias) {
            biases.push_back({ (llama_token) tok, b });
        }
        llama_sampler_chain_add(smpl,
            llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                (int32_t) biases.size(),
                biases.data()));
    }

    // 2. Grammar — constrain logits before any probability filtering.
    //    Triggers switch the sampler to "lazy" mode: grammar stays off until
    //    one of the patterns/tokens appears in the output, then activates for
    //    the remainder. Used e.g. for tool calls where the model writes prose
    //    first and only constrains once it emits <tool_call>.
    if (!p_.grammar.empty()) {
        const bool lazy =
            !p_.grammar_trigger_patterns.empty() || !p_.grammar_trigger_tokens.empty();
        if (lazy) {
            std::vector<const char *> pat_ptrs;
            pat_ptrs.reserve(p_.grammar_trigger_patterns.size());
            for (const auto & p : p_.grammar_trigger_patterns) {
                pat_ptrs.push_back(p.c_str());
            }
            std::vector<llama_token> tok_ids;
            tok_ids.reserve(p_.grammar_trigger_tokens.size());
            for (auto t : p_.grammar_trigger_tokens) {
                tok_ids.push_back((llama_token)t);
            }
            llama_sampler_chain_add(smpl,
                llama_sampler_init_grammar_lazy_patterns(
                    vocab, p_.grammar.c_str(), "root",
                    pat_ptrs.data(), pat_ptrs.size(),
                    tok_ids.data(),  tok_ids.size()));
        } else {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_grammar(vocab, p_.grammar.c_str(), "root"));
        }
    }

    if (p_.mirostat == 0) {
        // 3. Repeat penalty
        if (p_.repeat_penalty != 1.0f && p_.repeat_last_n != 0) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_penalties(llama_vocab_n_tokens(vocab),
                                             p_.repeat_last_n, p_.repeat_penalty,
                                             0.0f, 0.0f));
        }
        // 4. DRY (don't-repeat-yourself) repetition penalty
        if (p_.dry_multiplier > 0.0f) {
            std::vector<const char *> breakers;
            breakers.reserve(p_.dry_sequence_breakers.size());
            for (const auto & b : p_.dry_sequence_breakers) {
                breakers.push_back(b.c_str());
            }
            llama_sampler_chain_add(smpl,
                llama_sampler_init_dry(vocab,
                                       p_.dry_multiplier, p_.dry_base,
                                       p_.dry_allowed_length, p_.dry_penalty_last_n,
                                       breakers.data(), breakers.size()));
        }
        // 5. Top-n-sigma
        if (p_.top_n_sigma >= 0.0f) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_top_n_sigma(p_.top_n_sigma));
        }
        // 6. Top-k
        if (p_.top_k > 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(p_.top_k));
        }
        // 7. Typical-p (locally typical sampling)
        if (p_.typical_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_typical(p_.typical_p, 1));
        }
        // 8. Top-p
        if (p_.top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(p_.top_p, 1));
        }
        // 9. Min-p
        if (p_.min_p > 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(p_.min_p, 1));
        }
        // 10. XTC (exclude top choices)
        if (p_.xtc_probability > 0.0f) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_xtc(p_.xtc_probability, p_.xtc_threshold, 1, seed));
        }
        // 11. Temperature — temp_ext (dynamic temperature / entropy scaling)
        //     when a dynatemp range is set, plain temp otherwise.
        if (p_.dynatemp_range > 0.0f) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_temp_ext(p_.temperature, p_.dynatemp_range,
                                            p_.dynatemp_exponent));
        } else {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(p_.temperature));
        }
        // 12. Final token selector — adaptive-p when a target is configured,
        //     plain dist otherwise.
        if (p_.adaptive_p_target >= 0.0f) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_adaptive_p(p_.adaptive_p_target,
                                              p_.adaptive_p_decay, seed));
        } else {
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
        }
    } else {
        // Mirostat replaces the filter stack (mirrors upstream behaviour):
        // temp → mirostat, which selects the token itself.
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(p_.temperature));
        if (p_.mirostat == 1) {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_mirostat(llama_vocab_n_tokens(vocab), seed,
                                            p_.mirostat_tau, p_.mirostat_eta, 100));
        } else {
            llama_sampler_chain_add(smpl,
                llama_sampler_init_mirostat_v2(seed, p_.mirostat_tau, p_.mirostat_eta));
        }
    }

    llama_batch batch =
        llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
    llama_token new_token_id = 0;
    int32_t n_decoded = 0;

    // Buffer for stop-sequence lookahead. Characters here have been decoded
    // but not yet forwarded to JS; they are flushed once we know they are
    // not the beginning of a stop sequence.
    std::string pending;
    std::string utf8_carry;   // incomplete multi-byte sequence from previous chunk

    // Per-token logprob snapshot, attached to whatever chunks emit() flushes
    // before the next sample. With stop-sequence lookahead a chunk can span
    // multiple sampled tokens; in that case only the most recent token's
    // logprob rides along (documented contract).
    float                                       cur_logprob = 0.0f;
    std::vector<std::pair<std::string, float>>  cur_top;
    // Scratch for top-K selection, reused across tokens. Declared outside the
    // decode loop and clear()'d (not reallocated) each token so we don't churn
    // a vocab-sized buffer (~1 MB at 128k vocab) on every sampled token.
    std::vector<std::pair<float, int>>          logprob_scratch;

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
            if (p_.want_logprobs) {
                ch.has_logprobs = true;
                ch.logprob      = cur_logprob;
                ch.top_logprobs = cur_top;
            }
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

        // Compute logprobs for this token from the raw model logits BEFORE
        // any sampler-stage filtering (so the user sees the model's
        // distribution, not the post-grammar/penalty distribution).
        if (p_.want_logprobs) {
            const float * logits  = llama_get_logits_ith(ctx, -1);
            const int     n_vocab = llama_vocab_n_tokens(vocab);
            float max_logit = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < n_vocab; ++i) {
                if (logits[i] > max_logit) max_logit = logits[i];
            }
            double sum = 0.0;
            for (int i = 0; i < n_vocab; ++i) {
                sum += std::exp((double)(logits[i] - max_logit));
            }
            const float lse = max_logit + (float) std::log(sum);
            cur_logprob = logits[new_token_id] - lse;

            cur_top.clear();
            if (p_.top_logprobs_n > 0) {
                auto & idx = logprob_scratch;
                idx.clear();                       // retains capacity
                idx.reserve((size_t) n_vocab);     // no-op after the first token
                for (int i = 0; i < n_vocab; ++i) idx.emplace_back(logits[i], i);
                const int K = std::min(p_.top_logprobs_n, n_vocab);
                std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                    [](const std::pair<float,int> & a,
                       const std::pair<float,int> & b) { return a.first > b.first; });
                cur_top.reserve((size_t) K);
                for (int i = 0; i < K; ++i) {
                    char tbuf[256];
                    int tn = llama_token_to_piece(vocab, idx[i].second,
                                                  tbuf, sizeof(tbuf), 0, true);
                    std::string ttext = (tn > 0) ? std::string(tbuf, (size_t) tn)
                                                 : std::string();
                    cur_top.emplace_back(std::move(ttext), idx[i].first - lse);
                }
            }
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            owner_->clear_active_request();
            SetError("llama_token_to_piece failed");
            return;
        }

        pending.append(buf, (size_t)n);

        // --- Stop sequence detection ---
        if (!p_.stop_sequences.empty()) {
            // Search the whole pending buffer, not just its tail: a single
            // sampled token can contain a stop sequence mid-piece (e.g. a
            // "\n\n" token when stopping on "\n"), or even trailing text
            // after it. Take the earliest match across all stop sequences.
            size_t stop_pos = std::string::npos;
            for (const auto & stop : p_.stop_sequences) {
                if (stop.empty()) continue;
                size_t pos = pending.find(stop);
                if (pos < stop_pos) stop_pos = pos;
            }
            if (stop_pos != std::string::npos) {
                // Flush everything before the stop sequence, drop the stop
                // itself and anything after it, then halt.
                if (stop_pos > 0) {
                    emit(pending.substr(0, stop_pos));
                }
                pending.clear();
                break;
            }

            // Flush only the portion that cannot be the start of a stop sequence.
            size_t hold = trailing_stop_prefix(pending, p_.stop_sequences);
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

        if (p_.n_predict > 0 && n_decoded >= p_.n_predict) {
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

    owner_->clear_active_request();
    // smpl_owner releases the sampler chain on scope exit.
}

void GenerateWorker::OnProgress(const TokenChunk * chunks, size_t count) {
    Napi::Env env = token_cb_.Env();
    Napi::HandleScope scope(env);
    // Token text is guaranteed-valid UTF-8 by complete_utf8_boundary(), so we
    // hand it to V8 as a String — V8 can short-string-optimise and the JS side
    // doesn't need to decode a Buffer per token. If the JS callback throws,
    // bail out of the loop: with NAPI_DISABLE_CPP_EXCEPTIONS a pending
    // exception silently neutralises subsequent N-API calls.
    for (size_t i = 0; i < count; ++i) {
        const TokenChunk & ck = chunks[i];
        Napi::String text = Napi::String::New(env, ck.text);
        if (!ck.has_logprobs) {
            token_cb_.Call({text});
        } else {
            // cb(text, logprob, topLogprobs) — JS layer assembles the object.
            Napi::Number lp = Napi::Number::New(env, (double) ck.logprob);
            Napi::Array  top = Napi::Array::New(env, ck.top_logprobs.size());
            for (size_t j = 0; j < ck.top_logprobs.size(); ++j) {
                Napi::Object e = Napi::Object::New(env);
                e.Set("token",   Napi::String::New(env, ck.top_logprobs[j].first));
                e.Set("logprob", Napi::Number::New(env, (double) ck.top_logprobs[j].second));
                top.Set((uint32_t) j, e);
            }
            token_cb_.Call({text, lp, top});
        }
        if (env.IsExceptionPending()) return;
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
