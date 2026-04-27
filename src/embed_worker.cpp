#include "embed_worker.h"
#include "addon_state.h"
#include "llama_model.h"

#include <cstring>
#include <memory>

EmbedWorker::EmbedWorker(Napi::Function & done_cb,
                         LlamaModel *     owner,
                         std::string      text)
    : Napi::AsyncWorker(done_cb.Env()),
      owner_(owner),
      text_(std::move(text))
{
    done_cb_ = Napi::Persistent(done_cb);
    state_   = done_cb.Env().GetInstanceData<AddonState>();
}

void EmbedWorker::Execute() {
    WorkerGuard guard(state_);
    if (guard.shutting_down()) {
        SetError("env is shutting down");
        return;
    }

    std::unique_lock<std::mutex> lock(owner_->ctx_mutex());
    if (owner_->disposed()) {
        SetError("model is disposed");
        return;
    }

    // Ensure context with the model's configured embedding settings; reset
    // KV between embed calls so each input is independent.
    std::string ctx_err;
    if (!owner_->PrepareContextLocked(owner_->n_ctx_default(),
                                      /*reset=*/true, ctx_err)) {
        SetError(ctx_err);
        return;
    }

    llama_context *     ctx   = owner_->ctx();
    const llama_vocab * vocab = owner_->vocab();
    llama_model *       model = owner_->model();

    // Tokenize. add_special=true picks up BOS/CLS as the model expects.
    int32_t n_neg = -llama_tokenize(vocab, text_.c_str(), (int32_t) text_.size(),
                                    nullptr, 0, /*add_special=*/true,
                                    /*parse_special=*/true);
    if (n_neg <= 0) {
        SetError("llama_tokenize: empty result");
        return;
    }
    std::vector<llama_token> tokens((size_t) n_neg);
    if (llama_tokenize(vocab, text_.c_str(), (int32_t) text_.size(),
                       tokens.data(), (int32_t) tokens.size(),
                       /*add_special=*/true, /*parse_special=*/true) < 0) {
        SetError("llama_tokenize: failed");
        return;
    }
    if ((uint32_t) tokens.size() > owner_->n_ctx_default()) {
        SetError("input exceeds context size (" +
                 std::to_string(tokens.size()) + " > " +
                 std::to_string(owner_->n_ctx_default()) + ")");
        return;
    }

    // Build a single-sequence batch covering the whole prompt. The pooled
    // embedding API does not require per-token logits; for POOLING_TYPE_NONE
    // we ask for logits on the last token so its embedding is retrievable.
    using BatchPtr = std::unique_ptr<llama_batch, void(*)(llama_batch *)>;
    BatchPtr batch{
        new llama_batch(llama_batch_init((int32_t) tokens.size(), 0, 1)),
        [](llama_batch * b) { llama_batch_free(*b); delete b; }
    };
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch->token[i]    = tokens[i];
        batch->pos[i]      = (llama_pos) i;
        batch->n_seq_id[i] = 1;
        batch->seq_id[i][0] = 0;
        batch->logits[i]   = (i == tokens.size() - 1);
    }
    batch->n_tokens = (int32_t) tokens.size();

    if (llama_decode(ctx, *batch) != 0) {
        SetError("llama_decode failed in embedding mode");
        return;
    }

    const int32_t n_embd = llama_model_n_embd(model);
    const enum llama_pooling_type pooling = llama_pooling_type(ctx);

    const float * emb = nullptr;
    if (pooling == LLAMA_POOLING_TYPE_NONE) {
        emb = llama_get_embeddings_ith(ctx, batch->n_tokens - 1);
    } else {
        emb = llama_get_embeddings_seq(ctx, /*seq_id=*/0);
    }
    if (!emb) {
        SetError("llama_get_embeddings returned null (model may not support embeddings)");
        return;
    }

    embedding_.assign(emb, emb + n_embd);
}

void EmbedWorker::OnOK() {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);

    // Hand back as Float32Array so the caller doesn't pay a per-element
    // double-conversion cost for typical n_embd in the 384–4096 range.
    auto buf = Napi::ArrayBuffer::New(env, embedding_.size() * sizeof(float));
    std::memcpy(buf.Data(), embedding_.data(), embedding_.size() * sizeof(float));
    auto arr = Napi::Float32Array::New(env, embedding_.size(), buf, 0);

    done_cb_.Call({env.Null(), arr});
}

void EmbedWorker::OnError(const Napi::Error & error) {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    done_cb_.Call({error.Value()});
}
