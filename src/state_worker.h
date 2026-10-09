#pragma once

#include <napi.h>
#include <cstdint>
#include <string>
#include <vector>

#include "llama_model.h"

// SaveStateWorker / LoadStateWorker wrap llama_state_save_file /
// llama_state_load_file on a libuv worker thread, serialized with
// generate()/embed() on LlamaModel::ctx_mutex_.
//
// The optional token list is llama.cpp's session-token convention: the
// prompt tokens that produced the saved KV cache, stored in the file and
// returned on load so the caller can skip re-prefilling. The wrapper does
// not track prompt tokens natively, so the caller supplies them (usually
// via model.tokenize()) — an empty list is valid and simply restores the
// KV cache without the token record.
//
// State files are NOT portable across llama.cpp versions, models, or
// context configurations (upstream bumps LLAMA_SESSION_VERSION).

class SaveStateWorker : public Napi::AsyncWorker {
public:
    SaveStateWorker(Napi::Function &          done_cb,
                    LlamaModel *              owner,
                    std::string               path,
                    std::vector<llama_token>  tokens);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error & error) override;

private:
    Napi::FunctionReference  done_cb_;
    LlamaModel *             owner_;
    std::string              path_;
    std::vector<llama_token> tokens_;
};

class LoadStateWorker : public Napi::AsyncWorker {
public:
    LoadStateWorker(Napi::Function & done_cb,
                    LlamaModel *     owner,
                    std::string      path);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error & error) override;

private:
    Napi::FunctionReference  done_cb_;
    LlamaModel *             owner_;
    std::string              path_;
    std::vector<llama_token> tokens_out_;
};
