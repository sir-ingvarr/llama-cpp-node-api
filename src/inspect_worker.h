#pragma once

#include <napi.h>
#include <string>

struct AddonState;

// InspectWorker reads GGUF metadata via gguf_init_from_file with no_alloc=true.
// No tensor weights are loaded; only the header, KV pairs, and tensor
// descriptors. Runs on a libuv worker thread because file I/O may block.
class InspectWorker : public Napi::AsyncWorker {
public:
    InspectWorker(Napi::Function & done_cb, std::string path);

    void Execute() override;
    void OnOK() override;
    void OnError(const Napi::Error & error) override;

private:
    Napi::FunctionReference done_cb_;
    std::string             path_;
    AddonState *            state_ = nullptr;

    // Captured on the worker thread, materialised into JS on OnOK().
    struct KV {
        std::string name;
        // Discriminator + parsed value live in the heterogeneous payload below.
        // We pre-build a small AST-like representation so OnOK only marshals.
        // For arrays of scalars we keep raw bytes + element type + count.
        int                       type = 0;          // gguf_type
        int                       arr_elem_type = 0; // valid when type == ARRAY
        bool                      b = false;
        int64_t                   i = 0;
        uint64_t                  u = 0;
        double                    f = 0.0;
        std::string               s;
        // For arrays of strings:
        std::vector<std::string>  arr_strs;
        // For arrays of scalars: raw bytes (length = count * sizeof(elem)).
        std::vector<uint8_t>      arr_bytes;
        size_t                    arr_count = 0;
    };

    struct TensorInfo {
        std::string name;
        int         type = 0;     // ggml_type
        uint64_t    offset = 0;
        uint64_t    size = 0;
    };

    uint32_t              version_   = 0;
    uint64_t              alignment_ = 0;
    uint64_t              data_off_  = 0;
    std::vector<KV>          kvs_;
    std::vector<TensorInfo>  tensors_;
};

// Registers `inspect(path, done)` on `exports`.
void RegisterInspect(Napi::Env env, Napi::Object exports);
