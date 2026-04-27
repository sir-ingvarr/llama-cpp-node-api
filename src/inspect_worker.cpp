#include "inspect_worker.h"

#include <cstring>
#include <exception>
#include <memory>
#include <vector>

#include "addon_state.h"
#include "ggml.h"
#include "gguf.h"

// Refuse arrays whose declared count is implausibly large — guards against
// corrupt or malicious GGUF files asking us to allocate gigabytes.
static constexpr size_t INSPECT_MAX_ARRAY_LEN = 1u << 24;  // 16M elements

InspectWorker::InspectWorker(Napi::Function & done_cb, std::string path)
    : Napi::AsyncWorker(done_cb.Env()),
      path_(std::move(path))
{
    done_cb_ = Napi::Persistent(done_cb);
    state_   = done_cb.Env().GetInstanceData<AddonState>();
}

// ---------------------------------------------------------------------------
// Worker thread: open the file, copy out everything we need, free the ctx.
// We deliberately do not keep gguf_context alive past Execute(): all data is
// snapshotted into plain C++ types so OnOK() only marshals to V8.
// ---------------------------------------------------------------------------
void InspectWorker::Execute() {
    WorkerGuard guard(state_);
    if (guard.shutting_down()) {
        SetError("env is shutting down");
        return;
    }

    using GgufPtr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
    GgufPtr gguf{nullptr, &gguf_free};
    {
        gguf_init_params p { /* no_alloc */ true, /* ctx */ nullptr };
        gguf.reset(gguf_init_from_file(path_.c_str(), p));
    }
    if (!gguf) {
        SetError("gguf_init_from_file failed: " + path_);
        return;
    }

    // Wrap the snapshot pass in a try/catch — gguf_get_arr_str / vector growth
    // can throw bad_alloc on malformed or pathologically large files, and we
    // must surface that as a SetError rather than letting it cross the
    // AsyncWorker boundary (NAPI_DISABLE_CPP_EXCEPTIONS is in effect).
    try {
        version_   = gguf_get_version(gguf.get());
        alignment_ = gguf_get_alignment(gguf.get());
        data_off_  = gguf_get_data_offset(gguf.get());

        const int64_t n_kv = gguf_get_n_kv(gguf.get());
        kvs_.reserve((size_t) n_kv);
        for (int64_t i = 0; i < n_kv; ++i) {
            KV kv;
            kv.name = gguf_get_key(gguf.get(), i);
            kv.type = (int) gguf_get_kv_type(gguf.get(), i);

            switch (kv.type) {
                case GGUF_TYPE_UINT8:   kv.u = gguf_get_val_u8 (gguf.get(), i); break;
                case GGUF_TYPE_INT8:    kv.i = gguf_get_val_i8 (gguf.get(), i); break;
                case GGUF_TYPE_UINT16:  kv.u = gguf_get_val_u16(gguf.get(), i); break;
                case GGUF_TYPE_INT16:   kv.i = gguf_get_val_i16(gguf.get(), i); break;
                case GGUF_TYPE_UINT32:  kv.u = gguf_get_val_u32(gguf.get(), i); break;
                case GGUF_TYPE_INT32:   kv.i = gguf_get_val_i32(gguf.get(), i); break;
                case GGUF_TYPE_UINT64:  kv.u = gguf_get_val_u64(gguf.get(), i); break;
                case GGUF_TYPE_INT64:   kv.i = gguf_get_val_i64(gguf.get(), i); break;
                case GGUF_TYPE_FLOAT32: kv.f = gguf_get_val_f32(gguf.get(), i); break;
                case GGUF_TYPE_FLOAT64: kv.f = gguf_get_val_f64(gguf.get(), i); break;
                case GGUF_TYPE_BOOL:    kv.b = gguf_get_val_bool(gguf.get(), i); break;
                case GGUF_TYPE_STRING:  kv.s = gguf_get_val_str(gguf.get(), i); break;
                case GGUF_TYPE_ARRAY: {
                    const int et = (int) gguf_get_arr_type(gguf.get(), i);
                    kv.arr_elem_type = et;
                    const size_t n  = gguf_get_arr_n(gguf.get(), i);
                    if (n > INSPECT_MAX_ARRAY_LEN) {
                        SetError("gguf array '" + kv.name +
                                 "' too large: " + std::to_string(n) +
                                 " elements (max " +
                                 std::to_string(INSPECT_MAX_ARRAY_LEN) + ")");
                        return;
                    }
                    kv.arr_count = n;
                    if (et == GGUF_TYPE_STRING) {
                        kv.arr_strs.reserve(n);
                        for (size_t j = 0; j < n; ++j) {
                            kv.arr_strs.emplace_back(gguf_get_arr_str(gguf.get(), i, j));
                        }
                    } else if (et != GGUF_TYPE_ARRAY) {
                        // Scalar element type: copy raw bytes.
                        size_t elem = 0;
                        switch (et) {
                            case GGUF_TYPE_UINT8:   case GGUF_TYPE_INT8:
                            case GGUF_TYPE_BOOL:    elem = 1; break;
                            case GGUF_TYPE_UINT16:  case GGUF_TYPE_INT16:   elem = 2; break;
                            case GGUF_TYPE_UINT32:  case GGUF_TYPE_INT32:
                            case GGUF_TYPE_FLOAT32: elem = 4; break;
                            case GGUF_TYPE_UINT64:  case GGUF_TYPE_INT64:
                            case GGUF_TYPE_FLOAT64: elem = 8; break;
                            default: elem = 0;
                        }
                        if (elem) {
                            kv.arr_bytes.resize(elem * n);
                            const void * src = gguf_get_arr_data(gguf.get(), i);
                            if (src && !kv.arr_bytes.empty()) {
                                std::memcpy(kv.arr_bytes.data(), src, kv.arr_bytes.size());
                            }
                        }
                    }
                    // Nested arrays (GGUF_TYPE_ARRAY of GGUF_TYPE_ARRAY) are not
                    // representable in current GGUF and intentionally unhandled.
                    break;
                }
                default: break;
            }
            kvs_.push_back(std::move(kv));
        }

        const int64_t n_t = gguf_get_n_tensors(gguf.get());
        tensors_.reserve((size_t) n_t);
        for (int64_t i = 0; i < n_t; ++i) {
            TensorInfo t;
            t.name   = gguf_get_tensor_name(gguf.get(), i);
            t.type   = (int) gguf_get_tensor_type(gguf.get(), i);
            t.offset = gguf_get_tensor_offset(gguf.get(), i);
            t.size   = gguf_get_tensor_size(gguf.get(), i);
            tensors_.push_back(std::move(t));
        }
    } catch (const std::exception & e) {
        SetError(std::string("inspect: ") + e.what());
        return;
    }
    // gguf_context freed by GgufPtr's deleter on scope exit.
}

// ---------------------------------------------------------------------------
// Marshal the snapshot into a JS object.
// ---------------------------------------------------------------------------

static Napi::Value scalar_to_js(Napi::Env env, int gguf_type,
                                bool b, int64_t i, uint64_t u, double f,
                                const std::string & s) {
    switch (gguf_type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_UINT32:
            return Napi::Number::New(env, (double) u);
        case GGUF_TYPE_UINT64:
            return Napi::BigInt::New(env, u);
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_INT16:
        case GGUF_TYPE_INT32:
            return Napi::Number::New(env, (double) i);
        case GGUF_TYPE_INT64:
            return Napi::BigInt::New(env, i);
        case GGUF_TYPE_FLOAT32:
        case GGUF_TYPE_FLOAT64:
            return Napi::Number::New(env, f);
        case GGUF_TYPE_BOOL:
            return Napi::Boolean::New(env, b);
        case GGUF_TYPE_STRING:
            return Napi::String::New(env, s);
        default:
            return env.Null();
    }
}

// memcpy into a typed local rather than reinterpret-casting through the byte
// buffer — both for strict-aliasing safety (vector<uint8_t> may not satisfy
// alignof(double) on all platforms) and because reinterpret_cast through
// std::vector<uint8_t>::data() is technically UB even when alignment happens
// to be sufficient.
template <typename T>
static T read_at(const uint8_t * p, size_t j) {
    T v;
    std::memcpy(&v, p + j * sizeof(T), sizeof(T));
    return v;
}

static Napi::Value scalar_array_to_js(Napi::Env env, int et,
                                      const std::vector<uint8_t> & bytes,
                                      size_t count) {
    Napi::Array arr = Napi::Array::New(env, count);
    const uint8_t * p = bytes.data();
    auto put = [&](size_t j, Napi::Value v) { arr.Set((uint32_t) j, v); };
    for (size_t j = 0; j < count; ++j) {
        switch (et) {
            case GGUF_TYPE_UINT8:   put(j, Napi::Number::New(env, (double) read_at<uint8_t >(p, j))); break;
            case GGUF_TYPE_INT8:    put(j, Napi::Number::New(env, (double) read_at<int8_t  >(p, j))); break;
            case GGUF_TYPE_UINT16:  put(j, Napi::Number::New(env, (double) read_at<uint16_t>(p, j))); break;
            case GGUF_TYPE_INT16:   put(j, Napi::Number::New(env, (double) read_at<int16_t >(p, j))); break;
            case GGUF_TYPE_UINT32:  put(j, Napi::Number::New(env, (double) read_at<uint32_t>(p, j))); break;
            case GGUF_TYPE_INT32:   put(j, Napi::Number::New(env, (double) read_at<int32_t >(p, j))); break;
            case GGUF_TYPE_UINT64:  put(j, Napi::BigInt::New(env, read_at<uint64_t>(p, j)));          break;
            case GGUF_TYPE_INT64:   put(j, Napi::BigInt::New(env, read_at<int64_t >(p, j)));          break;
            case GGUF_TYPE_FLOAT32: put(j, Napi::Number::New(env, (double) read_at<float   >(p, j))); break;
            case GGUF_TYPE_FLOAT64: put(j, Napi::Number::New(env, read_at<double>(p, j)));            break;
            case GGUF_TYPE_BOOL:    put(j, Napi::Boolean::New(env, read_at<int8_t>(p, j) != 0));      break;
            default:                put(j, env.Null());                                               break;
        }
    }
    return arr;
}

void InspectWorker::OnOK() {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);

    Napi::Object out = Napi::Object::New(env);
    out.Set("version",    Napi::Number::New(env, version_));
    out.Set("alignment",  Napi::Number::New(env, (double) alignment_));
    out.Set("dataOffset", Napi::BigInt::New(env, data_off_));

    Napi::Object meta = Napi::Object::New(env);
    for (const KV & kv : kvs_) {
        if (kv.type == GGUF_TYPE_ARRAY) {
            if (kv.arr_elem_type == GGUF_TYPE_STRING) {
                Napi::Array arr = Napi::Array::New(env, kv.arr_strs.size());
                for (size_t j = 0; j < kv.arr_strs.size(); ++j) {
                    arr.Set((uint32_t) j, Napi::String::New(env, kv.arr_strs[j]));
                }
                meta.Set(kv.name, arr);
            } else {
                meta.Set(kv.name,
                    scalar_array_to_js(env, kv.arr_elem_type, kv.arr_bytes, kv.arr_count));
            }
        } else {
            meta.Set(kv.name,
                scalar_to_js(env, kv.type, kv.b, kv.i, kv.u, kv.f, kv.s));
        }
    }
    out.Set("metadata", meta);

    Napi::Array tarr = Napi::Array::New(env, tensors_.size());
    for (size_t j = 0; j < tensors_.size(); ++j) {
        const TensorInfo & t = tensors_[j];
        Napi::Object o = Napi::Object::New(env);
        o.Set("name",   Napi::String::New(env, t.name));
        o.Set("type",   Napi::String::New(env, ggml_type_name((ggml_type) t.type)));
        o.Set("offset", Napi::BigInt::New(env, t.offset));
        o.Set("size",   Napi::BigInt::New(env, t.size));
        tarr.Set((uint32_t) j, o);
    }
    out.Set("tensors", tarr);

    done_cb_.Call({env.Null(), out});
}

void InspectWorker::OnError(const Napi::Error & error) {
    Napi::Env env = done_cb_.Env();
    Napi::HandleScope scope(env);
    done_cb_.Call({error.Value()});
}

// inspect(path, done)
static Napi::Value Inspect(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsFunction()) {
        Napi::TypeError::New(env, "inspect(path: string, done: fn)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    std::string path  = info[0].As<Napi::String>().Utf8Value();
    Napi::Function cb = info[1].As<Napi::Function>();
    auto * w = new InspectWorker(cb, std::move(path));
    w->Queue();
    return env.Undefined();
}

void RegisterInspect(Napi::Env env, Napi::Object exports) {
    exports.Set("inspect", Napi::Function::New(env, Inspect, "inspect"));
}
