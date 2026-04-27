#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

// Per-env state shared across LlamaModel instances and AsyncWorkers.
// Tracks workers in-flight on the libuv thread pool so that a Node env
// shutdown can wait for them to drain before freeing the llama backend.
struct AddonState {
    // Number of worker `Execute()` bodies currently running. Inc/dec via
    // `WorkerGuard` at the top of each worker's `Execute()`.
    std::atomic<int>  in_flight{0};

    // Set once by the env cleanup hook. Workers check it as a fast-exit
    // signal; `LlamaModel::AbortCallback` also reads it so an in-flight
    // `llama_decode` returns promptly during shutdown instead of decoding
    // to the next token boundary.
    std::atomic<bool> shutting_down{false};

    std::mutex                mu;
    std::condition_variable   cv;

    // Wait (with timeout) for `in_flight` to reach zero.
    bool wait_for_drain(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, timeout, [this]() {
            return in_flight.load(std::memory_order_acquire) == 0;
        });
    }
};

// RAII helper: increments AddonState::in_flight on construction, decrements
// on destruction; notifies the env cleanup hook when the count hits zero.
// Instantiate at the top of every worker's `Execute()` body.
class WorkerGuard {
public:
    explicit WorkerGuard(AddonState * s) : s_(s) {
        s_->in_flight.fetch_add(1, std::memory_order_acq_rel);
    }
    ~WorkerGuard() {
        if (s_->in_flight.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::lock_guard<std::mutex> lk(s_->mu);
            s_->cv.notify_all();
        }
    }
    WorkerGuard(const WorkerGuard &)             = delete;
    WorkerGuard & operator=(const WorkerGuard &) = delete;

    bool shutting_down() const {
        return s_->shutting_down.load(std::memory_order_acquire);
    }

private:
    AddonState * s_;
};
