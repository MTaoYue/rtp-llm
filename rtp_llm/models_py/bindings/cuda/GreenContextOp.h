#pragma once

#include <pybind11/pybind11.h>
#include <cuda_runtime_api.h>

namespace rtp_llm {

// Wraps a CUDA Green Context + non-blocking Stream pair.
//
// A Green Context allows partitioning the GPU's streaming multiprocessors (SMs)
// between independent work streams so that two compute chains can overlap on
// physically separate SM partitions.
//
// Requires CUDART_VERSION >= 12040 (CUDA 12.4+, when cudaGreenCtxCreate was
// introduced).  If the library is compiled against an older runtime the
// constructor raises a std::runtime_error.
//
// Typical Python usage:
//   gc = rtp_llm_ops.GreenContextStream(device=0, sm_count=72)
//   ext_stream = torch.cuda.ExternalStream(gc.stream_ptr(), device=device)
//   with torch.cuda.stream(ext_stream):
//       ...  # kernels run on the green-context stream
class GreenContextStream {
public:
    // Create a green context on `device` requesting at least `sm_count` SMs,
    // then create a non-blocking CUDA stream bound to that context.
    GreenContextStream(int device, int sm_count);
    ~GreenContextStream();

    // Non-copyable.
    GreenContextStream(const GreenContextStream&)            = delete;
    GreenContextStream& operator=(const GreenContextStream&) = delete;

    // Returns the raw cudaStream_t handle cast to uint64_t.
    // Pass to torch.cuda.ExternalStream(stream_ptr, device=...).
    uint64_t stream_ptr() const {
        return reinterpret_cast<uint64_t>(stream_);
    }

    // Returns the actual SM count allocated by the green context (may differ
    // from the requested count due to alignment/granularity constraints).
    int actual_sm_count() const {
        return actual_sm_count_;
    }

private:
    cudaExecutionContext_t ctx_{nullptr};
    cudaStream_t           stream_{nullptr};
    int                    actual_sm_count_{0};
    int                    device_{0};
};

// Wraps a pair of CUDA Green Contexts that partition the device's SM resources
// without overlap.
//
// The device SM pool is split once: partition A receives `sm_count_a` SMs
// (rounded up to alignment), and partition B receives the remainder.
// Total actual_sm_count_a() + actual_sm_count_b() <= total device SMs.
//
// Typical Python usage:
//   pair = rtp_llm_ops.GreenContextStreamPair(device=0, sm_count_a=74)
//   stream_a = torch.cuda.ExternalStream(pair.stream_a_ptr(), device=device)
//   stream_b = torch.cuda.ExternalStream(pair.stream_b_ptr(), device=device)
//   print(pair.actual_sm_count_a(), pair.actual_sm_count_b())
class GreenContextStreamPair {
public:
    // Split the device SM pool: A gets sm_count_a SMs (≥ requested, rounded to
    // alignment), B gets the remainder.  Both are backed by non-blocking streams.
    GreenContextStreamPair(int device, int sm_count_a);
    ~GreenContextStreamPair();

    GreenContextStreamPair(const GreenContextStreamPair&)            = delete;
    GreenContextStreamPair& operator=(const GreenContextStreamPair&) = delete;

    uint64_t stream_a_ptr() const {
        return reinterpret_cast<uint64_t>(stream_a_);
    }
    uint64_t stream_b_ptr() const {
        return reinterpret_cast<uint64_t>(stream_b_);
    }
    int actual_sm_count_a() const {
        return sm_count_a_;
    }
    int actual_sm_count_b() const {
        return sm_count_b_;
    }

private:
    cudaExecutionContext_t ctx_a_{nullptr};
    cudaExecutionContext_t ctx_b_{nullptr};
    cudaStream_t           stream_a_{nullptr};
    cudaStream_t           stream_b_{nullptr};
    int                    sm_count_a_{0};
    int                    sm_count_b_{0};
    int                    device_{0};
};

void registerGreenContextStream(pybind11::module& m);

}  // namespace rtp_llm
