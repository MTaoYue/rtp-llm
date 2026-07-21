#include "rtp_llm/models_py/bindings/cuda/GreenContextOp.h"

#include <sstream>
#include <stdexcept>
#include <string>

namespace rtp_llm {

namespace {

#define CHECK_CUDA_GC(call)                                                                                            \
    do {                                                                                                               \
        cudaError_t _err = (call);                                                                                     \
        if (_err != cudaSuccess) {                                                                                     \
            std::ostringstream _oss;                                                                                   \
            _oss << "[GreenContextStream] " << #call << " failed: " << cudaGetErrorString(_err) << " (error "          \
                 << static_cast<int>(_err) << ")";                                                                     \
            throw std::runtime_error(_oss.str());                                                                      \
        }                                                                                                              \
    } while (0)

}  // namespace

GreenContextStream::GreenContextStream(int device, int sm_count): device_(device) {
#if CUDART_VERSION >= 12040
    CHECK_CUDA_GC(cudaSetDevice(device));

    // Query the full SM resource for the device.
    cudaDevResource full_sm{};
    CHECK_CUDA_GC(cudaDeviceGetDevResource(device, &full_sm, cudaDevResourceTypeSm));

    // Split: request one group of at least sm_count SMs.  The runtime rounds up
    // to alignment boundaries, so actual_sm_count_ may exceed sm_count.
    cudaDevResource group_sm{};
    cudaDevResource remaining_sm{};
    unsigned int    nb_groups = 1u;
    CHECK_CUDA_GC(cudaDevSmResourceSplitByCount(
        &group_sm, &nb_groups, &full_sm, &remaining_sm, 0u, static_cast<unsigned int>(sm_count)));

    if (nb_groups == 0u) {
        throw std::runtime_error("[GreenContextStream] cudaDevSmResourceSplitByCount returned 0 groups "
                                 "(sm_count="
                                 + std::to_string(sm_count) + " may exceed device SM count)");
    }

    actual_sm_count_ = static_cast<int>(group_sm.sm.smCount);

    // Build a resource descriptor from the allocated SM group.
    cudaDevResourceDesc_t desc{};
    CHECK_CUDA_GC(cudaDevResourceGenerateDesc(&desc, &group_sm, 1u));

    // Create the green execution context.
    CHECK_CUDA_GC(cudaGreenCtxCreate(&ctx_, desc, device, 0u));

    // Create a non-blocking stream bound to the green context.
    CHECK_CUDA_GC(cudaExecutionCtxStreamCreate(&stream_, ctx_, cudaStreamNonBlocking, 0 /*priority*/));
#else
    throw std::runtime_error("[GreenContextStream] Requires CUDART_VERSION >= 12040 (CUDA 12.4+). "
                             "Current CUDART_VERSION="
                             + std::to_string(CUDART_VERSION));
#endif
}

GreenContextStream::~GreenContextStream() {
#if CUDART_VERSION >= 12040
    if (stream_ != nullptr) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    if (ctx_ != nullptr) {
        cudaExecutionCtxDestroy(ctx_);
        ctx_ = nullptr;
    }
#endif
}

GreenContextStreamPair::GreenContextStreamPair(int device, int sm_count_a): device_(device) {
#if CUDART_VERSION >= 12040
    CHECK_CUDA_GC(cudaSetDevice(device));

    // Query the full SM resource for the device.
    cudaDevResource full_sm{};
    CHECK_CUDA_GC(cudaDeviceGetDevResource(device, &full_sm, cudaDevResourceTypeSm));

    // Split the resource: group A gets sm_count_a SMs, the remainder goes to B.
    // cudaDevSmResourceSplitByCount with nbGroups=1 produces one result group
    // and stores the leftover in `remaining`.
    cudaDevResource group_a{};
    cudaDevResource remaining{};
    unsigned int    nb_groups = 1u;
    CHECK_CUDA_GC(cudaDevSmResourceSplitByCount(
        &group_a, &nb_groups, &full_sm, &remaining, 0u, static_cast<unsigned int>(sm_count_a)));

    if (nb_groups == 0u) {
        throw std::runtime_error("[GreenContextStreamPair] cudaDevSmResourceSplitByCount returned 0 groups "
                                 "(sm_count_a="
                                 + std::to_string(sm_count_a) + " may exceed device SM count)");
    }

    sm_count_a_ = static_cast<int>(group_a.sm.smCount);
    sm_count_b_ = static_cast<int>(remaining.sm.smCount);

    if (sm_count_b_ == 0) {
        throw std::runtime_error("[GreenContextStreamPair] No SMs left for partition B after allocating "
                                 + std::to_string(sm_count_a_)
                                 + " SMs to partition A.  "
                                   "Reduce sm_count_a.");
    }

    // Build resource descriptors for A and B.
    cudaDevResourceDesc_t desc_a{};
    CHECK_CUDA_GC(cudaDevResourceGenerateDesc(&desc_a, &group_a, 1u));

    cudaDevResourceDesc_t desc_b{};
    CHECK_CUDA_GC(cudaDevResourceGenerateDesc(&desc_b, &remaining, 1u));

    // Create the two green execution contexts.
    CHECK_CUDA_GC(cudaGreenCtxCreate(&ctx_a_, desc_a, device, 0u));
    CHECK_CUDA_GC(cudaGreenCtxCreate(&ctx_b_, desc_b, device, 0u));

    // Create non-blocking streams bound to each context.
    CHECK_CUDA_GC(cudaExecutionCtxStreamCreate(&stream_a_, ctx_a_, cudaStreamNonBlocking, 0));
    CHECK_CUDA_GC(cudaExecutionCtxStreamCreate(&stream_b_, ctx_b_, cudaStreamNonBlocking, 0));
#else
    throw std::runtime_error("[GreenContextStreamPair] Requires CUDART_VERSION >= 12040 (CUDA 12.4+). "
                             "Current CUDART_VERSION="
                             + std::to_string(CUDART_VERSION));
#endif
}

GreenContextStreamPair::~GreenContextStreamPair() {
#if CUDART_VERSION >= 12040
    auto sync_destroy = [](cudaStream_t& s, cudaExecutionContext_t& c) {
        if (s) {
            cudaStreamSynchronize(s);
            cudaStreamDestroy(s);
            s = nullptr;
        }
        if (c) {
            cudaExecutionCtxDestroy(c);
            c = nullptr;
        }
    };
    sync_destroy(stream_a_, ctx_a_);
    sync_destroy(stream_b_, ctx_b_);
#endif
}

void registerGreenContextStream(pybind11::module& m) {
    pybind11::class_<GreenContextStream>(m, "GreenContextStream")
        .def(pybind11::init<int, int>(),
             pybind11::arg("device"),
             pybind11::arg("sm_count"),
             "Create a green context on `device` with at least `sm_count` SMs and "
             "bind a non-blocking CUDA stream to it.  Each call draws independently "
             "from the device SM pool; use GreenContextStreamPair for non-overlapping "
             "partitions.")
        .def("stream_ptr",
             &GreenContextStream::stream_ptr,
             "Raw CUDA stream handle (uint64_t).  Pass to "
             "torch.cuda.ExternalStream(stream_ptr, device=device).")
        .def("actual_sm_count",
             &GreenContextStream::actual_sm_count,
             "Actual SM count allocated to this green context "
             "(may exceed the requested count due to alignment).");

    pybind11::class_<GreenContextStreamPair>(m, "GreenContextStreamPair")
        .def(pybind11::init<int, int>(),
             pybind11::arg("device"),
             pybind11::arg("sm_count_a"),
             "Split the device SM pool into two non-overlapping partitions.  "
             "Partition A receives at least `sm_count_a` SMs (rounded up to "
             "alignment); partition B receives the remainder.  "
             "actual_sm_count_a() + actual_sm_count_b() <= total device SMs.")
        .def(
            "stream_a_ptr", &GreenContextStreamPair::stream_a_ptr, "Raw CUDA stream handle for partition A (uint64_t).")
        .def(
            "stream_b_ptr", &GreenContextStreamPair::stream_b_ptr, "Raw CUDA stream handle for partition B (uint64_t).")
        .def("actual_sm_count_a",
             &GreenContextStreamPair::actual_sm_count_a,
             "Actual SM count allocated to partition A.")
        .def("actual_sm_count_b",
             &GreenContextStreamPair::actual_sm_count_b,
             "Actual SM count allocated to partition B "
             "(total device SMs minus partition A's allocation).");
}

}  // namespace rtp_llm
