#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "config.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace nanochat {

// GEMM for bf16: C = A @ B^T (row-major A, row-major B, row-major C)
// A: [M, K], B: [N, K], C: [M, N]
using GemmBf16 = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t,                      // ElementA
    cutlass::layout::RowMajor,                // LayoutA
    cutlass::bfloat16_t,                      // ElementB
    cutlass::layout::ColumnMajor,             // LayoutB (transposed row-major)
    cutlass::bfloat16_t,                      // ElementC
    cutlass::layout::RowMajor,                // LayoutC
    float,                                    // ElementAccumulator
    cutlass::arch::OpClassTensorOp,           // Use tensor cores
    cutlass::arch::Sm80,                      // Ampere+ (compatible with Ada)
    cutlass::gemm::GemmShape<128, 128, 32>,   // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,     // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,      // InstructionShape
    cutlass::epilogue::thread::LinearCombination<
        cutlass::bfloat16_t, 8, float, float>,// EpilogueOp
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3                                         // Stages
>;

static GemmBf16 gemm_bf16_op;

// C[M,N] = A[M,K] @ B[N,K]^T (bf16 precision)
void gemm_half(nv_bfloat16* C, const nv_bfloat16* A, const nv_bfloat16* B, int M, int N, int K, cudaStream_t stream) {
    typename GemmBf16::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,  // batch count
        {1.0f, 0.0f},  // alpha, beta
        reinterpret_cast<const cutlass::bfloat16_t*>(A),
        reinterpret_cast<const cutlass::bfloat16_t*>(B),
        reinterpret_cast<cutlass::bfloat16_t*>(C),
        reinterpret_cast<cutlass::bfloat16_t*>(C),
        M * K,  // batch stride A
        N * K,  // batch stride B
        M * N,  // batch stride C
        M * N,  // batch stride D
        K,      // lda
        K,      // ldb
        N,      // ldc
        N       // ldd
    );

    cutlass::Status status = gemm_bf16_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM bf16 failed\n");
    }
}

// Strided batched GEMM for Q @ K^T with [seq, heads, dim] layout
// Using SIMT for flexible stride support (tensor cores need aligned strides)
using GemmQK = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t,
    cutlass::layout::RowMajor,
    cutlass::bfloat16_t,
    cutlass::layout::ColumnMajor,             // K is transposed
    cutlass::bfloat16_t,                      // Output bf16
    cutlass::layout::RowMajor,
    float,                                    // Accumulator fp32
    cutlass::arch::OpClassSimt,               // SIMT for flexible strides
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2
>;

// Strided batched GEMM for scores @ V
// Using SIMT for flexible stride support
using GemmSV = cutlass::gemm::device::GemmUniversal<
    cutlass::bfloat16_t,                      // scores bf16
    cutlass::layout::RowMajor,
    cutlass::bfloat16_t,                      // V bf16
    cutlass::layout::RowMajor,
    cutlass::bfloat16_t,                      // output bf16
    cutlass::layout::RowMajor,
    float,                                    // Accumulator fp32
    cutlass::arch::OpClassSimt,               // SIMT for flexible strides
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    2
>;

static GemmQK gemm_qk_op;
static GemmSV gemm_sv_op;

// Q[seq_q, heads, dim] @ K[seq_k, heads, dim]^T -> scores[heads, seq_q, seq_k]
// No transpose needed - uses strides to handle interleaved layout
void gemm_qk_strided(
    nv_bfloat16* scores,          // [heads, seq_q, seq_k] bf16
    const nv_bfloat16* Q,         // [seq_q, heads, dim]
    const nv_bfloat16* K,         // [seq_k, heads, dim]
    int seq_q, int seq_k, int heads, int dim,
    float scale,
    cudaStream_t stream) {

    typename GemmQK::Arguments args(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {seq_q, seq_k, dim},              // M, N, K
        heads,                             // batch count
        {scale, 0.0f},                     // alpha, beta
        reinterpret_cast<const cutlass::bfloat16_t*>(Q),
        reinterpret_cast<const cutlass::bfloat16_t*>(K),
        reinterpret_cast<cutlass::bfloat16_t*>(scores),
        reinterpret_cast<cutlass::bfloat16_t*>(scores),
        dim,                               // batch stride A (to next head in Q)
        dim,                               // batch stride B (to next head in K)
        seq_q * seq_k,                     // batch stride C (to next head in scores)
        seq_q * seq_k,                     // batch stride D
        heads * dim,                       // lda (stride between rows in Q)
        heads * dim,                       // ldb (stride between rows in K)
        seq_k,                             // ldc
        seq_k                              // ldd
    );

    cutlass::Status status = gemm_qk_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS gemm_qk_strided failed: %d\n", (int)status);
    }
}

// scores[heads, seq_q, seq_k] @ V[seq_k, heads, dim] -> out[seq_q, heads, dim]
// Uses strides to avoid transpose
void gemm_sv_strided(
    nv_bfloat16* out,             // [seq_q, heads, dim]
    const nv_bfloat16* scores,    // [heads, seq_q, seq_k] bf16
    const nv_bfloat16* V,         // [seq_k, heads, dim]
    int seq_q, int seq_k, int heads, int dim,
    cudaStream_t stream) {

    typename GemmSV::Arguments args(
        cutlass::gemm::GemmUniversalMode::kBatched,
        {seq_q, dim, seq_k},              // M, N, K
        heads,                             // batch count
        {1.0f, 0.0f},                      // alpha, beta
        reinterpret_cast<const cutlass::bfloat16_t*>(scores),
        reinterpret_cast<const cutlass::bfloat16_t*>(V),
        reinterpret_cast<cutlass::bfloat16_t*>(out),
        reinterpret_cast<cutlass::bfloat16_t*>(out),
        seq_q * seq_k,                     // batch stride A (to next head in scores)
        dim,                               // batch stride B (to next head in V)
        dim,                               // batch stride C (to next head in out)
        dim,                               // batch stride D
        seq_k,                             // lda (stride between rows in scores)
        heads * dim,                       // ldb (stride between rows in V)
        heads * dim,                       // ldc (stride between rows in out)
        heads * dim                        // ldd
    );

    cutlass::Status status = gemm_sv_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS gemm_sv_strided failed: %d\n", (int)status);
    }
}

}  // namespace nanochat
