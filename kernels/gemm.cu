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
static GemmBf16 gemm_bf16_residual_op;

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

// D[M,N] = A[M,K] @ B[N,K]^T + residual[M,N] (fused residual add)
void gemm_half_residual(nv_bfloat16* D, const nv_bfloat16* A, const nv_bfloat16* B,
                        const nv_bfloat16* residual, int M, int N, int K, cudaStream_t stream) {
    typename GemmBf16::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        1,  // batch count
        {1.0f, 1.0f},  // alpha=1, beta=1: D = 1*A@B^T + 1*residual
        reinterpret_cast<const cutlass::bfloat16_t*>(A),
        reinterpret_cast<const cutlass::bfloat16_t*>(B),
        reinterpret_cast<const cutlass::bfloat16_t*>(residual),  // C = residual (read)
        reinterpret_cast<cutlass::bfloat16_t*>(D),               // D = output (write)
        M * K,  // batch stride A
        N * K,  // batch stride B
        M * N,  // batch stride C
        M * N,  // batch stride D
        K,      // lda
        K,      // ldb
        N,      // ldc
        N       // ldd
    );

    cutlass::Status status = gemm_bf16_residual_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS GEMM bf16 residual failed\n");
    }
}

}  // namespace nanochat
