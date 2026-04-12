// kernels/layernorm/15_embed.cu
#include "layernorm/15_embed.cuh"

__global__ void embedding_kernel(
    const int* __restrict__ token_ids,
    int seq_len, int start_pos, int d_model,
    const float* __restrict__ wte,
    const float* __restrict__ wpe,
    float* __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;
    if (idx >= total) return;

    int token_pos = idx / d_model;
    int dim = idx % d_model;
    int token_id = token_ids[token_pos];
    int abs_pos = start_pos + token_pos;

    out[token_pos * d_model + dim] = wte[token_id * d_model + dim]
                                   + wpe[abs_pos * d_model + dim];
}

void run_embedding(const int* token_ids, int seq_len, int start_pos, int d_model,
                   const float* wte, const float* wpe, float* out) {
    int total = seq_len * d_model;
    int block = 256;
    int grid = (total + block - 1) / block;
    embedding_kernel<<<grid, block>>>(token_ids, seq_len, start_pos, d_model, wte, wpe, out);
}
