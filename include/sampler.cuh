// include/sampler.cuh
#pragma once
#include <cuda_runtime.h>

int sample_argmax(const float* d_logits, int vocab_size);

int sample_top_k(const float* d_logits, int vocab_size,
                 int k, float temperature, unsigned int seed);
