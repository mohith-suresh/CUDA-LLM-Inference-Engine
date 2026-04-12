// src/sampler.cu
#include "sampler.cuh"
#include "timer.cuh"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>

int sample_argmax(const float* d_logits, int vocab_size) {
    std::vector<float> h(vocab_size);
    CUDA_CHECK(cudaMemcpy(h.data(), d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return (int)(std::max_element(h.begin(), h.end()) - h.begin());
}

int sample_top_k(const float* d_logits, int vocab_size,
                 int k, float temperature, unsigned int seed) {
    std::vector<float> h(vocab_size);
    CUDA_CHECK(cudaMemcpy(h.data(), d_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    if (k > vocab_size) k = vocab_size;
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](int a, int b) { return h[a] > h[b]; });

    float max_val = h[indices[0]];
    float sum = 0.0f;
    std::vector<float> probs(k);
    for (int i = 0; i < k; ++i) {
        probs[i] = expf((h[indices[i]] - max_val) / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < k; ++i) probs[i] /= sum;

    static unsigned int last_seed = 0;
    if (seed != last_seed) { srand(seed); last_seed = seed; }
    float r = (float)rand() / RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < k; ++i) {
        cum += probs[i];
        if (r < cum) return indices[i];
    }
    return indices[k - 1];
}
