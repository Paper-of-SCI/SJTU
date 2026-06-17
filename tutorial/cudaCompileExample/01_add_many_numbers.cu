#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err));                 \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

__global__ void add_many_numbers_kernel(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 8;
    const int bytes = n * sizeof(int);

    int h_a[n] = {0, 1, 2, 3, 4, 5, 6, 7};
    int h_b[n] = {100, 101, 102, 103, 104, 105, 106, 107};
    int h_c[n] = {0, 0, 0, 0, 0, 0, 0, 0};

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 4;
    int blocks = (n + threads_per_block - 1) / threads_per_block;

    add_many_numbers_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        int expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            ok = false;
        }
        std::printf("c[%d] = %d + %d = %d\n", i, h_a[i], h_b[i], h_c[i]);
    }

    std::printf("check: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
