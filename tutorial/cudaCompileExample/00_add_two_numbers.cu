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

__global__ void add_two_numbers_kernel(const int* a, const int* b, int* c) {
    *c = *a + *b;
}

int main() {
    int h_a = 2;
    int h_b = 3;
    int h_c = 0;

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_b, sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_c, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice));

    add_two_numbers_kernel<<<1, 1>>>(d_a, d_b, d_c);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    std::printf("%d + %d = %d\n", h_a, h_b, h_c);
    std::printf("check: %s\n", h_c == 5 ? "PASS" : "FAIL");

    return h_c == 5 ? 0 : 1;
}
