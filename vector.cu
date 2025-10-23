#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Kernel
__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(int argc, char** argv) {
    // Arreglos de tamaño N
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Fallo al reservar memoria en host.\n");
        return 1;
    }

    // Inicializar A y B
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 0.5f;
        h_B[i] = 2.0f * i + 1.0f;
    }

    // Reservar memoria
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_OK(cudaMalloc((void**)&d_A, bytes));
    CUDA_OK(cudaMalloc((void**)&d_B, bytes));
    CUDA_OK(cudaMalloc((void**)&d_C, bytes));

    // Copia de A y B a device
    CUDA_OK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Suma de vectores
    int TPB = 256;
    int blocks = (N + TPB - 1) / TPB;
    vecAdd<<<blocks, TPB>>>(d_A, d_B, d_C, N);
    CUDA_OK(cudaPeekAtLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // Copiar los resultados de host
    CUDA_OK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    printf("Primeros 10 resultados C[i] = A[i] + B[i]:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("i=%d  %.3f + %.3f = %.3f\n", i, h_A[i], h_B[i], h_C[i]);
    }
    // Limpieza
    CUDA_OK(cudaFree(d_A));
    CUDA_OK(cudaFree(d_B));
    CUDA_OK(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);

    return 0;
}
