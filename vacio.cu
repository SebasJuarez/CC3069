#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

// Medicion de errores
#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


// Kernel: escribe 1 en out[i] si in[i] >= umbral, de lo contrario 0.
// TODO(1): agrega el calificador correcto para que esta función se ejecute en la GPU.
// TODO(2): completa los parámetros que necesita el kernel (entrada, salida, N y umbral).
__global__ void umbralizar(const int* in, int* out, int N, int umbral) {
    // TODO(3): calcula el índice global i usando blockIdx.x, blockDim.x y threadIdx.x.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO(4): control de límites para evitar accesos fuera de rango.
    if (i < N) {
        // TODO(5): escribe 1 si in[i] >= umbral, si no 0, en out[i].
        // Sugerencia: out[i] = (in[i] >= umbral) ? 1 : 0;
        out[i] = (in[i] >= umbral) ? 1 : 0;
    }
}

int main(int argc, char** argv) {
    // ----- Parámetros del problema -----
    // Permite pasar N y UMBRAL por línea de comandos: ./prog [N] [UMBRAL]
    const int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);  // por defecto ~16.7M
    const int UMBRAL = (argc > 2) ? std::atoi(argv[2]) : 50;

    printf("N = %d, UMBRAL = %d\n", N, UMBRAL);

    // Memoria host
    int *h_in  = new int[N];
    int *h_out = new int[N];

    // Inicialización en host
    for (int i = 0; i < N; ++i) h_in[i] = i & 1023;

    // Memoria device
    int *d_in  = nullptr;
    int *d_out = nullptr;

    // ----- Reserva de memoria en GPU -----
    CUDA_OK(cudaMalloc((void**)&d_in,  N * sizeof(int)));
    CUDA_OK(cudaMalloc((void**)&d_out, N * sizeof(int)));

    // ----- Medición de tiempo -----
    // a) cronómetro total (host) para pipeline completo
    auto t0_total = std::chrono::high_resolution_clock::now();

    // b) events para medir H2D, kernel y D2H por separado
    cudaEvent_t eH2D_s, eH2D_t, eK_s, eK_t, eD2H_s, eD2H_t;
    CUDA_OK(cudaEventCreate(&eH2D_s));
    CUDA_OK(cudaEventCreate(&eH2D_t));
    CUDA_OK(cudaEventCreate(&eK_s));
    CUDA_OK(cudaEventCreate(&eK_t));
    CUDA_OK(cudaEventCreate(&eD2H_s));
    CUDA_OK(cudaEventCreate(&eD2H_t));

    // ----- Copia Host -> Device -----
    CUDA_OK(cudaEventRecord(eH2D_s));
    CUDA_OK(cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaEventRecord(eH2D_t));
    CUDA_OK(cudaEventSynchronize(eH2D_t));

    // ----- Configuración de lanzamiento -----
    /* TODO: p.ej., 128 */
    /* TODO: fórmula de techo para cubrir N */
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    // ----- Lanzamiento del kernel -----
    CUDA_OK(cudaEventRecord(eK_s));
    umbralizar<<<gridSize, blockSize>>>(d_in, d_out, N, UMBRAL);
    CUDA_OK(cudaEventRecord(eK_t));
    // (Recomendado) Sincroniza para esperar a que termine el kernel.
    // TODO(11): usa cudaDeviceSynchronize();
    /* TODO */ 
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventSynchronize(eK_t));

    // ----- Copia Device -> Host -----
    CUDA_OK(cudaEventRecord(eD2H_s));
    CUDA_OK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaEventRecord(eD2H_t));
    CUDA_OK(cudaEventSynchronize(eD2H_t));

    auto t1_total = std::chrono::high_resolution_clock::now();
    double ms_total = std::chrono::duration<double, std::milli>(t1_total - t0_total).count();

    float ms_h2d=0, ms_kernel=0, ms_d2h=0;
    CUDA_OK(cudaEventElapsedTime(&ms_h2d,   eH2D_s, eH2D_t));
    CUDA_OK(cudaEventElapsedTime(&ms_kernel,eK_s,   eK_t));
    CUDA_OK(cudaEventElapsedTime(&ms_d2h,   eD2H_s, eD2H_t));

    // ----- Verificación rápida -----
    int ok = 1;
    for (int i = 0; i < 10 && i < N; ++i) {
        int expect = (h_in[i] >= UMBRAL) ? 1 : 0;
        if (h_out[i] != expect) ok = 0;
        printf("in[%2d]=%4d -> out=%d (esperado=%d)\n", i, h_in[i], h_out[i], expect);
    }
    printf("Check rapido: %s\n", ok ? "OK" : "MAL");

    // ----- Tiempos -----
    printf("\n== Tiempos GPU ==\n");
    printf("H2D (ms):   %.3f\n", ms_h2d);
    printf("Kernel (ms):%.3f\n", ms_kernel);
    printf("D2H (ms):   %.3f\n", ms_d2h);
    printf("Total (ms): %.3f  (incluye H2D + kernel + D2H)\n", ms_total);

    // ----- Limpieza -----
    /* TODO */ 
    CUDA_OK(cudaFree(d_in));
    /* TODO */ 
    CUDA_OK(cudaFree(d_out));
    CUDA_OK(cudaEventDestroy(eH2D_s));
    CUDA_OK(cudaEventDestroy(eH2D_t));
    CUDA_OK(cudaEventDestroy(eK_s));
    CUDA_OK(cudaEventDestroy(eK_t));
    CUDA_OK(cudaEventDestroy(eD2H_s));
    CUDA_OK(cudaEventDestroy(eD2H_t));

    delete[] h_in;
    delete[] h_out;

    printf("Finalizado (host).\n");
    return 0;
}
