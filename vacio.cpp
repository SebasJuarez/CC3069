#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
    const int UMBRAL = (argc > 2) ? std::atoi(argv[2]) : 50;

    printf("CPU-Only | N = %d, UMBRAL = %d\n", N, UMBRAL);

    std::vector<int> in(N), out(N);

    for (int i = 0; i < N; ++i) in[i] = i & 1023;

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; ++i)
        out[i] = (in[i] >= UMBRAL) ? 1 : 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    for (int i = 0; i < 10 && i < N; ++i) {
        int expect = (in[i] >= UMBRAL) ? 1 : 0;
        printf("in[%2d]=%4d -> out=%d (esperado=%d)\n", i, in[i], out[i], expect);
    }

    printf("\n== Tiempo CPU ==\n");
    printf("Loop CPU (ms): %.3f\n", ms);
    return 0;
}
