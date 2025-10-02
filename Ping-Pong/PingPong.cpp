#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <iomanip>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) std::cerr << "Este inciso requiere ejecutar con -np 2\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Parámetros: 
    // N iteraciones (default 100), warmup (default 10)
    // Para correr el programa: mpirun -np 2 ./pingpong [N] [WARMUP]
    int N = 100;
    int WARMUP = 10;
    if (argc >= 2) N = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) WARMUP = std::max(0, std::atoi(argv[2]));

    const int TAG = 0;
    int valor = 42;
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<double> rtt;
    if (rank == 0) rtt.reserve(N);

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        if (rank == 0) {
            MPI_Send(&valor, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD);
            MPI_Recv(&valor, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD, &status);
        } else { // rank 1
            MPI_Recv(&valor, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD, &status);
            MPI_Send(&valor, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD);
        }
    }

    // Iteraciones
    for (int i = 0; i < N; ++i) {
        if (rank == 0) {
            double t0 = MPI_Wtime();
            MPI_Send(&valor, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD);
            MPI_Recv(&valor, 1, MPI_INT, 1, TAG, MPI_COMM_WORLD, &status);
            double t1 = MPI_Wtime();
            rtt.push_back(t1 - t0);
        } else {
            MPI_Recv(&valor, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD, &status);
            MPI_Send(&valor, 1, MPI_INT, 0, TAG, MPI_COMM_WORLD);
        }
    }

    // Datos para el CSV
    if (rank == 0) {
        double sum = 0.0, minv = 1e9, maxv = 0.0;
        for (double x : rtt) { sum += x; if (x < minv) minv = x; if (x > maxv) maxv = x; }
        double avg = sum / rtt.size();

        std::ofstream out("pingpong_times.csv");
        out << "iteration,roundtrip_seconds,roundtrip_microseconds,message_bytes\n";
        const size_t msg_bytes = sizeof(int);
        for (int i = 0; i < N; ++i) {
            double us = rtt[i] * 1e6;
            out << (i+1) << "," << std::setprecision(9) << rtt[i] << "," << (long long)us << "," << msg_bytes << "\n";
        }
        out.close();

        std::cout << "Resultados guardados en pingpong_times.csv\n";
        std::cout << "Iteraciones medidas: " << N << " (warmup=" << WARMUP << ")\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "RTT promedio = " << (avg * 1e6) << " us | "
                  << "min = " << (minv * 1e6) << " us | "
                  << "max = " << (maxv * 1e6) << " us\n";
    }

    MPI_Finalize();
    return 0;
}
