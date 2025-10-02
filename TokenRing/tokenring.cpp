#include <mpi.h>
#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) std::cerr << "Se requieren al menos 2 procesos.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Parámetros: R = vueltas del anillo
    int R = 1;                              
    if (argc >= 2) R = std::max(1, std::atoi(argv[1]));

    const int TAG = 0;
    const int EMPTY = -1;
    const int next = (rank + 1) % size;
    const int prev = (rank - 1 + size) % size;

  
    int token = (rank == 0) ? 42 : EMPTY;
    int recv_token = EMPTY;

    if (rank == 0)
        std::cout << "Token inicial = " << token << " | Vueltas a dar = " << R << "\n";

    MPI_Barrier(MPI_COMM_WORLD);

    const int total_hops = R * size;
    int vueltas_completadas = 0;

    for (int step = 0; step < total_hops; ++step) {
        MPI_Status st;
        MPI_Sendrecv(
            &token, 1, MPI_INT, next, TAG,
            &recv_token, 1, MPI_INT, prev, TAG,
            MPI_COMM_WORLD, &st
        );

        token = recv_token;

        if (rank == 0 && token != EMPTY) {
            ++vueltas_completadas;
            std::cout << "Rank 0 recibio el token (vuelta " << vueltas_completadas << "/" << R
                      << ") en el paso " << (step + 1) << "\n";
        }
    }

    if (rank == 0) {
        if (vueltas_completadas == R)
            std::cout << "Token Ring con MPI_Sendrecv completado correctamente.\n";
        else
            std::cout << "Advertencia: vueltas contadas = " << vueltas_completadas
                      << " (esperadas " << R << ")\n";
    }

    MPI_Finalize();
    return 0;
}