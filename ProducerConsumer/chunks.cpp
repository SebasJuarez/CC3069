#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <cstdlib>

enum Tags : int { TAG_HDR=1, TAG_DATA=2, TAG_RES=3, TAG_STOP=9 };

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0, size=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) std::cerr << "Este ejercicio requiere -np 2\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Parámetros:
    // N = arreglo
    // CH_MIN = mínimo de chunks
    // CH_MAX = máximo de chunks
    // STEP_MUL = crecimiento 
    long long N = 1'000'000;
    int CH_MIN = 1024;
    int CH_MAX = 1<<18;
    int STEP_MUL = 2;

    if (argc >= 2) N        = std::max(1LL, std::atoll(argv[1]));
    if (argc >= 3) CH_MIN   = std::max(1,   std::atoi (argv[2]));
    if (argc >= 4) CH_MAX   = std::max(CH_MIN, std::atoi(argv[3]));
    if (argc >= 5) STEP_MUL = std::max(2,   std::atoi (argv[4]));

    std::vector<int> CHS;
    for (long long ch = CH_MIN; ch <= CH_MAX; ch *= STEP_MUL) CHS.push_back((int)ch);

    std::ofstream out;
    if (rank == 0) {
        out.open("chunks_summary.csv");
        out << "N_elems,chunk_elems,chunk_bytes,num_chunks,total_seconds,MBps\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int CH : CHS) {
        if (rank == 0) {
            // ===== Producer =====
            std::vector<int> A((size_t)N, 1);
            const long long chunks = (N + CH - 1) / CH;

            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();

            long long total_sum = 0;
            for (long long c = 0; c < chunks; ++c) {
                const long long start = c * CH;
                const int count = (int)std::min<long long>(CH, N - start);

                MPI_Send(&count, 1, MPI_INT, 1, TAG_HDR, MPI_COMM_WORLD);
                MPI_Send(A.data() + start, count, MPI_INT, 1, TAG_DATA, MPI_COMM_WORLD);

                long long partial = 0;
                MPI_Recv(&partial, 1, MPI_LONG_LONG, 1, TAG_RES, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                total_sum += partial;
            }

            double t1 = MPI_Wtime();
            double total_seconds = t1 - t0;


            int dummy = 0; MPI_Request req;
            MPI_Isend(&dummy, 1, MPI_INT, 1, TAG_STOP, MPI_COMM_WORLD, &req);
            MPI_Wait(&req, MPI_STATUS_IGNORE);


            const double MBps = ( (N * sizeof(int)) / 1e6 ) / std::max(1e-12, total_seconds);
            if (total_sum != N) {
                std::cerr << "[0] ADVERTENCIA: suma total=" << total_sum
                          << " (esperada " << N << ")\n";
            }
            if (out.is_open()) {
                out << N << "," << CH << "," << (CH * (int)sizeof(int)) << ","
                    << chunks << "," << std::setprecision(9) << total_seconds << ","
                    << std::fixed << std::setprecision(3) << MBps << "\n";
            }

            MPI_Barrier(MPI_COMM_WORLD);

            std::cout << "[0] CH=" << CH
                      << " elems (" << CH * (int)sizeof(int) << " bytes)"
                      << " | chunks=" << chunks
                      << " | total=" << total_seconds << " s"
                      << " | MB/s=" << MBps << "\n";

        } else {
            // ===== Consumer =====
            MPI_Barrier(MPI_COMM_WORLD);
            while (true) {
                int count = 0;
                MPI_Status st;
                MPI_Recv(&count, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);

                if (st.MPI_TAG == TAG_STOP) {
                    break;
                }
                std::vector<int> buf(count);
                MPI_Recv(buf.data(), count, MPI_INT, 0, TAG_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                long long partial = 0;
                for (int x : buf) partial += x;

                MPI_Send(&partial, 1, MPI_LONG_LONG, 0, TAG_RES, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    if (rank == 0 && out.is_open()) out.close();
    MPI_Finalize();
    return 0;
}
