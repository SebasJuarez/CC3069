#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>

static inline void do_work(long long iters) {
    volatile double x = 1.0;
    for (long long i = 0; i < iters; ++i) {
        x = std::sqrt(x + 1.000000119);
    }
}

struct Stats { double avg, minv, maxv; };
static Stats stats_of(const std::vector<double>& v) {
    Stats s{};
    s.minv = *std::min_element(v.begin(), v.end());
    s.maxv = *std::max_element(v.begin(), v.end());
    s.avg  = std::accumulate(v.begin(), v.end(), 0.0) / std::max<size_t>(1, v.size());
    return s;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0, size=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) std::cerr << "Este programa requiere -np 2\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Parámetros (usando defaults pero se pueden introducir): N, WARMUP, WORK_ITERS
    int N = 30;
    int WARMUP = 5;
    long long WORK_ITERS = 20000;
    if (argc >= 2) N = std::max(1, std::atoi(argv[1]));
    if (argc >= 3) WARMUP = std::max(0, std::atoi(argv[2]));
    if (argc >= 4) WORK_ITERS = std::max(0LL, std::atoll(argv[3]));

    std::vector<int> sizes = {
        1, 8, 64, 512, 4096, 32768, 262144, 1048576
    };

    const int TAG = 0;

    // CSV
    std::ofstream raw, summary;
    if (rank == 1) {
        raw.open("recv_anticipada_raw.csv");
        raw << "size_bytes,trial,recv_seconds,recv_microseconds,work_iters_done\n";
        summary.open("recv_anticipada_summary.csv");
        summary << "size_bytes,avg_us,min_us,max_us,avg_MBps\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int sz : sizes) {
        std::vector<char> sbuf(sz, 7), rbuf(sz, 0);

        // Warmup
        for (int w = 0; w < WARMUP; ++w) {
            if (rank == 0) {
                MPI_Send(sbuf.data(), sz, MPI_BYTE, 1, TAG, MPI_COMM_WORLD);
            } else {
                MPI_Request rq;
                MPI_Irecv(rbuf.data(), sz, MPI_BYTE, 0, TAG, MPI_COMM_WORLD, &rq);
                int done = 0;
                while (!done) {
                    do_work(WORK_ITERS);
                    MPI_Test(&rq, &done, MPI_STATUS_IGNORE);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Mediciones
        std::vector<double> times;
        times.reserve(N);

        for (int t = 0; t < N; ++t) {
            MPI_Barrier(MPI_COMM_WORLD);

            if (rank == 0) {
                MPI_Send(sbuf.data(), sz, MPI_BYTE, 1, TAG, MPI_COMM_WORLD);
            } else {
                MPI_Request rq;
                MPI_Irecv(rbuf.data(), sz, MPI_BYTE, 0, TAG, MPI_COMM_WORLD, &rq);

                const double t0 = MPI_Wtime();
                int done = 0;
                long long work_iters_done = 0;
                while (!done) {
                    do_work(WORK_ITERS);
                    work_iters_done += WORK_ITERS;
                    MPI_Test(&rq, &done, MPI_STATUS_IGNORE);
                }
                const double t1 = MPI_Wtime();
                const double dt = t1 - t0;
                times.push_back(dt);

                if (raw.is_open()) {
                    raw << sz << "," << (t+1) << ","
                        << std::setprecision(9) << dt << ","
                        << static_cast<long long>(dt * 1e6) << ","
                        << work_iters_done << "\n";
                }
            }
        }

        if (rank == 1) {
            auto s = stats_of(times);
            double avg_MBps = (sz / 1e6) / s.avg;
            summary << sz << ","
                    << std::fixed << std::setprecision(3)
                    << (s.avg * 1e6) << ","
                    << (s.minv * 1e6) << ","
                    << (s.maxv * 1e6) << ","
                    << std::setprecision(2) << avg_MBps << "\n";

            std::cout << "[Rank 1] size=" << sz
                      << " bytes | avg=" << (s.avg*1e6) << " us"
                      << " | min=" << (s.minv*1e6) << " us"
                      << " | max=" << (s.maxv*1e6) << " us"
                      << " | MB/s=" << avg_MBps << "\n";
        }
    }

    if (rank == 1) {
        if (raw.is_open()) raw.close();
        if (summary.is_open()) summary.close();
        std::cout << ">> Resultados: recv_anticipada_raw.csv y recv_anticipada_summary.csv\n";
    }

    MPI_Finalize();
    return 0;
}
