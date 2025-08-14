// Encabezado (para no olvidar)
// Compilar (GCC):   gcc -O3 -fopenmp Lab1.c -o Lab1 -lm
// Ejecutar:         ./Lab1 --help   (para ver opciones)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>

// ==========================
// Utilidades de medición
// ==========================

typedef struct { double secs; long double result; } run_out;

static inline double now() { return omp_get_wtime(); }

static inline void touch_array_int(int *a, size_t n) {
    volatile int sink = 0;
    for (size_t i = 0; i < n; ++i) sink ^= a[i];
}

// ==========================
// Parte 1
// Sumar enteros
// ==========================

#define N_PART1 100000000u

static run_out part1_seq_sum(size_t N) {
    double t0 = now();
    unsigned long long acc = 0ULL;
    for (size_t i = 0; i < N; ++i) acc += (unsigned long long)i;
    double t1 = now();
    run_out o = { .secs = t1 - t0, .result = (long double)acc };
    return o;
}

static run_out part1_par_no_reduction_critical(size_t N) {
    // Paralelo SIN reduction — usando critical
    unsigned long long acc = 0ULL;
    double t0 = now();
    #pragma omp parallel
    {
        unsigned long long local = 0ULL;
        #pragma omp for nowait
        for (size_t i = 0; i < N; ++i) local += (unsigned long long)i;
        #pragma omp critical
        acc += local;
    }
    double t1 = now();
    run_out o = { .secs = t1 - t0, .result = (long double)acc };
    return o;
}

static run_out part1_par_no_reduction_atomic(size_t N) {
    // Paralelo SIN reduction — usando atomic
    unsigned long long acc = 0ULL;
    double t0 = now();
    #pragma omp parallel
    {
        unsigned long long local = 0ULL;
        #pragma omp for nowait
        for (size_t i = 0; i < N; ++i) local += (unsigned long long)i;
        #pragma omp atomic
        acc += local;
    }
    double t1 = now();
    run_out o = { .secs = t1 - t0, .result = (long double)acc };
    return o;
}

static run_out part1_par_with_reduction(size_t N) {
    unsigned long long acc = 0ULL;
    double t0 = now();
    #pragma omp parallel for reduction(+:acc)
    for (size_t i = 0; i < N; ++i) acc += (unsigned long long)i;
    double t1 = now();
    run_out o = { .secs = t1 - t0, .result = (long double)acc };
    return o;
}

// ==========================
// Parte 2
// Trabajo desigual
// ==========================

static long double do_flops(size_t iters) {
    long double acc = 0.0L;
    for (size_t k = 0; k < iters; ++k) {
        long double x = (long double)k + 1.0L;
        acc += sqrtl(x) + sinl(x * 1e-6L);
    }
    return acc;
}

static run_out part2_uneven(size_t N, size_t inner, const char *sched, int chunk) {
    omp_sched_t kind = omp_sched_static;

    // Configurar schedule
    if (sched && strcmp(sched, "dynamic") == 0) {
        omp_set_schedule(omp_sched_dynamic, chunk > 0 ? chunk : 1);
        kind = omp_sched_dynamic;
    } else if (sched && strcmp(sched, "guided") == 0) {
        omp_set_schedule(omp_sched_guided, chunk > 0 ? chunk : 1);
        kind = omp_sched_guided;
    } else {
        omp_set_schedule(omp_sched_static, chunk > 0 ? chunk : 0);
        kind = omp_sched_static;
    }

    long double guard = 0.0L;
    double t0 = now();

    #pragma omp parallel
    {
        long double local = 0.0L;
        #pragma omp for schedule(runtime)
        for (long long i = 0; i < (long long)N; ++i) {
            size_t weight = (size_t)(i % 32);
            local += do_flops((size_t)inner * (1 + weight));
        }
        #pragma omp atomic
        guard += local;
    }

    double t1 = now();
    omp_sched_t k; int c;
    omp_get_schedule(&k, &c);

    run_out o = { .secs = t1 - t0, .result = guard + k + c };
    return o;
}

// ==========================
// Parte 3
// Contar pares
// ==========================

#define N_PART3 10000000u

static void fill_array_seq_int(int *a, size_t n) {
    #pragma omp parallel for
    for (long long i = 0; i < (long long)n; ++i) a[i] = (int)i;
}

static run_out count_even_seq(const int *a, size_t n) {
    double t0 = now();
    unsigned long long c = 0ULL;
    for (size_t i = 0; i < n; ++i) if ((a[i] & 1) == 0) ++c;
    double t1 = now();
    run_out o = { .secs = t1 - t0, .result = (long double)c };
    return o;
}

static run_out count_even_par_critical(const int *a, size_t n) {
    double t0 = now();
    unsigned long long c = 0ULL;
    #pragma omp parallel
    {
        unsigned long long local = 0ULL;
        #pragma omp for nowait
        for (long long i = 0; i < (long long)n; ++i) if ((a[i] & 1) == 0) ++local;
        #pragma omp critical
        c += local;
    }
    double t1 = now();
    run_out o = { .secs = t1 - t0, .result = (long double)c };
    return o;
}

static run_out count_even_par_reduction(const int *a, size_t n) {
    double t0 = now();
    unsigned long long c = 0ULL;
    #pragma omp parallel for reduction(+:c)
    for (long long i = 0; i < (long long)n; ++i) if ((a[i] & 1) == 0) ++c;
    double t1 = now();
    run_out o = { .secs = t1 - t0, .result = (long double)c };
    return o;
}

// ==========================
// Parte 4
// Mejora de codigos anteriores
// ==========================

static void sweep_threads_and_report(const char *kernel, int min_threads, int max_threads) {
    if (min_threads < 1) min_threads = 1;
    if (max_threads < min_threads) max_threads = min_threads;

    printf("kernel,threads,seconds,speedup,efficiency,extra\n");

    // Medir base para referencia de los tiempos
    double base = 0.0; long double extra = 0.0L;

    if (strcmp(kernel, "part1_sum") == 0) {
        run_out b = part1_seq_sum(N_PART1);
        base = b.secs; extra = b.result;
        printf("# base_result=%Lg\n", extra);
        for (int t = min_threads; t <= max_threads; t *= 2) {
            omp_set_num_threads(t);
            run_out r = part1_par_with_reduction(N_PART1);
            double speed = base / r.secs;
            double eff = speed / t;
            printf("%s,%d,%.6f,%.3f,%.3f,%Lg\n", kernel, t, r.secs, speed, eff, r.result);
            if (t == 1 && min_threads == 1) base = r.secs;
            if (t == 0) break;
        }
    } else if (strcmp(kernel, "part2_uneven_static") == 0 || strcmp(kernel, "part2_uneven_dynamic") == 0 || strcmp(kernel, "part2_uneven_guided") == 0) {
        const char *sched = strstr(kernel, "static") ? "static" : (strstr(kernel, "dynamic") ? "dynamic" : "guided");
        size_t N = 200000; size_t inner = 32; int chunk = 0;
        // como base forzamos 1 hilo
        omp_set_num_threads(1);
        run_out b = part2_uneven(N, inner, sched, chunk);
        base = b.secs; extra = b.result;
        printf("# base_result=%Lg\n", extra);
        for (int t = min_threads; t <= max_threads; t *= 2) {
            omp_set_num_threads(t);
            run_out r = part2_uneven(N, inner, sched, chunk);
            double speed = base / r.secs;
            double eff = speed / t;
            printf("%s,%d,%.6f,%.3f,%.3f,%Lg\n", kernel, t, r.secs, speed, eff, r.result);
        }
    } else if (strcmp(kernel, "part3_even") == 0) {
        int *a = (int*) aligned_alloc(64, sizeof(int) * (size_t)N_PART3);
        fill_array_seq_int(a, N_PART3);
        run_out b = count_even_seq(a, N_PART3);
        base = b.secs; extra = b.result;
        printf("# base_result=%Lg\n", extra);
        for (int t = min_threads; t <= max_threads; t *= 2) {
            omp_set_num_threads(t);
            run_out r = count_even_par_reduction(a, N_PART3);
            double speed = base / r.secs;
            double eff = speed / t;
            printf("%s,%d,%.6f,%.3f,%.3f,%Lg\n", kernel, t, r.secs, speed, eff, r.result);
        }
        free(a);
    } else {
        fprintf(stderr, "[sweep] kernel desconocido: %s\n", kernel);
    }
}

// ==========================
// Menú
// ==========================

static void usage(const char *prog) {
    printf("\nPara ejecutar el programa: %s <comando> [opciones]\n\n", prog);
    printf("Comandos principales (numero de la parte):\n");
    printf("  part1          — Suma 0..(N-1) con variantes (seq, no-reduction, reduction)\n");
    printf("  part2          — Trabajo desigual con schedules (static/dynamic/guided)\n");
    printf("  part3          — Conteo de pares en arreglo (seq, critical, reduction)\n");
    printf("  sweep          — Barrido de hilos y salida CSV para gráficas (parte 4 usando codigo de cualquier parte anterior)\n\n");

    printf("Opciones comunes:\n");
    printf("  --threads T       Fija omp_set_num_threads(T) (por defecto, variable del entorno OMP_NUM_THREADS o la del sistema).\n");
    printf("  --repeats R       Repetir medición R veces y reportar el mejor tiempo.\n");

    printf("\npart1 opciones:\n");
    printf("  --variant seq | critical | atomic | reduction\n");

    printf("\npart2 opciones:\n");
    printf("  --N N             Tamaño del bucle externo (default 200000)\n");
    printf("  --inner K         Carga base por iteración (default 32)\n");
    printf("  --schedule S      S en {static,dynamic,guided} (default static)\n");
    printf("  --chunk C         Chunk size (entero, default 0 -> decide OpenMP)\n");

    printf("\npart3 no tiene opciones adicionales.\n");

    printf("\nsweep opciones:\n");
    printf("  --kernel K        K en {part1_sum, part2_uneven_static, part2_uneven_dynamic, part2_uneven_guided, part3_even}\n");
    printf("  --min T           Hilos mínimos (default 1)\n");
    printf("  --max T           Hilos máximos (default 8)\n\n");
}

static double best_of(int repeats, run_out (*fn)(void*), void *arg, long double *res_out) {
    double best = 1e100; long double best_res = 0.0L;
    for (int r = 0; r < repeats; ++r) {
        run_out o = fn(arg);
        if (o.secs < best) { best = o.secs; best_res = o.result; }
    }
    if (res_out) *res_out = best_res;
    return best;
}

// main y wrappers

typedef struct { size_t N; } arg_sz;
typedef struct { size_t N; size_t inner; const char* sched; int chunk; } arg_p2;
typedef struct { const int *a; size_t n; int variant; } arg_p3;

static run_out wrap_part1_seq(void *p){ arg_sz *a=(arg_sz*)p; return part1_seq_sum(a->N);} 
static run_out wrap_part1_crit(void *p){ arg_sz *a=(arg_sz*)p; return part1_par_no_reduction_critical(a->N);} 
static run_out wrap_part1_atom(void *p){ arg_sz *a=(arg_sz*)p; return part1_par_no_reduction_atomic(a->N);} 
static run_out wrap_part1_red (void *p){ arg_sz *a=(arg_sz*)p; return part1_par_with_reduction(a->N);} 

static run_out wrap_part2(void *p){ arg_p2 *a=(arg_p2*)p; return part2_uneven(a->N,a->inner,a->sched,a->chunk);} 

static run_out wrap_part3_seq(void *p){ arg_p3 *a=(arg_p3*)p; return count_even_seq(a->a,a->n);} 
static run_out wrap_part3_crit(void *p){ arg_p3 *a=(arg_p3*)p; return count_even_par_critical(a->a,a->n);} 
static run_out wrap_part3_red (void *p){ arg_p3 *a=(arg_p3*)p; return count_even_par_reduction(a->a,a->n);} 

int main(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0) { usage(argv[0]); return 0; }

    const char *cmd = argv[1];
    int threads = -1;
    int repeats = 1;

    const char *variant = "reduction"; // default razonable

    size_t p2N = 200000; size_t p2inner = 32; const char *p2sched = "static"; int p2chunk = 0;

    const char *kernel = "part1_sum"; int tmin = 1, tmax = 8;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--threads") == 0 && i+1 < argc) { threads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--repeats") == 0 && i+1 < argc) { repeats = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--variant") == 0 && i+1 < argc) { variant = argv[++i]; }
        else if (strcmp(argv[i], "--N") == 0 && i+1 < argc) { p2N = (size_t)atoll(argv[++i]); }
        else if (strcmp(argv[i], "--inner") == 0 && i+1 < argc) { p2inner = (size_t)atoll(argv[++i]); }
        else if (strcmp(argv[i], "--schedule") == 0 && i+1 < argc) { p2sched = argv[++i]; }
        else if (strcmp(argv[i], "--chunk") == 0 && i+1 < argc) { p2chunk = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--kernel") == 0 && i+1 < argc) { kernel = argv[++i]; }
        else if (strcmp(argv[i], "--min") == 0 && i+1 < argc) { tmin = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--max") == 0 && i+1 < argc) { tmax = atoi(argv[++i]); }
        else { }
    }

    if (threads > 0) omp_set_num_threads(threads);

    if (strcmp(cmd, "part1") == 0) {
        arg_sz a = { .N = N_PART1 };
        run_out r; double secs; long double outv;
        if (strcmp(variant, "seq") == 0) {
            secs = best_of(repeats, wrap_part1_seq, &a, &outv);
        } else if (strcmp(variant, "critical") == 0) {
            secs = best_of(repeats, wrap_part1_crit, &a, &outv);
        } else if (strcmp(variant, "atomic") == 0) {
            secs = best_of(repeats, wrap_part1_atom, &a, &outv);
        } else if (strcmp(variant, "reduction") == 0) {
            secs = best_of(repeats, wrap_part1_red, &a, &outv);
        } else {
            fprintf(stderr, "variant desconocida: %s\n", variant); return 1;
        }
        printf("PART1 variant=%s N=%u time=%.6f result=%Lg\n", variant, N_PART1, secs, outv);
        return 0;
    }

    if (strcmp(cmd, "part2") == 0) {
        arg_p2 a = { .N = p2N, .inner = p2inner, .sched = p2sched, .chunk = p2chunk };
        long double outv; double secs = best_of(repeats, wrap_part2, &a, &outv);
        printf("PART2 schedule=%s chunk=%d N=%zu inner=%zu time=%.6f guard=%Lg\n", p2sched, p2chunk, p2N, p2inner, secs, outv);
        return 0;
    }

    if (strcmp(cmd, "part3") == 0) {
        int *a = (int*) aligned_alloc(64, sizeof(int) * (size_t)N_PART3);
        fill_array_seq_int(a, N_PART3);
        run_out b = count_even_seq(a, N_PART3);
        printf("PART3 baseline(seq) N=%u time=%.6f evens=%Lg\n", N_PART3, b.secs, b.result);

        // critical
        run_out c = count_even_par_critical(a, N_PART3);
        int Tcrit = omp_get_max_threads();
        double speed_c = b.secs / c.secs;
        double eff_c = speed_c / Tcrit;
        printf("PART3 critical time=%.6f speedup=%.3f efficiency=%.3f evens=%Lg\n", c.secs, speed_c, eff_c, c.result);

        // reduction
        run_out d = count_even_par_reduction(a, N_PART3);
        int Tred = omp_get_max_threads();
        double speed_r = b.secs / d.secs;
        double eff_r = speed_r / Tred;
        printf("PART3 reduction time=%.6f speedup=%.3f efficiency=%.3f evens=%Lg\n", d.secs, speed_r, eff_r, d.result);

        free(a);
        return 0;
    }

    if (strcmp(cmd, "sweep") == 0) {
        sweep_threads_and_report(kernel, tmin, tmax);
        return 0;
    }

    usage(argv[0]);
    return 0;
}
