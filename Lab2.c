// Compilar: gcc -O2 -fopenmp Lab2.c -o lab2 -lpthread -lm
// Correr: ./lab2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

// ------------------------------ Utilidades comunes ------------------------------
static inline double now_seconds(void) { return omp_get_wtime(); }

static void fill_array_double(double *a, size_t n) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) a[i] = (double)(i + 1);
}

static void check_malloc(void *p) {
    if (!p) { fprintf(stderr, "Error: memoria insuficiente.\n"); exit(1); }
}

static void banner(const char* title) {
    printf("\n==================== %s ====================\n", title);
}

// ------------------------------ Parte 1: N-elementos (sqrt) ------------------------------
// Codigo Secuencial
static double parte1_secuencial(double *a, size_t n) {
    double t0 = now_seconds();
    for (size_t i = 0; i < n; ++i) a[i] = sqrt(a[i]);
    return now_seconds() - t0;
}

// Codigo Paralelo
typedef enum { SCH_STATIC = 0, SCH_DYNAMIC = 1, SCH_GUIDED = 2 } SchedKind;

static double parte1_paralelo(double *a, size_t n, SchedKind kind, int chunk) {
    double t0 = now_seconds();
    if (kind == SCH_STATIC) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) a[i] = sqrt(a[i]);
    } else if (kind == SCH_DYNAMIC) {
        if (chunk <= 0) chunk = 1;
        #pragma omp parallel for schedule(dynamic, chunk)
        for (size_t i = 0; i < n; ++i) a[i] = sqrt(a[i]);
    } else {
        if (chunk <= 0) chunk = 1;
        #pragma omp parallel for schedule(guided, chunk)
        for (size_t i = 0; i < n; ++i) a[i] = sqrt(a[i]);
    }
    return now_seconds() - t0;
}

static const char* sched_name(SchedKind k) {
    return (k==SCH_STATIC) ? "static" : (k==SCH_DYNAMIC ? "dynamic" : "guided");
}

static void run_parte1(void) {
    banner("Parte 1: N-elementos (sqrt)");
    size_t N;
    int threads, sched_opt, chunk;
    printf("Tamaño del arreglo N (>100000): ");
    if (scanf("%zu", &N) != 1 || N < 100001) { printf("Usando N=1000000 por defecto.\n"); N = 1000000; }
    printf("Número de hilos (e.g., 1,2,4,8): ");
    if (scanf("%d", &threads) != 1 || threads < 1) { threads = 4; printf("Usando %d hilos.\n", threads); }
    printf("Scheduler (0=static, 1=dynamic, 2=guided): ");
    if (scanf("%d", &sched_opt) != 1 || sched_opt < 0 || sched_opt > 2) { sched_opt = 0; }
    printf("Chunk size (para dynamic/guided, >0 recomendado): ");
    if (scanf("%d", &chunk) != 1) { chunk = 0; }

    double *A = (double*)malloc(N * sizeof(double));
    check_malloc(A);

    // Secuencial
    fill_array_double(A, N);
    double t_seq = parte1_secuencial(A, N);

    // Paralelo
    fill_array_double(A, N);
    omp_set_num_threads(threads);
    double t_par = parte1_paralelo(A, N, (SchedKind)sched_opt, chunk);

    printf("\nResultados Parte 1 (N=%zu, threads=%d, sched=%s, chunk=%d)\n",
           N, threads, sched_name((SchedKind)sched_opt), chunk);
    printf("Tiempo secuencial: %.6f s\n", t_seq);
    printf("Tiempo paralelo  : %.6f s\n", t_par);
    printf("Speedup          : %.2fx\n", t_seq / t_par);

    free(A);
}

// ------------------------------ Parte 2: Suma (reduction/atomic/critical) ------------------------------
static double sum_reduction(const double *a, size_t n, double *out) {
    double t0 = now_seconds();
    double s = 0.0;
    #pragma omp parallel for reduction(+:s) schedule(static)
    for (size_t i = 0; i < n; ++i) s += a[i];
    *out = s;
    return now_seconds() - t0;
}

static double sum_atomic(const double *a, size_t n, double *out) {
    double t0 = now_seconds();
    double s = 0.0;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            #pragma omp atomic
            s += a[i];
        }
    }
    *out = s;
    return now_seconds() - t0;
}

static double sum_critical(const double *a, size_t n, double *out) {
    double t0 = now_seconds();
    double s = 0.0;
    #pragma omp parallel
    {
        double local = 0.0;
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) local += a[i];
        #pragma omp critical
        {
            s += local;
        }
    }
    *out = s;
    return now_seconds() - t0;
}

static void run_parte2(void) {
    banner("Parte 2: Coherencia y seguridad de hilos (sumatoria)");

    size_t N;
    int threads;
    printf("Tamaño del arreglo N (>100000): ");
    if (scanf("%zu", &N) != 1 || N < 100001) { N = 1000000; printf("Usando N=%zu.\n", N); }
    printf("Número de hilos (e.g., 1,2,4,8): ");
    if (scanf("%d", &threads) != 1 || threads < 1) { threads = 4; printf("Usando %d hilos.\n", threads); }

    double *A = (double*)malloc(N * sizeof(double));
    check_malloc(A);
    fill_array_double(A, N);

    omp_set_num_threads(threads);
    double s_red, s_atom, s_crit;
    double t_red = sum_reduction(A, N, &s_red);
    double t_atom = sum_atomic(A, N, &s_atom);
    double t_crit = sum_critical(A, N, &s_crit);

    printf("\nResultados Parte 2 (N=%zu, threads=%d)\n", N, threads);
    printf("Reduction:  suma=%0.3f  tiempo=%.6f s\n", s_red,  t_red);
    printf("Atomic:     suma=%0.3f  tiempo=%.6f s\n", s_atom, t_atom);
    printf("Critical:   suma=%0.3f  tiempo=%.6f s\n", s_crit, t_crit);

    double ref = s_red;
    int ok_atom = fabs(s_atom - ref) < 1e-6 * fabs(ref);
    int ok_crit = fabs(s_crit - ref) < 1e-6 * fabs(ref);
    printf("Verificación: atomic=%s, critical=%s\n",
           ok_atom ? "OK" : "MAL", ok_crit ? "OK" : "MAL");

    free(A);
}

// ------------------------------ Parte 3: Producer-Consumer (Pthreads) ------------------------------
typedef struct {
    int *buf;
    int capacity;
    int count;
    int in, out;
    pthread_mutex_t mtx;
    pthread_cond_t not_full;
    pthread_cond_t not_empty;

    long total_to_produce;
    long produced_so_far;
    long consumed_so_far;

    long processed_sum;
} BoundedBuffer;

static void bb_init(BoundedBuffer *q, int capacity, long total_to_produce) {
    q->buf = (int*)malloc(sizeof(int)*capacity);
    check_malloc(q->buf);
    q->capacity = capacity;
    q->count = 0;
    q->in = 0; q->out = 0;
    pthread_mutex_init(&q->mtx, NULL);
    pthread_cond_init(&q->not_full, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    q->total_to_produce = total_to_produce;
    q->produced_so_far = 0;
    q->consumed_so_far = 0;
    q->processed_sum = 0;
}

static void bb_destroy(BoundedBuffer *q) {
    pthread_mutex_destroy(&q->mtx);
    pthread_cond_destroy(&q->not_full);
    pthread_cond_destroy(&q->not_empty);
    free(q->buf);
}

static void bb_push(BoundedBuffer *q, int value) {
    pthread_mutex_lock(&q->mtx);
    while (q->count == q->capacity) {
        pthread_cond_wait(&q->not_full, &q->mtx);
    }
    q->buf[q->in] = value;
    q->in = (q->in + 1) % q->capacity;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mtx);
}

static int bb_pop(BoundedBuffer *q, int *value) {
    pthread_mutex_lock(&q->mtx);
    while (q->count == 0) {
        if (q->consumed_so_far >= q->total_to_produce &&
            q->produced_so_far >= q->total_to_produce) {
            pthread_mutex_unlock(&q->mtx);
            return 0;
        }
        pthread_cond_wait(&q->not_empty, &q->mtx);
    }
    *value = q->buf[q->out];
    q->out = (q->out + 1) % q->capacity;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mtx);
    return 1;
}

typedef struct {
    BoundedBuffer *q;
    int id;
    long quota;
} ProducerArgs;

typedef struct {
    BoundedBuffer *q;
    int id;
} ConsumerArgs;

static void* producer_thread(void *arg) {
    ProducerArgs *pa = (ProducerArgs*)arg;
    BoundedBuffer *q = pa->q;
    for (long i = 0; i < pa->quota; ++i) {
        int value = pa->id * 1000000 + (int)i;
        bb_push(q, value);

        pthread_mutex_lock(&q->mtx);
        q->produced_so_far++;
        pthread_mutex_unlock(&q->mtx);
    }
    return NULL;
}

static void* consumer_thread(void *arg) {
    ConsumerArgs *ca = (ConsumerArgs*)arg;
    BoundedBuffer *q = ca->q;
    int value;
    while (1) {
        if (!bb_pop(q, &value)) break;
        pthread_mutex_lock(&q->mtx);
        q->processed_sum += value;
        q->consumed_so_far++;
        pthread_mutex_unlock(&q->mtx);
    }
    return NULL;
}

static void run_parte3(void) {
    banner("Parte 3: Producer-Consumer (buffer acotado, Pthreads)");

    int capacity, nprod, ncons;
    long total;
    printf("Tamaño de buffer (capacidad, p. ej. 8, 32, 128): ");
    if (scanf("%d", &capacity) != 1 || capacity < 1) { capacity = 32; printf("Usando %d.\n", capacity); }
    printf("Total a producir (N): ");
    if (scanf("%ld", &total) != 1 || total < 1) { total = 100000; printf("Usando N=%ld.\n", total); }
    printf("# productores: ");
    if (scanf("%d", &nprod) != 1 || nprod < 1) { nprod = 2; printf("Usando %d productores.\n", nprod); }
    printf("# consumidores: ");
    if (scanf("%d", &ncons) != 1 || ncons < 1) { ncons = 2; printf("Usando %d consumidores.\n", ncons); }

    BoundedBuffer q;
    bb_init(&q, capacity, total);

    pthread_t *tp = (pthread_t*)malloc(sizeof(pthread_t)*nprod);
    pthread_t *tc = (pthread_t*)malloc(sizeof(pthread_t)*ncons);
    check_malloc(tp); check_malloc(tc);

    ProducerArgs *pargs = (ProducerArgs*)malloc(sizeof(ProducerArgs)*nprod);
    ConsumerArgs *cargs = (ConsumerArgs*)malloc(sizeof(ConsumerArgs)*ncons);
    check_malloc(pargs); check_malloc(cargs);

    long base = total / nprod, rem = total % nprod;
    for (int i = 0; i < nprod; ++i) {
        pargs[i].q = &q;
        pargs[i].id = i;
        pargs[i].quota = base + (i < rem ? 1 : 0);
    }
    for (int i = 0; i < ncons; ++i) {
        cargs[i].q = &q;
        cargs[i].id = i;
    }

    double t0 = now_seconds();

    for (int i = 0; i < nprod; ++i) pthread_create(&tp[i], NULL, producer_thread, &pargs[i]);
    for (int i = 0; i < ncons; ++i) pthread_create(&tc[i], NULL, consumer_thread, &cargs[i]);

    for (int i = 0; i < nprod; ++i) pthread_join(tp[i], NULL);

    pthread_mutex_lock(&q.mtx);
    pthread_cond_broadcast(&q.not_empty);
    pthread_mutex_unlock(&q.mtx);

    for (int i = 0; i < ncons; ++i) pthread_join(tc[i], NULL);

    double elapsed = now_seconds() - t0;

    printf("\nResultados Parte 3\n");
    printf("Buffer=%d | N=%ld | Prod=%d | Cons=%d\n", q.capacity, q.total_to_produce, nprod, ncons);
    printf("Producidos=%ld | Consumidos=%ld | Tiempo=%.6f s | Suma procesada=%ld\n",
           q.produced_so_far, q.consumed_so_far, elapsed, q.processed_sum);


    free(tp); free(tc); free(pargs); free(cargs);
    bb_destroy(&q);
}

// Menú principal 
static void menu(void) {
    int opt = -1;
    while (opt != 0) {
        printf("\n==== LAB 2 - Menu ====\n");
        printf("1) Parte 1: N-elementos (sqrt) secuencial vs paralelo (scheduler/chunk)\n");
        printf("2) Parte 2: Suma con reduction vs atomic vs critical\n");
        printf("3) Parte 3: Producer-Consumer (buffer acotado) con Pthreads\n");
        printf("0) Salir\n");
        printf("Opción: ");
        if (scanf("%d", &opt) != 1) { while (getchar()!='\n'); opt = -1; continue; }
        if      (opt == 1) run_parte1();
        else if (opt == 2) run_parte2();
        else if (opt == 3) run_parte3();
        else if (opt == 0) printf("Saliendo...\n");
        else printf("Opción inválida.\n");
    }
}

int main(void) {
    srand((unsigned)time(NULL));
    menu();
    return 0;
}
