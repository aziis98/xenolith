#define XE_TEST_ALLOC
#define XE_TEST_OUTPUT_COUNT
#define XE_TEST_PREFILL_TILE
static int xe_test_prefill_tile_rows;
#include "../xenolith.c"

static double session_now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static double session_median(const double *values, int count) {
    double sorted[15];
    memcpy(sorted, values, (size_t)count * sizeof(*sorted));
    for (int i = 1; i < count; i++) {
        double value = sorted[i];
        int j = i;
        while (j > 0 && sorted[j - 1] > value) {
            sorted[j] = sorted[j - 1];
            j--;
        }
        sorted[j] = value;
    }
    return sorted[count / 2];
}

static int session_tile_bench(xe_engine *e, const int32_t *tokens, int rows,
                              int rounds) {
    if (rounds < 3 || rounds > 15 || !(rounds & 1))
        xe_fatal("tile benchmark rounds must be odd from 3 through 15");
    double times[2][15];
    for (int variant = 0; variant < 2; variant++) {
        xe_test_prefill_tile_rows = variant ? 16 : 32;
        xe_session *s = xe_session_new(e);
        xe_prefill_batch_run(s, tokens, rows, 0, 0);
        xe_session_free(s);
    }
    for (int round = 0; round < rounds; round++) {
        for (int position = 0; position < 2; position++) {
            int variant = (round + position) & 1;
            xe_test_prefill_tile_rows = variant ? 16 : 32;
            xe_session *s = xe_session_new(e);
            double start = session_now();
            xe_prefill_batch_run(s, tokens, rows, 0, 0);
            times[variant][round] = session_now() - start;
            xe_session_free(s);
        }
        printf("prefill-tile: M%d round %d m32 %.6f s m16 %.6f s\n",
               rows, round + 1, times[0][round], times[1][round]);
    }
    double m32 = session_median(times[0], rounds);
    double m16 = session_median(times[1], rounds);
    printf("prefill-tile: M%d median m32 %.6f s m16 %.6f s speedup %.6fx\n",
           rows, m32, m16, m32 / m16);
    return 0;
}

static int session_crossover_bench(xe_engine *e, const int32_t *tokens,
                                   int rows, int rounds) {
    if (rounds < 3 || rounds > 15 || !(rounds & 1))
        xe_fatal("crossover benchmark rounds must be odd from 3 through 15");
    double times[2][15];
    xe_test_prefill_tile_rows = 0;
    for (int round = -1; round < rounds; round++) {
        for (int position = 0; position < 2; position++) {
            int variant = ((round < 0 ? 0 : round) + position) & 1;
            xe_session *s = xe_session_new(e);
            double start = session_now();
            if (variant) {
                xe_prefill_batch_run(s, tokens, rows, 0, 1);
            } else {
                for (int row = 0; row < rows; row++)
                    xe_decode_token_mode(
                        s, tokens[row], row, XE_MOE_GENERIC_BATCHED, 1,
                        XE_SOFTCAP_SECOND_LOOP, row == rows - 1);
            }
            double seconds = session_now() - start;
            xe_session_free(s);
            if (round >= 0) times[variant][round] = seconds;
        }
        if (round >= 0)
            printf("prefill-crossover: M%d round %d cpu %.6f s gpu %.6f s\n",
                   rows, round + 1, times[0][round], times[1][round]);
    }
    double cpu = session_median(times[0], rounds);
    double gpu = session_median(times[1], rounds);
    printf("prefill-crossover: M%d median cpu %.6f s gpu %.6f s speedup %.6fx\n",
           rows, cpu, gpu, cpu / gpu);
    return 0;
}

static double session_rel(const float *a, const float *b, size_t n) {
    double error = 0.0;
    double reference = 0.0;
    for (size_t i = 0; i < n; i++) {
        double difference = (double)a[i] - b[i];
        error += difference * difference;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static double session_half_rel(const _Float16 *a, const _Float16 *b,
                               size_t n) {
    double error = 0.0;
    double reference = 0.0;
    for (size_t i = 0; i < n; i++) {
        double difference = (double)a[i] - (double)b[i];
        error += difference * difference;
        reference += (double)b[i] * (double)b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static int session_argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static void session_tokens_read(const char *path, int32_t *tokens, int count) {
    FILE *f = fopen(path, "rb");
    if (!f) xe_fatal("%s: %s", path, strerror(errno));
    for (int i = 0; i < count; i++) {
        long value;
        if (fscanf(f, " %ld", &value) != 1)
            xe_fatal("%s: expected %d tokens", path, count);
        tokens[i] = (int32_t)value;
        int separator = fgetc(f);
        if (i + 1 < count && separator != ',')
            xe_fatal("%s: bad token separator", path);
    }
    fclose(f);
}

static size_t session_kv_elements(int rows) {
    size_t elements = 0;
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        int dimension = XE_IS_GLOBAL(layer)
            ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
        int heads = XE_IS_GLOBAL(layer)
            ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
        elements += (size_t)heads * rows * dimension;
    }
    return elements;
}

static void session_kv_copy(xe_session *s, int rows, _Float16 *k,
                            _Float16 *v) {
    size_t offset = 0;
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        int global = XE_IS_GLOBAL(layer);
        int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
        int heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
        int capacity = global ? XE_CTX : XE_SWA_WINDOW;
        const _Float16 *source_k = xe_kv_layer_ptr(s, layer, 0);
        const _Float16 *source_v = xe_kv_layer_ptr(s, layer, 1);
        for (int head = 0; head < heads; head++) {
            size_t count = (size_t)rows * dimension;
            memcpy(k + offset, source_k + (size_t)head * capacity * dimension,
                   count * sizeof(*k));
            memcpy(v + offset, source_v + (size_t)head * capacity * dimension,
                   count * sizeof(*v));
            offset += count;
        }
    }
}

static int session_gpu_split(xe_engine *e, int32_t *tokens, int rows,
                             int split, int cpu_tail) {
    if (split < 1 || split >= rows)
        xe_fatal("GPU split must be inside the batch");
    size_t kv_elements = session_kv_elements(rows);
    _Float16 *direct_k = xe_alloc(
        NULL, kv_elements * sizeof(*direct_k), XE_MEM_HOST);
    _Float16 *direct_v = xe_alloc(
        NULL, kv_elements * sizeof(*direct_v), XE_MEM_HOST);
    _Float16 *split_k = xe_alloc(
        NULL, kv_elements * sizeof(*split_k), XE_MEM_HOST);
    _Float16 *split_v = xe_alloc(
        NULL, kv_elements * sizeof(*split_v), XE_MEM_HOST);
    float *direct_hidden = xe_alloc(
        NULL, XE_EMBD * sizeof(*direct_hidden), XE_MEM_HOST);
    float *direct_logits = xe_alloc(
        NULL, XE_VOCAB * sizeof(*direct_logits), XE_MEM_HOST);
    float *direct_next = xe_alloc(
        NULL, XE_VOCAB * sizeof(*direct_next), XE_MEM_HOST);
    xe_test_prefill_tile_rows = 0;
    xe_session *direct = xe_session_new(e);
    double direct_start = session_now();
    xe_prefill_batch_run(direct, tokens, rows, 0, 1);
    double direct_seconds = session_now() - direct_start;
    memcpy(direct_hidden, direct->hidden, XE_EMBD * sizeof(*direct_hidden));
    memcpy(direct_logits, direct->logits, XE_VOCAB * sizeof(*direct_logits));
    session_kv_copy(direct, rows, direct_k, direct_v);
    int32_t continuation[8];
    for (int step = 0; step < 8; step++) {
        continuation[step] = session_argmax(direct->logits, XE_VOCAB);
        xe_decode_token(direct, continuation[step], rows + step);
        if (step == 0)
            memcpy(direct_next, direct->logits,
                   XE_VOCAB * sizeof(*direct_next));
    }
    xe_session_reset(direct);
    xe_session *partitioned = direct;
    double split_start = session_now();
    xe_prefill_batch_run(partitioned, tokens, split, 0, 0);
    if (cpu_tail) {
        for (int pos = split; pos < rows; pos++)
            xe_decode_token_mode(partitioned, tokens[pos], pos,
                                 XE_MOE_GENERIC_BATCHED, 1,
                                 XE_SOFTCAP_SECOND_LOOP, pos == rows - 1);
    } else {
        xe_prefill_batch_run(partitioned, tokens + split, rows - split,
                             split, 1);
    }
    double split_seconds = session_now() - split_start;
    session_kv_copy(partitioned, rows, split_k, split_v);
    double hidden_error = session_rel(
        partitioned->hidden, direct_hidden, XE_EMBD);
    double logits_error = session_rel(
        partitioned->logits, direct_logits, XE_VOCAB);
    double k_error = session_half_rel(split_k, direct_k, kv_elements);
    double v_error = session_half_rel(split_v, direct_v, kv_elements);
    int top_ok = session_argmax(partitioned->logits, XE_VOCAB)
                 == session_argmax(direct_logits, XE_VOCAB);
    size_t kv_mismatches = 0;
    for (size_t i = 0; i < kv_elements; i++)
        kv_mismatches += split_k[i] != direct_k[i]
                         || split_v[i] != direct_v[i];
    int greedy_mismatches = 0;
    int first_greedy_mismatch = -1;
    double next_error = 0.0;
    int next_top_ok = 0;
    for (int step = 0; step < 8; step++) {
        int split_token = session_argmax(partitioned->logits, XE_VOCAB);
        if (continuation[step] != split_token) {
            greedy_mismatches++;
            if (first_greedy_mismatch < 0) first_greedy_mismatch = step;
        }
        xe_decode_token(partitioned, continuation[step], rows + step);
        if (step == 0) {
            next_error = session_rel(partitioned->logits, direct_next,
                                     XE_VOCAB);
            next_top_ok = session_argmax(partitioned->logits, XE_VOCAB)
                          == session_argmax(direct_next, XE_VOCAB);
        }
    }
    int ok = hidden_error < 0.1 && logits_error < 0.1 && top_ok
             && k_error < 0.15 && v_error < 0.15
             && next_error < 0.1 && next_top_ok
             && greedy_mismatches == 0;
    printf("prefill-%s-split: M%d %d+%d direct %.6f s split %.6f s hidden %.3e logits %.3e top %s KV %.3e/%.3e mismatches %zu/%zu handoff %.3e top %s greedy %d/8 first %d %s\n",
           cpu_tail ? "hybrid" : "gpu", rows, split, rows - split,
           direct_seconds, split_seconds,
           hidden_error, logits_error, top_ok ? "same" : "different",
           k_error, v_error, kv_mismatches, 2 * kv_elements, next_error,
           next_top_ok ? "same" : "different", greedy_mismatches,
           first_greedy_mismatch, ok ? "PASS" : "FAIL");
    xe_session_free(partitioned);
    xe_free(NULL, direct_next, XE_MEM_HOST);
    xe_free(NULL, direct_logits, XE_MEM_HOST);
    xe_free(NULL, direct_hidden, XE_MEM_HOST);
    xe_free(NULL, split_v, XE_MEM_HOST);
    xe_free(NULL, split_k, XE_MEM_HOST);
    xe_free(NULL, direct_v, XE_MEM_HOST);
    xe_free(NULL, direct_k, XE_MEM_HOST);
    return ok ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [rows [batch_start [mode-or-ids [args...]]]]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int rows = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 32;
    int batch_start = argc > 3 ? (int)strtol(argv[3], NULL, 10) : 0;
    int split_mode = argc > 4
                     && (!strcmp(argv[4], "split")
                         || !strcmp(argv[4], "hybrid"));
    int hybrid_mode = argc > 4 && !strcmp(argv[4], "hybrid");
    int cpu_tail = argc > 5 && !split_mode
                   ? (int)strtol(argv[5], NULL, 10) : 0;
    if (rows < 1 || rows > 512)
        xe_fatal("prefill session test supports M1 through M512");
    if (batch_start < 0 || cpu_tail < 0
        || batch_start + rows + cpu_tail > 1024)
        xe_fatal("prefill session batch start out of range");
    int total = batch_start + rows + cpu_tail;
    int32_t tokens[1033];
    if (split_mode && argc > 6)
        session_tokens_read(argv[6], tokens, total);
    else if (argc > 4 && strcmp(argv[4], "bench")
             && strcmp(argv[4], "crossover") && !split_mode)
        session_tokens_read(argv[4], tokens, total);
    else for (int row = 0; row < total; row++) tokens[row] = 2 + row;
    xe_engine *e = xe_engine_open(model);
    if (argc > 4 && !strcmp(argv[4], "bench")) {
        int rounds = argc > 5 ? (int)strtol(argv[5], NULL, 10) : 5;
        int result = session_tile_bench(e, tokens, rows, rounds);
        xe_engine_close(e);
        return result;
    }
    if (argc > 4 && !strcmp(argv[4], "crossover")) {
        int rounds = argc > 5 ? (int)strtol(argv[5], NULL, 10) : 5;
        int result = session_crossover_bench(e, tokens, rows, rounds);
        xe_engine_close(e);
        return result;
    }
    if (split_mode) {
        int split = argc > 5 ? (int)strtol(argv[5], NULL, 10) : rows - 1;
        int result = session_gpu_split(e, tokens, rows, split, hybrid_mode);
        xe_engine_close(e);
        return result;
    }
    size_t kv_elements = session_kv_elements(total);
    _Float16 *reference_k = xe_alloc(
        NULL, kv_elements * sizeof(*reference_k), XE_MEM_HOST);
    _Float16 *reference_v = xe_alloc(
        NULL, kv_elements * sizeof(*reference_v), XE_MEM_HOST);
    float *reference_hidden = xe_alloc(
        NULL, XE_EMBD * sizeof(*reference_hidden), XE_MEM_HOST);
    float *reference_logits = xe_alloc(
        NULL, XE_VOCAB * sizeof(*reference_logits), XE_MEM_HOST);
    float *reference_next = xe_alloc(
        NULL, XE_VOCAB * sizeof(*reference_next), XE_MEM_HOST);

    xe_session *cpu = xe_session_new(e);
    double cpu_start = session_now();
    for (int row = 0; row < total; row++)
        xe_decode_token_mode(cpu, tokens[row], row, XE_MOE_GENERIC_BATCHED, 1,
                             XE_SOFTCAP_SECOND_LOOP, row == total - 1);
    double cpu_seconds = session_now() - cpu_start;
    memcpy(reference_hidden, cpu->hidden, XE_EMBD * sizeof(*reference_hidden));
    memcpy(reference_logits, cpu->logits,
           XE_VOCAB * sizeof(*reference_logits));
    session_kv_copy(cpu, total, reference_k, reference_v);
    int32_t continuation[8];
    for (int step = 0; step < 8; step++) {
        continuation[step] = session_argmax(cpu->logits, XE_VOCAB);
        xe_decode_token(cpu, continuation[step], total + step);
        if (step == 0)
            memcpy(reference_next, cpu->logits,
                   XE_VOCAB * sizeof(*reference_next));
    }
    xe_session_reset(cpu);
    xe_session *gpu = cpu;
    size_t allocations = xe_test_allocations;
    size_t outputs = xe_test_output_calls;
    double gpu_start = session_now();
    xe_tokens prefix = { tokens, batch_start, total };
    if (batch_start) xe_session_sync(gpu, &prefix);
    xe_prefill_batch_run(gpu, tokens + batch_start, rows, batch_start,
                         cpu_tail == 0);
    for (int pos = batch_start + rows; pos < total; pos++)
        xe_decode_token_mode(gpu, tokens[pos], pos, XE_MOE_GENERIC_BATCHED, 1,
                             XE_SOFTCAP_SECOND_LOOP, pos == total - 1);
    double gpu_seconds = session_now() - gpu_start;
    int hot_path_ok = allocations == xe_test_allocations
                      && xe_test_output_calls == outputs + 1 + !!batch_start;
    _Float16 *gpu_k = xe_alloc(NULL, kv_elements * sizeof(*gpu_k), XE_MEM_HOST);
    _Float16 *gpu_v = xe_alloc(NULL, kv_elements * sizeof(*gpu_v), XE_MEM_HOST);
    session_kv_copy(gpu, total, gpu_k, gpu_v);
    double hidden_error = session_rel(gpu->hidden, reference_hidden, XE_EMBD);
    double logits_error = session_rel(gpu->logits, reference_logits, XE_VOCAB);
    int top_ok = session_argmax(gpu->logits, XE_VOCAB)
                 == session_argmax(reference_logits, XE_VOCAB);
    size_t kv_mismatches = 0;
    for (size_t i = 0; i < kv_elements; i++)
        kv_mismatches += gpu_k[i] != reference_k[i]
                         || gpu_v[i] != reference_v[i];
    double k_error = session_half_rel(gpu_k, reference_k, kv_elements);
    double v_error = session_half_rel(gpu_v, reference_v, kv_elements);
    if (argc > 6 && !split_mode) {
        float *golden = xe_alloc(
            NULL, XE_VOCAB * sizeof(*golden), XE_MEM_HOST);
        FILE *f = fopen(argv[6], "rb");
        if (!f) xe_fatal("%s: %s", argv[6], strerror(errno));
        if (fread(golden, sizeof(*golden), XE_VOCAB, f) != XE_VOCAB)
            xe_fatal("%s: short logits file", argv[6]);
        fclose(f);
        printf("prefill-golden: CPU %.3e top %d/%d GPU %.3e top %d/%d\n",
               session_rel(reference_logits, golden, XE_VOCAB),
               session_argmax(reference_logits, XE_VOCAB),
               session_argmax(golden, XE_VOCAB),
               session_rel(gpu->logits, golden, XE_VOCAB),
               session_argmax(gpu->logits, XE_VOCAB),
               session_argmax(golden, XE_VOCAB));
        xe_free(NULL, golden, XE_MEM_HOST);
    }
    int greedy_mismatches = 0;
    int first_greedy_mismatch = -1;
    double next_error = 0.0;
    int next_top_ok = 0;
    for (int step = 0; step < 8; step++) {
        int predicted = session_argmax(gpu->logits, XE_VOCAB);
        if (predicted != continuation[step]) {
            greedy_mismatches++;
            if (first_greedy_mismatch < 0) first_greedy_mismatch = step;
        }
        xe_decode_token(gpu, continuation[step], total + step);
        if (step == 0) {
            next_error = session_rel(gpu->logits, reference_next, XE_VOCAB);
            next_top_ok = session_argmax(gpu->logits, XE_VOCAB)
                          == session_argmax(reference_next, XE_VOCAB);
        }
    }
    int ok = hidden_error < 0.1 && logits_error < 0.1 && top_ok
             && k_error < 0.15 && v_error < 0.15
             && next_error < 0.1 && next_top_ok
             && greedy_mismatches == 0
             && gpu->n_tokens == total + 8 && hot_path_ok;
    printf("prefill-session: start %d M%d CPU %.6f s GPU %.6f s %.3fx hidden %.3e logits %.3e top %s KV %.3e/%.3e mismatches %zu/%zu handoff %.3e top %s greedy %d/8 first %d hot %s %s\n",
           batch_start, rows, cpu_seconds, gpu_seconds,
           cpu_seconds / gpu_seconds,
           hidden_error, logits_error, top_ok ? "same" : "different",
           k_error, v_error, kv_mismatches, 2 * kv_elements, next_error,
           next_top_ok ? "same" : "different",
           greedy_mismatches, first_greedy_mismatch,
           hot_path_ok ? "no-alloc/one-output" : "FAIL",
           ok ? "PASS" : "FAIL");
    xe_free(NULL, gpu_v, XE_MEM_HOST);
    xe_free(NULL, gpu_k, XE_MEM_HOST);
    xe_session_free(gpu);
    xe_free(NULL, reference_next, XE_MEM_HOST);
    xe_free(NULL, reference_logits, XE_MEM_HOST);
    xe_free(NULL, reference_hidden, XE_MEM_HOST);
    xe_free(NULL, reference_v, XE_MEM_HOST);
    xe_free(NULL, reference_k, XE_MEM_HOST);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
