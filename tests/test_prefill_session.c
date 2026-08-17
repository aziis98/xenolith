#define XE_TEST_ALLOC
#define XE_TEST_OUTPUT_COUNT
#include "../xenolith.c"

static double session_now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
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

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [rows [batch_start [mode-or-ids [args...]]]]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int rows = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 32;
    int batch_start = argc > 3 ? (int)strtol(argv[3], NULL, 10) : 0;
    if (rows != 32 && rows != 512)
        xe_fatal("prefill session test supports M32 or M512");
    if (batch_start < 0 || batch_start + rows > 1024)
        xe_fatal("prefill session batch start out of range");
    int total = batch_start + rows;
    int32_t tokens[1025];
    for (int row = 0; row < total; row++) tokens[row] = 2 + row;
    xe_engine *e = xe_engine_open(model);
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
    tokens[total] = session_argmax(reference_logits, XE_VOCAB);
    xe_decode_token(cpu, tokens[total], total);
    memcpy(reference_next, cpu->logits, XE_VOCAB * sizeof(*reference_next));
    xe_session_free(cpu);

    xe_session *gpu = xe_session_new(e);
    size_t allocations = xe_test_allocations;
    size_t outputs = xe_test_output_calls;
    double gpu_start = session_now();
    xe_tokens prefix = { tokens, batch_start, total };
    if (batch_start) xe_session_sync(gpu, &prefix);
    xe_prefill_batch_run(gpu, tokens + batch_start, rows, batch_start, 1);
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
    xe_decode_token(gpu, tokens[total], total);
    double next_error = session_rel(gpu->logits, reference_next, XE_VOCAB);
    int next_top_ok = session_argmax(gpu->logits, XE_VOCAB)
                      == session_argmax(reference_next, XE_VOCAB);
    int ok = hidden_error < 0.1 && logits_error < 0.1 && top_ok
             && k_error < 0.15 && v_error < 0.15
             && next_error < 0.1 && next_top_ok
             && gpu->n_tokens == total + 1 && hot_path_ok;
    printf("prefill-session: start %d M%d CPU %.6f s GPU %.6f s %.3fx hidden %.3e logits %.3e top %s KV %.3e/%.3e mismatches %zu/%zu handoff %.3e top %s hot %s %s\n",
           batch_start, rows, cpu_seconds, gpu_seconds,
           cpu_seconds / gpu_seconds,
           hidden_error, logits_error, top_ok ? "same" : "different",
           k_error, v_error, kv_mismatches, 2 * kv_elements, next_error,
           next_top_ok ? "same" : "different",
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
