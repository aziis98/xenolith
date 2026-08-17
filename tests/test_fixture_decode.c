#define XE_TEST_ALLOC
#define XE_TEST_OUTPUT_COUNT
#include "../xenolith.c"

static char *fixture_read(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) xe_fatal("%s: %s", path, strerror(errno));
    if (fseek(f, 0, SEEK_END) != 0) xe_fatal("%s: seek failed", path);
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) xe_fatal("%s: seek failed", path);
    char *p = xe_alloc(NULL, (size_t)n + 1, XE_MEM_HOST);
    if (fread(p, 1, (size_t)n, f) != (size_t)n) xe_fatal("%s: short read", path);
    fclose(f);
    p[n] = 0;
    *size = (size_t)n;
    return p;
}

static int fixture_ids(const char *path, int32_t *ids, int cap) {
    size_t size;
    char *text = fixture_read(path, &size);
    (void)size;
    char *p = text;
    int n = 0;
    while (*p) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',') p++;
        if (!*p) break;
        char *end;
        long value = strtol(p, &end, 10);
        if (end == p || n == cap) xe_fatal("%s: bad token list", path);
        ids[n++] = (int32_t)value;
        p = end;
    }
    xe_free(NULL, text, XE_MEM_HOST);
    return n;
}

static double fixture_rel(const float *a, const float *b, int n) {
    double error = 0.0;
    double reference = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        error += d * d;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static int fixture_argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

static void fixture_session_sync(xe_session *s, int32_t *ids, int n) {
    xe_tokens prefix = { ids, n, n };
    xe_session_sync(s, &prefix);
}

static double fixture_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int fixture_sync(xe_engine *e, int32_t *ids, int n, const float *want) {
    xe_session *s = xe_session_new(e);
    size_t allocations = xe_test_allocations;
    size_t outputs = xe_test_output_calls;
    fixture_session_sync(s, ids, n - 1);
    outputs++;
    int output_count_ok = xe_test_output_calls == outputs;
    fixture_session_sync(s, ids, n);
    outputs++;
    output_count_ok = output_count_ok && xe_test_output_calls == outputs;
    int ok = memcmp(xe_session_logits(s), want, XE_VOCAB * sizeof(float)) == 0;
    ok = ok && s->n_tokens == n && memcmp(s->tokens, ids, (size_t)n * sizeof(*ids)) == 0;
    fixture_session_sync(s, ids, n);
    output_count_ok = output_count_ok && xe_test_output_calls == outputs;
    ok = ok && memcmp(xe_session_logits(s), want, XE_VOCAB * sizeof(float)) == 0;

    fixture_session_sync(s, ids, n - 1);
    outputs++;
    output_count_ok = output_count_ok && xe_test_output_calls == outputs;
    fixture_session_sync(s, ids, n);
    outputs++;
    output_count_ok = output_count_ok && xe_test_output_calls == outputs;
    ok = ok && memcmp(xe_session_logits(s), want, XE_VOCAB * sizeof(float)) == 0;

    int32_t changed[8192];
    memcpy(changed, ids, (size_t)n * sizeof(*ids));
    changed[n - 1] = (changed[n - 1] + 1) % XE_VOCAB;
    fixture_session_sync(s, changed, n);
    outputs++;
    output_count_ok = output_count_ok && xe_test_output_calls == outputs;
    fixture_session_sync(s, ids, n);
    outputs++;
    output_count_ok = output_count_ok && xe_test_output_calls == outputs;
    ok = ok && memcmp(xe_session_logits(s), want, XE_VOCAB * sizeof(float)) == 0;

    static const int32_t expected[] = {
        715, 236772, 759, 569, 9105, 236772, 759, 569
    };
    int32_t generated[8];
    int32_t prompt[8200];
    memcpy(prompt, ids, (size_t)n * sizeof(*ids));
    xe_sampler greedy = { 0.0f, 0, 1.0f, 1 };
    for (int i = 0; i < 8; i++) {
        generated[i] = xe_session_next(s, &greedy);
        prompt[n + i] = generated[i];
        if (i + 1 < 8) {
            fixture_session_sync(s, prompt, n + i + 1);
            outputs++;
            output_count_ok = output_count_ok && xe_test_output_calls == outputs;
        }
    }
    int continuation_ok = memcmp(generated, expected, sizeof expected) == 0;
    if (!continuation_ok) {
        printf("fixture: greedy got");
        for (int i = 0; i < 8; i++) printf(" %d", generated[i]);
        printf("\n");
    }
    ok = ok && continuation_ok;
    printf("fixture: greedy continuation %s\n", continuation_ok ? "PASS" : "FAIL");
    ok = ok && output_count_ok && allocations == xe_test_allocations;
    printf("fixture: session output count %s\n", output_count_ok ? "PASS" : "FAIL");
    printf("fixture: session append/reuse/shorter/diverge %s\n", ok ? "PASS" : "FAIL");
    xe_session_free(s);
    return ok;
}

static int fixture_run(xe_engine *e, const char *name, int with_reference) {
    char ids_path[1024];
    char logits_path[1024];
    snprintf(ids_path, sizeof ids_path, "tests/golden/%s.ids", name);
    snprintf(logits_path, sizeof logits_path, "tests/golden/%s.logits", name);
    int32_t ids[8192];
    int n = fixture_ids(ids_path, ids, 8192);
    size_t logits_bytes;
    float *golden = (float *)fixture_read(logits_path, &logits_bytes);
    if (logits_bytes != XE_VOCAB * sizeof(float)) xe_fatal("%s: bad logits size", logits_path);

    xe_session *s = xe_session_new(e);
    size_t allocations = xe_test_allocations;
    double start = fixture_now();
    fixture_session_sync(s, ids, n);
    double seconds = fixture_now() - start;
    int finite = 1;
    for (int i = 0; i < XE_VOCAB; i++) finite = finite && isfinite(s->logits[i]);
    double rel = fixture_rel(s->logits, golden, XE_VOCAB);
    int got_top = fixture_argmax(s->logits, XE_VOCAB);
    int golden_top = fixture_argmax(golden, XE_VOCAB);
    int no_alloc = allocations == xe_test_allocations;
    int ok = finite && no_alloc && rel <= 0.6;
    printf("fixture: %s n=%d %.3f s %.3f tok/s rel %.3e top %d/%d alloc %s %s\n",
           name, n, seconds, (double)n / seconds, rel, got_top, golden_top,
           no_alloc ? "none" : "HOT-PATH", ok ? "PASS" : "FAIL");

    if (!strcmp(name, "short")) ok = fixture_sync(e, ids, n, s->logits) && ok;

    if (with_reference) {
        ref_state ref;
        memset(&ref, 0, sizeof ref);
        ref.q8 = 2;
        ref_state_init(&ref, n);
        double ref_start = fixture_now();
        ref_forward(e, ids, n, &ref);
        double ref_seconds = fixture_now() - ref_start;
        double ref_rel = fixture_rel(s->logits, ref.logits, XE_VOCAB);
        int ref_top = fixture_argmax(ref.logits, XE_VOCAB);
        int ref_ok = ref_rel <= 0.6;
        printf("fixture: %s V1-oracle %.3f s rel %.3e top %d/%d %s\n",
               name, ref_seconds, ref_rel, got_top, ref_top, ref_ok ? "PASS" : "FAIL");
        ok = ok && ref_ok;
        ref_state_free(&ref);
    }

    xe_session_free(s);
    xe_free(NULL, golden, XE_MEM_HOST);
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [short|long [v1]]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    xe_engine *e = xe_engine_open(model);
    int with_reference = argc > 3 && !strcmp(argv[3], "v1");
    int ok = 1;
    if (argc <= 2 || !strcmp(argv[2], "short")) ok &= fixture_run(e, "short", with_reference);
    if (argc <= 2 || !strcmp(argv[2], "long")) ok &= fixture_run(e, "long", with_reference);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
