#define XE_TEST_ALLOC
#define XE_TEST_OUTPUT_COUNT
#define XE_TEST_SESSION
#include "../xenolith.c"

size_t xe_test_prefill_batches;
size_t xe_test_decode_tokens;

static int sync_tokens_equal(const xe_session *s, const int32_t *tokens,
                             int n) {
    return s->n_tokens == n
           && memcmp(s->tokens, tokens, (size_t)n * sizeof(*tokens)) == 0;
}

static int sync_logits_finite(const xe_session *s) {
    for (int i = 0; i < XE_VOCAB; i++)
        if (!isfinite(s->logits[i])) return 0;
    return 1;
}

static void sync_counts_reset(void) {
    xe_test_prefill_batches = 0;
    xe_test_decode_tokens = 0;
    xe_test_output_calls = 0;
}

static int sync_counts(size_t batches, size_t decodes, size_t outputs) {
    return xe_test_prefill_batches == batches
           && xe_test_decode_tokens == decodes
           && xe_test_output_calls == outputs;
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int32_t original[1568];
    int32_t branch[1568];
    for (int i = 0; i < 1568; i++) original[i] = 2 + i;
    xe_engine *e = xe_engine_open(model);
    xe_session *s = xe_session_new(e);
    size_t allocations = xe_test_allocations;
    int ok = 1;

    xe_tokens prefix = { original, 64, 1568 };
    sync_counts_reset();
    xe_session_sync(s, &prefix);
    int initial = sync_counts(1, 0, 1)
                  && sync_tokens_equal(s, original, 64)
                  && sync_logits_finite(s);
    ok &= initial;
    printf("session-sync: initial GPU batch %s\n", initial ? "PASS" : "FAIL");

    sync_counts_reset();
    xe_session_sync(s, &prefix);
    int reuse = sync_counts(0, 0, 0)
                && sync_tokens_equal(s, original, 64);
    ok &= reuse;
    printf("session-sync: exact reuse %s\n", reuse ? "PASS" : "FAIL");

    sync_counts_reset();
    xe_session_rewind(s, 32);
    int rewind = sync_counts(1, 0, 1)
                 && xe_session_position(s) == 32
                 && sync_tokens_equal(s, original, 32)
                 && sync_logits_finite(s);
    ok &= rewind;
    printf("session-sync: safe rewind %s\n", rewind ? "PASS" : "FAIL");

    sync_counts_reset();
    xe_session_sync(s, &prefix);
    int extend = sync_counts(1, 0, 1)
                 && sync_tokens_equal(s, original, 64)
                 && sync_logits_finite(s);
    ok &= extend;
    printf("session-sync: GPU extension %s\n", extend ? "PASS" : "FAIL");

    xe_session_reset(s);
    prefix.len = 1056;
    sync_counts_reset();
    xe_session_sync(s, &prefix);
    int long_initial = sync_counts(3, 0, 1)
                       && sync_tokens_equal(s, original, 1056)
                       && sync_logits_finite(s);
    ok &= long_initial;
    printf("session-sync: long M512x2+M32 %s\n",
           long_initial ? "PASS" : "FAIL");

    memcpy(branch, original, sizeof branch);
    branch[1055] = 4000;
    prefix.v = branch;
    sync_counts_reset();
    xe_session_sync(s, &prefix);
    int safe_branch = sync_counts(1, 0, 1)
                      && sync_tokens_equal(s, branch, 1056)
                      && sync_logits_finite(s);
    ok &= safe_branch;
    printf("session-sync: safe SWA branch %s\n",
           safe_branch ? "PASS" : "FAIL");

    memcpy(original, branch, sizeof original);
    original[1054] = 5000;
    prefix.v = original;
    sync_counts_reset();
    xe_session_sync(s, &prefix);
    int rebuild = sync_counts(3, 0, 1)
                  && sync_tokens_equal(s, original, 1056)
                  && sync_logits_finite(s);
    ok &= rebuild;
    printf("session-sync: overwritten SWA rebuild %s\n",
           rebuild ? "PASS" : "FAIL");

    prefix.len = 1568;
    sync_counts_reset();
    xe_session_sync(s, &prefix);
    int wrapped_batch = sync_counts(1, 0, 1)
                        && sync_tokens_equal(s, original, 1568)
                        && sync_logits_finite(s);
    ok &= wrapped_batch;
    printf("session-sync: wrapped SWA GPU extension %s\n",
           wrapped_batch ? "PASS" : "FAIL");

    int hot = allocations == xe_test_allocations;
    ok &= hot;
    printf("session-sync: hot allocation %s\n", hot ? "PASS" : "FAIL");
    xe_session_free(s);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
