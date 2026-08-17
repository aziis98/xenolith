#define _POSIX_C_SOURCE 200809L

#include "xenolith.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define XE_VOCAB 262144
#define XE_BOS 2
#define XE_ORACLE_THRESHOLD 0.6

#define XE_DELIM "\n__ggml_vocab_test__\n"

static const char *CORPUS[] = { "short", "long", "mixed", "ws", "nl", "nlonly", "json" };
#define N_CORPUS ((int)(sizeof CORPUS / sizeof CORPUS[0]))

static const char *ORACLE_PROMPTS[] = { "short", "long" };
#define N_ORACLE ((int)(sizeof ORACLE_PROMPTS / sizeof ORACLE_PROMPTS[0]))

static void usage(const char *prog) {
    fprintf(stderr, "usage: %s [--tok-only]\n", prog);
    fprintf(stderr, "env: XENOLITH_MODEL\n");
    exit(2);
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static char *read_whole(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "certify: cannot open %s\n", path);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "certify: short read on %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    buf[n] = '\0';
    if (len_out) *len_out = (size_t)n;
    return buf;
}

static int parse_ids(const char *s, int32_t *out, int cap) {
    int n = 0;
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' ||
               *p == ',' || *p == '[' || *p == ']') {
            p++;
        }
        if (!*p) break;
        char *e;
        long v = strtol(p, &e, 10);
        if (e == p) return -1;
        if (n >= cap) return -1;
        out[n++] = (int32_t)v;
        p = e;
    }
    return n;
}

static int ids_equal(const int32_t *a, int na, const int32_t *b, int nb) {
    if (na != nb) return 0;
    for (int i = 0; i < na; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

static int xenolith_tokenize(const xe_engine *e, const char *text, int32_t *out, int cap) {
    if (cap < 1) return -1;
    out[0] = xe_bos_id(e);
    return 1 + xe_encode_text(e, text, out + 1, cap - 1);
}

static float *load_logits(const char *path, long *n_out) {
    size_t bytes = 0;
    char *raw = read_whole(path, &bytes);
    if (!raw) return NULL;
    if (bytes == 0 || bytes % 4 != 0) {
        fprintf(stderr, "certify: %s size %zu not a positive multiple of 4\n", path, bytes);
        free(raw);
        return NULL;
    }
    *n_out = (long)(bytes / 4);
    return (float *)raw;
}

static int compare_logits(const char *name, const float *a, long na,
                          const float *b, long nb, double thr) {
    if (na != nb) {
        printf("certify:   %s FAIL (size mismatch: %ld vs %ld floats)\n", name, na, nb);
        return 0;
    }
    long n = na;
    double sum_d2 = 0.0, sum_b2 = 0.0;
    long top_a = 0, top_b = 0;
    double best_a = (double)a[0], best_b = (double)b[0];
    for (long i = 0; i < n; i++) {
        double av = (double)a[i], bv = (double)b[i];
        double d = av - bv;
        sum_d2 += d * d;
        sum_b2 += bv * bv;
        if (av > best_a) { best_a = av; top_a = i; }
        if (bv > best_b) { best_b = bv; top_b = i; }
    }
    double rms_d = sqrt(sum_d2 / (double)n);
    double rms_b = sqrt(sum_b2 / (double)n);
    double rel = rms_b > 0.0 ? rms_d / rms_b : (rms_d == 0.0 ? 0.0 : INFINITY);
    int top_match = (top_a == top_b);
    int pass = (rel <= thr);
    printf("certify:   %s rel_rms=%.6e rms_d=%.6e rms_b=%.6e top1_match=%s threshold=%.6e margin=%.2fx %s\n",
           name, rel, rms_d, rms_b, top_match ? "yes" : "no", thr,
           rel > 0.0 ? thr / rel : INFINITY, pass ? "PASS" : "FAIL");
    return pass;
}

static void here_dir(const char *argv0, char *out, size_t cap) {
    const char *slash = strrchr(argv0, '/');
    if (!slash) {
        snprintf(out, cap, ".");
        return;
    }
    size_t n = (size_t)(slash - argv0);
    if (n >= cap) n = cap - 1;
    memcpy(out, argv0, n);
    out[n] = '\0';
}

static int stage_fixtures(const xe_engine *e, const char *here) {
    char inp_path[2048], out_path[2048];
    snprintf(inp_path, sizeof inp_path, "%s/fixtures/ggml-vocab-gemma-4.gguf.inp", here);
    snprintf(out_path, sizeof out_path, "%s/fixtures/ggml-vocab-gemma-4.gguf.out", here);

    size_t inp_len = 0;
    char *inp = read_whole(inp_path, &inp_len);
    if (!inp) return 0;
    char *out = read_whole(out_path, NULL);
    if (!out) { free(inp); return 0; }

    char *seg[512];
    int ncases = 0;
    char *cur = inp;
    seg[ncases++] = cur;
    char *d;
    size_t dl = strlen(XE_DELIM);
    while ((d = strstr(cur, XE_DELIM)) != NULL) {
        *d = '\0';
        cur = d + dl;
        if (ncases >= (int)(sizeof seg / sizeof seg[0])) break;
        seg[ncases++] = cur;
    }

    char *line[512];
    int nlines = 0;
    char *lp = out;
    line[nlines++] = lp;
    for (char *q = out; *q; q++) {
        if (*q == '\n') {
            *q = '\0';
            char *nxt = q + 1;
            if (*nxt && nlines < (int)(sizeof line / sizeof line[0]))
                line[nlines++] = nxt;
        }
    }

    if (ncases != nlines) {
        printf("certify: stage a FAIL (%d input cases but %d output lines)\n", ncases, nlines);
        free(inp);
        free(out);
        return 0;
    }

    int pass = 0, fail = 0;
    int32_t got[8192], want[8192];
    for (int i = 0; i < ncases; i++) {
        int n_got = xenolith_tokenize(e, seg[i], got, (int)(sizeof got / sizeof got[0]));
        if (n_got < 1 || got[0] != XE_BOS) {
            printf("certify:   case %d FAIL (no BOS)\n", i);
            fail++;
            continue;
        }
        int n_want = parse_ids(line[i], want, (int)(sizeof want / sizeof want[0]));
        if (n_want < 0) {
            printf("certify:   case %d FAIL (bad expected ids)\n", i);
            fail++;
            continue;
        }
        if (ids_equal(got + 1, n_got - 1, want, n_want)) {
            pass++;
        } else {
            fail++;
            printf("certify:   case %d FAIL (want %d ids, got %d)\n", i, n_want, n_got - 1);
        }
    }

    free(inp);
    free(out);
    if (fail == 0)
        printf("certify: stage a PASS  %d/%d fixtures\n", pass, ncases);
    else
        printf("certify: stage a FAIL  %d/%d fixtures passed, %d failed\n", pass, ncases, fail);
    return fail == 0;
}

static int stage_corpus(const xe_engine *e, const char *here) {
    int pass = 0, fail = 0;
    for (int i = 0; i < N_CORPUS; i++) {
        char src[2048], gold[2048];
        snprintf(src, sizeof src, "%s/prompts/%s.txt", here, CORPUS[i]);
        snprintf(gold, sizeof gold, "%s/golden/%s.ids", here, CORPUS[i]);

        size_t len = 0;
        char *text = read_whole(src, &len);
        if (!text) { printf("certify:   %s FAIL (missing %s)\n", CORPUS[i], src); fail++; continue; }
        char *gtext = read_whole(gold, NULL);
        if (!gtext) { printf("certify:   %s FAIL (missing %s)\n", CORPUS[i], gold); free(text); fail++; continue; }

        int cap = (int)len + 16;
        int32_t *got = malloc((size_t)cap * sizeof *got);
        int32_t *want = malloc((size_t)cap * sizeof *want);
        int n_got = xenolith_tokenize(e, text, got, cap);
        int n_want = parse_ids(gtext, want, cap);

        int n = n_got;
        if (ids_equal(got, n_got, want, n_want)) {
            printf("certify:   %s PASS  n_tokens=%d\n", CORPUS[i], n);
            pass++;
        } else {
            printf("certify:   %s FAIL  n_tokens=%d (golden %d)\n", CORPUS[i], n_got, n_want);
            fail++;
        }
        free(got);
        free(want);
        free(text);
        free(gtext);
    }
    if (fail == 0)
        printf("certify: stage b PASS  %d/%d corpora\n", pass, N_CORPUS);
    else
        printf("certify: stage b FAIL  %d passed, %d failed\n", pass, fail);
    return fail == 0;
}

static int stage_oracle(xe_engine *e, const char *here) {
    int pass = 0, fail = 0;
    FILE *devnull = fopen("/dev/null", "w");
    for (int i = 0; i < N_ORACLE; i++) {
        char gids[2048], glog[2048];
        snprintf(gids, sizeof gids, "%s/golden/%s.ids", here, ORACLE_PROMPTS[i]);
        snprintf(glog, sizeof glog, "%s/golden/%s.logits", here, ORACLE_PROMPTS[i]);

        char *gtext = read_whole(gids, NULL);
        if (!gtext) { printf("certify:   %s FAIL (missing %s)\n", ORACLE_PROMPTS[i], gids); fail++; continue; }
        int32_t *ids = malloc(8192 * sizeof *ids);
        int n = parse_ids(gtext, ids, 8192);
        free(gtext);
        if (n <= 0) { printf("certify:   %s FAIL (bad golden ids)\n", ORACLE_PROMPTS[i]); free(ids); fail++; continue; }

        char tmpl[] = "/tmp/xenolith-certify-XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd < 0) { printf("certify:   %s FAIL (mkstemp)\n", ORACLE_PROMPTS[i]); free(ids); fail++; continue; }
        close(fd);

        double t0 = now_sec();
        xe_oracle(e, ids, n, tmpl, NULL, 0, devnull ? devnull : stdout);
        double dt = now_sec() - t0;
        free(ids);

        long na = 0, nb = 0;
        float *a = load_logits(tmpl, &na);
        float *b = load_logits(glog, &nb);
        unlink(tmpl);
        if (!a || !b) {
            printf("certify:   %s FAIL (cannot load logits)\n", ORACLE_PROMPTS[i]);
            free(a);
            free(b);
            fail++;
            continue;
        }
        printf("certify:   %s n_tokens=%d oracle_s=%.1f\n", ORACLE_PROMPTS[i], n, dt);
        if (compare_logits(ORACLE_PROMPTS[i], a, na, b, nb, XE_ORACLE_THRESHOLD))
            pass++;
        else
            fail++;
        free(a);
        free(b);
    }
    if (devnull) fclose(devnull);
    if (fail == 0)
        printf("certify: stage c PASS  %d/%d oracle prompts\n", pass, N_ORACLE);
    else
        printf("certify: stage c FAIL  %d passed, %d failed\n", pass, fail);
    return fail == 0;
}

static int stage_roundtrip(const xe_engine *e, const char *here) {
    int pass = 0, fail = 0;
    for (int i = 0; i < N_ORACLE; i++) {
        char src[2048];
        snprintf(src, sizeof src, "%s/prompts/%s.txt", here, ORACLE_PROMPTS[i]);
        size_t len = 0;
        char *text = read_whole(src, &len);
        if (!text) { printf("certify:   %s FAIL (missing %s)\n", ORACLE_PROMPTS[i], src); fail++; continue; }

        int cap = (int)len + 16;
        int32_t *ids = malloc((size_t)cap * sizeof *ids);
        int n = xenolith_tokenize(e, text, ids, cap);

        char *rt = malloc(len + 1);
        size_t rn = 0;
        int overflow = 0;
        for (int k = 1; k < n; k++) {
            char buf[512];
            int nb = xe_detokenize(e, ids[k], buf, (int)sizeof buf);
            if (rn + (size_t)nb > len) { overflow = 1; break; }
            memcpy(rt + rn, buf, (size_t)nb);
            rn += (size_t)nb;
        }
        if (!overflow && rn == len && memcmp(rt, text, len) == 0) {
            printf("certify:   %s PASS  %zu bytes reconstructed\n", ORACLE_PROMPTS[i], len);
            pass++;
        } else {
            printf("certify:   %s FAIL  reconstructed %zu of %zu bytes\n", ORACLE_PROMPTS[i], rn, len);
            fail++;
        }
        free(ids);
        free(rt);
        free(text);
    }
    if (fail == 0)
        printf("certify: stage d PASS  %d/%d roundtrips\n", pass, N_ORACLE);
    else
        printf("certify: stage d FAIL  %d passed, %d failed\n", pass, fail);
    return fail == 0;
}

static int stage_chat_prompt(xe_engine *e) {
    static const int32_t first[] = {
        2, 105, 2364, 107, 5379, 236748, 236764, 2229, 203360, 236881,
        106, 107, 105, 4368, 107, 100, 45518, 107, 101,
    };
    static const int32_t second[] = {
        107, 105, 2364, 107, 3900, 152039, 655, 236909, 887,
        236813, 6064, 514, 4775, 106, 107, 105, 4368, 107, 100, 45518,
        107, 101,
    };

    xe_tokens transcript = {0};
    int ok = xe_chat_prepare_reply(e, &transcript, "  ciao, come stai?  ");
    ok = ok && ids_equal(transcript.v, transcript.len,
                         first, (int)(sizeof first / sizeof first[0]));

    int32_t answer[64];
    int na = xe_encode_text(e, "Ciao!", answer, (int)(sizeof answer / sizeof answer[0]));
    for (int i = 0; i < na; i++) xe_tokens_push(&transcript, answer[i]);
    xe_tokens_push(&transcript, xe_eot_id(e));
    int before = transcript.len;
    ok = ok && xe_chat_prepare_reply(e, &transcript, "Scrivi <|turn> letteralmente");
    ok = ok && ids_equal(transcript.v + before, transcript.len - before,
                         second, (int)(sizeof second / sizeof second[0]));

    int32_t literal[32];
    int nl = xe_encode_text(e, "<|turn>", literal, (int)(sizeof literal / sizeof literal[0]));
    for (int i = 0; i < nl; i++) if (literal[i] == 105) ok = 0;
    if (xe_encode_text(e, "", literal, (int)(sizeof literal / sizeof literal[0])) != 0) ok = 0;

    xe_tokens full = {0};
    xe_tokens_push(&full, xe_bos_id(e));
    while (full.len < xe_context_size(e) - 1) xe_tokens_push(&full, xe_eot_id(e));
    int full_len = full.len;
    int32_t *full_v = full.v;
    if (xe_chat_prepare_reply(e, &full, "too much") != 0 ||
        full.len != full_len || full.v != full_v)
        ok = 0;

    xe_tokens_free(&transcript);
    xe_tokens_free(&full);
    printf("certify: stage e %s  template parity, live append, special-token isolation\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char **argv) {
    int tok_only = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--tok-only")) tok_only = 1;
        else usage(argv[0]);
    }

    const char *model = getenv("XENOLITH_MODEL");
    if (!model || !*model) {
        fprintf(stderr, "error: set XENOLITH_MODEL to the Gemma 4 GGUF path\n");
        usage(argv[0]);
    }

    char here[1024];
    here_dir(argv[0], here, sizeof here);

    printf("certify: model     %s\n", model);
    printf("certify: mode      %s\n", tok_only ? "tok-only (a,b,d,e)" : "full (a,b,c,d,e)");
    printf("certify: here      %s\n", here);

    double t_open = now_sec();
    xe_engine *e = tok_only ? xe_engine_open_vocab(model) : xe_engine_open(model);
    printf("certify: engine    opened in %.2fs (%s)\n", now_sec() - t_open,
           tok_only ? "vocab-only" : "full");

    int ok = 1;
    printf("certify: stage a/5 tokenizer exactness (llama.cpp fixtures)\n");
    ok &= stage_fixtures(e, here);
    printf("certify: stage b/5 corpus parity\n");
    ok &= stage_corpus(e, here);
    if (!tok_only) {
        printf("certify: stage c/5 oracle numerical\n");
        ok &= stage_oracle(e, here);
    } else {
        printf("certify: stage c/5 oracle numerical — skipped (tok-only)\n");
    }
    printf("certify: stage d/5 roundtrip\n");
    ok &= stage_roundtrip(e, here);
    printf("certify: stage e/5 chat protocol\n");
    ok &= stage_chat_prompt(e);

    xe_engine_close(e);

    if (ok) {
        printf("certify: PASS\n");
        return 0;
    }
    printf("certify: FAIL\n");
    return 1;
}
