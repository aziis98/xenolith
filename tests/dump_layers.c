#define _POSIX_C_SOURCE 200809L

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#define N_LAYERS 30

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int parse_ids(const char *s, llama_token **out) {
    int cap = 256, n = 0;
    llama_token *a = malloc((size_t)cap * sizeof *a);
    if (!a) return -1;
    const char *p = s;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' ||
               *p == ',' || *p == '[' || *p == ']') {
            p++;
        }
        if (!*p) break;
        char *e;
        long v = strtol(p, &e, 10);
        if (e == p) {
            free(a);
            return -1;
        }
        if (n == cap) {
            cap *= 2;
            llama_token *t = realloc(a, (size_t)cap * sizeof *a);
            if (!t) {
                free(a);
                return -1;
            }
            a = t;
        }
        a[n++] = (llama_token)v;
        p = e;
    }
    *out = a;
    return n;
}

typedef struct {
    const char *outdir;
    int names_only;
    int n_dumped;
    int n_failed;
    uint8_t *buf;
    size_t buf_cap;
    float *row;
    size_t row_cap;
} cb_state;

static int want_tensor(const char *name) {
    if (!strcmp(name, "inp_scaled")) return 1;
    for (int il = 0; il < N_LAYERS; il++) {
        char pat[64];
        snprintf(pat, sizeof pat, "l_out-%d", il);
        if (!strcmp(name, pat)) return 1;
        snprintf(pat, sizeof pat, "attn_out-%d", il);
        if (!strcmp(name, pat)) return 1;
    }
    return 0;
}

static void sanitize(const char *name, char *out, size_t cap) {
    size_t i = 0;
    for (; name[i] && i + 1 < cap; i++) {
        char c = name[i];
        int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                 (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        out[i] = ok ? c : '_';
    }
    out[i] = '\0';
}

static bool grow_u8(cb_state *s, size_t need) {
    if (need <= s->buf_cap) return true;
    uint8_t *p = realloc(s->buf, need);
    if (!p) return false;
    s->buf = p;
    s->buf_cap = need;
    return true;
}

static bool grow_f32(cb_state *s, size_t need) {
    if (need <= s->row_cap) return true;
    float *p = realloc(s->row, need * sizeof(float));
    if (!p) return false;
    s->row = p;
    s->row_cap = need;
    return true;
}

static bool cb_eval(struct ggml_tensor *t, bool ask, void *user_data) {
    cb_state *s = (cb_state *)user_data;

    if (ask) return s->names_only ? true : (want_tensor(t->name) != 0);

    if (s->names_only) {
        printf("%-28s %-5s [%lld, %lld, %lld, %lld]\n", t->name, ggml_type_name(t->type),
               (long long)t->ne[0], (long long)t->ne[1],
               (long long)t->ne[2], (long long)t->ne[3]);
        return true;
    }

    if (!want_tensor(t->name)) return true;

    if (t->ne[2] != 1 || t->ne[3] != 1) {
        fprintf(stderr, "error: %s has ne2=%lld ne3=%lld, expected 1\n", t->name,
                (long long)t->ne[2], (long long)t->ne[3]);
        s->n_failed++;
        return true;
    }
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) {
        fprintf(stderr, "error: %s has unsupported type %s\n", t->name, ggml_type_name(t->type));
        s->n_failed++;
        return true;
    }

    size_t nbytes = ggml_nbytes(t);
    size_t ne0 = (size_t)t->ne[0];
    if (!grow_u8(s, nbytes) || !grow_f32(s, ne0)) {
        fprintf(stderr, "error: out of memory for %s\n", t->name);
        s->n_failed++;
        return true;
    }

    ggml_backend_tensor_get(t, s->buf, 0, nbytes);

    const uint8_t *col = s->buf + (size_t)(t->ne[1] - 1) * t->nb[1];
    if (t->type == GGML_TYPE_F32) {
        memcpy(s->row, col, ne0 * sizeof(float));
    } else {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)col, s->row, (int64_t)ne0);
    }

    char safe[256];
    char path[1024];
    sanitize(t->name, safe, sizeof safe);
    snprintf(path, sizeof path, "%s/%s.bin", s->outdir, safe);

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s for writing\n", path);
        s->n_failed++;
        return true;
    }
    size_t wr = fwrite(s->row, sizeof(float), ne0, f);
    if (fclose(f) != 0 || wr != ne0) {
        fprintf(stderr, "error: short write to %s\n", path);
        s->n_failed++;
        return true;
    }

    s->n_dumped++;
    return true;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "usage: %s <model.gguf> \"id,id,id,...\" <outdir> [names]\n", argv[0]);
        return 2;
    }

    const char *model_path = argv[1];
    const char *outdir     = argv[3];
    int names_only = (argc == 5 && !strcmp(argv[4], "names"));

    if (argc == 5 && !names_only) {
        fprintf(stderr, "error: 4th argument must be \"names\"\n");
        return 2;
    }

    llama_token *ids = NULL;
    int n_ids = parse_ids(argv[2], &ids);
    if (n_ids <= 0) {
        fprintf(stderr, "error: could not parse token ids\n");
        free(ids);
        return 2;
    }

    if (!names_only && mkdir(outdir, 0777) != 0) {
        struct stat st;
        if (stat(outdir, &st) != 0 || !S_ISDIR(st.st_mode)) {
            fprintf(stderr, "error: cannot create directory %s\n", outdir);
            free(ids);
            return 1;
        }
    }

    cb_state cbs;
    cbs.outdir     = outdir;
    cbs.names_only = names_only;
    cbs.n_dumped   = 0;
    cbs.n_failed   = 0;
    cbs.buf        = NULL;
    cbs.buf_cap    = 0;
    cbs.row        = NULL;
    cbs.row_cap    = 0;

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.devices      = NULL;
    mparams.use_extra_bufts = false;

    double t0 = now_sec();
    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "error: failed to load model %s\n", model_path);
        free(ids);
        llama_backend_free();
        return 1;
    }
    double t_load = now_sec() - t0;

    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    for (int i = 0; i < n_ids; i++) {
        if (ids[i] < 0 || ids[i] >= n_vocab) {
            fprintf(stderr, "error: token id %d at position %d out of range [0,%d)\n",
                    ids[i], i, n_vocab);
            llama_model_free(model);
            free(ids);
            llama_backend_free();
            return 2;
        }
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t)n_ids;
    cparams.n_batch         = (uint32_t)n_ids;
    cparams.n_ubatch        = (uint32_t)n_ids;
    cparams.n_seq_max       = 1;
    cparams.n_threads       = 8;
    cparams.n_threads_batch = 8;
    cparams.embeddings      = false;
    cparams.offload_kqv     = false;
    cparams.op_offload      = false;
    cparams.no_perf         = true;
    cparams.swa_full        = true;
    cparams.kv_unified      = true;
    cparams.cb_eval           = cb_eval;
    cparams.cb_eval_user_data = &cbs;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        llama_model_free(model);
        free(ids);
        llama_backend_free();
        return 1;
    }

    struct llama_batch batch = llama_batch_init(n_ids, 0, 1);
    batch.n_tokens = n_ids;
    for (int i = 0; i < n_ids; i++) {
        batch.token[i]     = ids[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = (int8_t)(i == n_ids - 1);
    }

    double t1 = now_sec();
    int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        fprintf(stderr, "error: llama_decode returned %d\n", rc);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        free(ids);
        free(cbs.buf);
        free(cbs.row);
        llama_backend_free();
        return 1;
    }
    llama_synchronize(ctx);
    double t_decode = now_sec() - t1;

    printf("model      %s\n", model_path);
    printf("n_tokens   %d\n", n_ids);
    printf("mode       %s\n", names_only ? "names" : "dump");
    if (!names_only) {
        printf("outdir     %s\n", outdir);
        printf("dumped     %d\n", cbs.n_dumped);
        printf("failed     %d\n", cbs.n_failed);
    }
    printf("load_s     %.3f\n", t_load);
    printf("decode_s   %.3f\n", t_decode);
    fflush(stdout);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    free(ids);
    free(cbs.buf);
    free(cbs.row);
    llama_backend_free();
    return cbs.n_failed ? 1 : 0;
}
