#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "llama.h"

#define TOPK 8

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

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <model.gguf> \"id,id,id,...\" <out.bin>\n", argv[0]);
        return 2;
    }

    const char *model_path = argv[1];
    const char *out_path   = argv[3];

    llama_token *ids = NULL;
    int n_ids = parse_ids(argv[2], &ids);
    if (n_ids <= 0) {
        fprintf(stderr, "error: could not parse token ids\n");
        free(ids);
        return 2;
    }

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
        llama_backend_free();
        return 1;
    }
    llama_synchronize(ctx);
    double t_decode = now_sec() - t1;

    const float *logits = llama_get_logits_ith(ctx, n_ids - 1);
    if (!logits) {
        fprintf(stderr, "error: no logits for the last position\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        free(ids);
        llama_backend_free();
        return 1;
    }

    FILE *f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s for writing\n", out_path);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        free(ids);
        llama_backend_free();
        return 1;
    }
    size_t wr = fwrite(logits, sizeof(float), (size_t)n_vocab, f);
    if (fclose(f) != 0 || wr != (size_t)n_vocab) {
        fprintf(stderr, "error: short write to %s\n", out_path);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        free(ids);
        llama_backend_free();
        return 1;
    }

    int top_i[TOPK];
    float top_v[TOPK];
    int n_top = 0;
    for (int v = 0; v < n_vocab; v++) {
        float x = logits[v];
        if (n_top < TOPK) {
            int j = n_top++;
            while (j > 0 && top_v[j - 1] < x) {
                top_v[j] = top_v[j - 1];
                top_i[j] = top_i[j - 1];
                j--;
            }
            top_v[j] = x;
            top_i[j] = v;
        } else if (x > top_v[TOPK - 1]) {
            int j = TOPK - 1;
            while (j > 0 && top_v[j - 1] < x) {
                top_v[j] = top_v[j - 1];
                top_i[j] = top_i[j - 1];
                j--;
            }
            top_v[j] = x;
            top_i[j] = v;
        }
    }

    printf("model      %s\n", model_path);
    printf("n_tokens   %d\n", n_ids);
    printf("n_vocab    %d\n", n_vocab);
    printf("out        %s (%zu bytes)\n", out_path, (size_t)n_vocab * sizeof(float));
    printf("load_s     %.3f\n", t_load);
    printf("decode_s   %.3f\n", t_decode);
    for (int k = 0; k < n_top; k++) {
        printf("top%-2d      %7d  %.8f\n", k + 1, top_i[k], (double)top_v[k]);
    }
    fflush(stdout);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    free(ids);
    llama_backend_free();
    return 0;
}
