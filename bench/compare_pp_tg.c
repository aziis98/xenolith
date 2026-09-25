#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/resource.h>
#include <dirent.h>
#ifdef LLAMA
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
static struct llama_model *model;
static struct llama_context *ctx;
static struct ggml_threadpool *gen_pool, *batch_pool;
static int position;
#else
#include "xenolith.h"
static xe_engine *engine;
static xe_session *session;
#endif
static int32_t tokens[262144];
static int ntokens;
static int envint(const char *key, int fallback) {
    const char *s = getenv(key);
    return s ? atoi(s) : fallback;
}
static void need(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL %s\n", message); exit(1); }
}
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static double cpu(void) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + r.ru_stime.tv_sec + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6;
}
static void memory(const char *phase) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    unsigned long rss = 0, pss = 0, swap = 0;
    FILE *f = fopen("/proc/self/smaps_rollup", "r");
    char line[1024];
    if (f) {
        while (fgets(line, sizeof line, f)) {
            sscanf(line, "Rss: %lu", &rss);
            sscanf(line, "Pss: %lu", &pss);
            sscanf(line, "Swap: %lu", &swap);
        }
        fclose(f);
    }
    printf("{\"event\":\"memory\",\"phase\":\"%s\",\"rss_kib\":%lu,\"pss_kib\":%lu,\"swap_kib\":%lu,\"maxrss_kib\":%ld,\"major_faults\":%ld}\n", phase, rss, pss, swap, r.ru_maxrss, r.ru_majflt);
}
static void reset(void) {
#ifdef LLAMA
    llama_memory_clear(llama_get_memory(ctx), false);
    position = 0;
#else
    xe_session_reset(session);
#endif
}
static void sync_to(int n) {
#ifdef LLAMA
    while (position < n) {
        int count = n - position;
        if (count > 2048) count = 2048;
        need(llama_decode(ctx, llama_batch_get_one(tokens + position, count)) == 0, "decode");
        position += count;
    }
    llama_synchronize(ctx);
    need(llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == n - 1, "llama position");
#else
    xe_tokens prefix = {tokens, n, 262144};
    xe_session_sync(session, &prefix);
    need(xe_session_position(session) == n, "xenolith position");
#endif
}
static const float *logits(void) {
#ifdef LLAMA
    return llama_get_logits_ith(ctx, -1);
#else
    return xe_session_logits(session);
#endif
}
static int best(void) {
    const float *v = logits();
    need(v != NULL, "missing logits");
    int top = 0;
    for (int i = 0; i < 262144; i++) {
        need(isfinite(v[i]), "nonfinite logits");
        if (v[i] > v[top]) top = i;
    }
    return top;
}
static void result(const char *phase, int rep, int depth, int count, double start, double cstart, int reused) {
    double elapsed = now() - start;
    printf("{\"event\":\"measure\",\"phase\":\"%s\",\"rep\":%d,\"depth\":%d,\"tokens\":%d,\"seconds\":%.9f,\"cpu_seconds\":%.9f,\"tps\":%.6f,\"reused\":%d}\n", phase, rep, depth, count, elapsed, cpu() - cstart, count / elapsed, reused);
}
int main(int argc, char **argv) {
    need(argc == 5, "usage: compare_pp_tg MODEL TOKEN_FILE DEPTH REPS; use TG_ONLY=1 for depth=0 decode");
    setvbuf(stdout, NULL, _IOLBF, 0);
    int depth = atoi(argv[3]), reps = atoi(argv[4]);
#ifndef LLAMA
    if (depth == 0 && !envint("TG_ONLY", 0)) {
        engine = xe_engine_open_vocab(argv[1]);
        FILE *f = fopen(argv[2], "rb");
        need(f != NULL, "text open");
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        rewind(f);
        char *text = calloc(size + 1, 1);
        need(text && fread(text, 1, size, f) == (size_t)size, "text read");
        tokens[0] = xe_bos_id(engine);
        int n = xe_encode_text(engine, text, tokens + 1, 262143);
        need(n > 0, "tokenize");
        for (int i = 0; i <= n; i++) printf("%d\n", tokens[i]);
        fclose(f);
        free(text);
        xe_engine_close(engine);
        return 0;
    }
#endif
    FILE *f = fopen(argv[2], "r");
    need(f != NULL, "tokens open");
    while (ntokens < 262144 && fscanf(f, "%d", tokens + ntokens) == 1) ntokens++;
    fclose(f);
    int gen = envint("GEN", 128);
    need(depth >= 0 && depth + gen > 0 && depth + gen + 256 < ntokens && reps > 0, "input length");
    printf("{\"event\":\"start\",\"depth\":%d,\"reps\":%d,\"gen\":%d}\n", depth, reps, gen);
    double t = now(), c = cpu();
#ifdef LLAMA
    ggml_backend_load_all();
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = envint("NGL", 0);
    ggml_backend_dev_t cpu_only[] = {NULL};
    if (mp.n_gpu_layers == 0) mp.devices = cpu_only;
    mp.use_extra_bufts = envint("REPACK", 1);
    mp.load_mode = envint("LOAD", LLAMA_LOAD_MODE_MMAP);
    struct llama_model_tensor_buft_override overrides[2] = {0};
    if (envint("CPU_MOE", 0)) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        need(dev != NULL, "cpu device");
        overrides[0].pattern = ".*ffn_.*_exps.*";
        overrides[0].buft = ggml_backend_dev_buffer_type(dev);
        mp.tensor_buft_overrides = overrides;
    }
    model = llama_model_load_from_file(argv[1], mp);
    need(model != NULL, "model load");
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = envint("CTX", depth + gen + 512);
    cp.n_batch = 2048;
    cp.n_ubatch = envint("UBATCH", 512);
    cp.n_threads = envint("THREADS", 6);
    cp.n_threads_batch = envint("BTHREADS", cp.n_threads);
    cp.flash_attn_type = envint("FA", 0) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.swa_full = envint("SWA_FULL", 0);
    if (envint("KV8", 0)) cp.type_k = cp.type_v = GGML_TYPE_Q8_0;
    ctx = llama_init_from_model(model, cp);
    need(ctx != NULL, "context create");
    if (envint("POOLS", 0)) {
        struct ggml_threadpool_params gp = ggml_threadpool_params_default(cp.n_threads);
        struct ggml_threadpool_params bp = ggml_threadpool_params_default(cp.n_threads_batch);
        gp.strict_cpu = bp.strict_cpu = true;
        gp.poll = bp.poll = envint("POLL", 50);
        gp.paused = true;
        for (int i = 0; i < cp.n_threads; i++) gp.cpumask[envint("GEN_E", 0) ? 12 + i : 2 * i] = true;
        for (int i = 0; i < cp.n_threads_batch; i++) {
            int core = envint("BATCH_ALL", 0) ? (i < 6 ? 2 * i : 6 + i) : (envint("BATCH_E", 1) ? 12 + i : 2 * i);
            bp.cpumask[core] = true;
        }
        gen_pool = ggml_threadpool_new(&gp);
        batch_pool = ggml_threadpool_new(&bp);
        need(gen_pool && batch_pool, "threadpools");
        llama_attach_threadpool(ctx, gen_pool, batch_pool);
    }
    printf("{\"event\":\"config\",\"ngl\":%d,\"repack\":%d,\"ctx\":%u,\"ubatch\":%u,\"threads\":%d,\"batch_threads\":%d,\"fa\":%d,\"swa_full\":%d}\n", mp.n_gpu_layers, mp.use_extra_bufts, cp.n_ctx, cp.n_ubatch, cp.n_threads, cp.n_threads_batch, cp.flash_attn_type, cp.swa_full);
#else
    engine = xe_engine_open(argv[1]);
    session = xe_session_new(engine);
    printf("{\"event\":\"config\",\"ctx\":%d,\"workers\":%d,\"first_cpu\":%d}\n", xe_context_size(engine), xe_engine_worker_count(engine), xe_engine_worker_cpu(engine, 0));
#endif
    result("load", 0, 0, 0, t, c, 0);
    memory("loaded");
    t = now(); c = cpu();
    if (depth) sync_to(depth);
    if (gen) sync_to(depth + 1);
    result("warmup", 0, 0, depth + (gen > 0), t, c, 0);
    for (int rep = 0; rep < reps; rep++) {
        reset();
        t = now(); c = cpu();
        if (depth) sync_to(depth);
        if (depth) result("prefill", rep, 0, depth, t, c, 0);
        memory("prefilled");
        if (depth) printf("{\"event\":\"top1\",\"rep\":%d,\"depth\":%d,\"token\":%d}\n", rep, depth, best());
        if (depth && rep == 0 && getenv("LOGITS")) {
            FILE *dump = fopen(getenv("LOGITS"), "wb");
            need(dump && fwrite(logits(), sizeof(float), 262144, dump) == 262144 && fclose(dump) == 0, "logit dump");
        }
        t = now(); c = cpu();
        for (int i = 0; i < gen; i++) {
            double step = now();
            sync_to(depth + i + 1);
            double seconds = now() - step;
            printf("{\"event\":\"step\",\"rep\":%d,\"depth\":%d,\"i\":%d,\"seconds\":%.9f}\n", rep, depth, i, seconds);
        }
        if (gen) result("decode", rep, depth, gen, t, c, depth);
        memory("decoded");
        if (envint("EXTRAS", 0)) {
            t = now(); c = cpu();
            sync_to(depth + gen + 128);
            result("append128", rep, depth + gen, 128, t, c, depth + gen);
            int cut = depth / 2;
            int32_t original = tokens[cut];
            tokens[cut] = tokens[cut + 17];
            int reused = cut;
            t = now(); c = cpu();
#ifdef LLAMA
            llama_memory_t mem = llama_get_memory(ctx);
            if (llama_memory_seq_pos_min(mem, 0) > cut || !llama_memory_seq_rm(mem, 0, cut, -1)) {
                reset();
                reused = 0;
            } else position = cut;
            sync_to(depth);
#else
            xe_tokens prefix = {tokens, depth, 262144};
            xe_sync_report report;
            xe_session_sync_report(session, &prefix, &report);
            reused = report.reused;
#endif
            result("rewrite_half", rep, depth, depth - reused, t, c, reused);
            tokens[cut] = original;
        }
    }
    memory("done");
#ifdef LLAMA
    llama_free(ctx);
    if (gen_pool) ggml_threadpool_free(gen_pool);
    if (batch_pool) ggml_threadpool_free(batch_pool);
    llama_model_free(model);
    llama_backend_free();
#else
    xe_session_free(session);
    xe_engine_close(engine);
#endif
    return 0;
}
