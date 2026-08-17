#include "../xenolith.c"

static double prefill_rel_rms(const float *a, const float *b, size_t n) {
    double error = 0.0;
    double reference = 0.0;
    for (size_t i = 0; i < n; i++) {
        double difference = (double)a[i] - b[i];
        error += difference * difference;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static double prefill_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [rows]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    const int rows = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 32;
    const int width = XE_EMBD;
    const int columns = XE_Q_HEADS * XE_SWA_HEAD_DIM;
    const int blocks = width / 32;
    if (rows < 1 || rows > 512) xe_fatal("prefill projection rows out of range");
    xe_engine *e = xe_engine_open(model);
    float *input = xe_alloc(e, (size_t)rows * width * sizeof(*input),
                            XE_MEM_SHARED);
    float *row_scale = xe_alloc(e, rows * sizeof(*row_scale), XE_MEM_SHARED);
    xe_q8 q8 = {
        xe_alloc(e, (size_t)rows * width, XE_MEM_SHARED),
        xe_alloc(e, (size_t)rows * blocks * sizeof(*q8.d), XE_MEM_SHARED),
        xe_alloc(e, (size_t)rows * blocks * sizeof(*q8.sigma), XE_MEM_SHARED),
        rows * width
    };
    float *output = xe_alloc(e, (size_t)rows * columns * sizeof(*output),
                             XE_MEM_SHARED);
    float *normalized = xe_alloc(NULL, width * sizeof(*normalized), XE_MEM_HOST);
    int8_t *reference_qs = xe_alloc(NULL, (size_t)rows * width, XE_MEM_HOST);
    _Float16 *reference_d = xe_alloc(NULL, (size_t)rows * blocks
                                     * sizeof(*reference_d),
                                     XE_MEM_HOST);
    int16_t *reference_sigma = xe_alloc(NULL, (size_t)rows * blocks
                                        * sizeof(*reference_sigma),
                                        XE_MEM_HOST);
    for (int row = 0; row < rows; row++)
        xe_embed_decode(e, 2 + row, input + (size_t)row * width);

    double start = prefill_now();
    xe_prefill_input_append(e, input, e->layers[0].attn_norm, row_scale,
                            &q8, rows, width);
    xe_prefill_projection_append(e, &e->layers[0].attn_q, &q8, output,
                                 rows, columns, 1);
    xe_ze_check("zeCommandListHostSynchronize prefill projection test",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double cold_elapsed = prefill_now() - start;
    const int repetitions = 30;
    start = prefill_now();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        xe_prefill_input_append(e, input, e->layers[0].attn_norm, row_scale,
                                &q8, rows, width);
        xe_prefill_projection_append(e, &e->layers[0].attn_q, &q8, output,
                                     rows, columns, 1);
    }
    xe_ze_check("zeCommandListHostSynchronize prefill projection warm test",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double warm_elapsed = (prefill_now() - start) / repetitions;

    int q_mismatch = 0;
    int d_mismatch = 0;
    int sigma_mismatch = 0;
    int max_q_delta = 0;
    double scale_error = 0.0;
    double scale_reference = 0.0;
    double reconstruction_error = 0.0;
    double reconstruction_reference = 0.0;
    for (int row = 0; row < rows; row++) {
        const float *source = input + (size_t)row * width;
        float cpu_scale = xe_rms_scale_engine(e, source, width);
        double scale_difference = row_scale[row] - cpu_scale;
        scale_error += scale_difference * scale_difference;
        scale_reference += (double)cpu_scale * cpu_scale;
        xe_rmsnorm_scale(source, e->layers[0].attn_norm, width,
                         cpu_scale, normalized);
        xe_q8 reference = {
            reference_qs + (size_t)row * width,
            reference_d + (size_t)row * blocks,
            reference_sigma + (size_t)row * blocks, width
        };
        xe_quantize_q8(normalized, width, &reference);
        for (int i = 0; i < width; i++) {
            int block = i / 32;
            size_t index = (size_t)row * width + i;
            int delta = abs((int)reference.qs[i] - (int)q8.qs[index]);
            q_mismatch += delta != 0;
            if (delta > max_q_delta) max_q_delta = delta;
            double got = (double)q8.qs[index]
                         * (double)q8.d[(size_t)row * blocks + block];
            double want = (double)reference.qs[i] * (double)reference.d[block];
            double difference = got - want;
            reconstruction_error += difference * difference;
            reconstruction_reference += want * want;
        }
        for (int block = 0; block < blocks; block++) {
            size_t index = (size_t)row * blocks + block;
            d_mismatch += reference.d[block] != q8.d[index];
            sigma_mismatch += reference.sigma[block] != q8.sigma[index];
        }
    }

    float gpu_samples[256];
    float cpu_samples[256];
    float semantic_samples[256];
    int sample = 0;
    int sample_rows = rows < 8 ? rows : 8;
    for (int row = 0; row < sample_rows; row++) {
        xe_q8 row_q8 = {
            q8.qs + (size_t)row * width,
            q8.d + (size_t)row * blocks,
            q8.sigma + (size_t)row * blocks,
            width
        };
        xe_q8 semantic_q8 = {
            reference_qs + (size_t)row * width,
            reference_d + (size_t)row * blocks,
            reference_sigma + (size_t)row * blocks,
            width
        };
        for (int column = 0; column < columns; column += 128) {
            gpu_samples[sample] = output[(size_t)row * columns + column];
            cpu_samples[sample] = xe_q4_q8_dot(
                &e->layers[0].attn_q, column, width, &row_q8);
            semantic_samples[sample] = xe_q4_q8_dot(
                &e->layers[0].attn_q, column, width, &semantic_q8);
            sample++;
        }
    }
    double scale_rel = sqrt(scale_error / (scale_reference + 1e-30));
    double projection_rel = prefill_rel_rms(gpu_samples, cpu_samples, sample);
    double semantic_rel = prefill_rel_rms(gpu_samples, semantic_samples, sample);
    double reconstruction_rel = sqrt(reconstruction_error
                                     / (reconstruction_reference + 1e-30));
    int ok = scale_rel < 2e-6 && max_q_delta <= 1
             && reconstruction_rel < 5e-4 && projection_rel < 2e-5
             && semantic_rel < 2e-5;
    printf("prefill-production: M%d Q cold %.6f ms warm %.6f ms scale %.3e q8 %d/%d/%d max-delta %d reconstruction %.3e projection same-q8 %.3e semantic %.3e samples %d %s\n",
           rows, cold_elapsed * 1000.0, warm_elapsed * 1000.0,
           scale_rel, q_mismatch, d_mismatch,
           sigma_mismatch, max_q_delta,
           reconstruction_rel, projection_rel, semantic_rel, sample,
           ok ? "PASS" : "FAIL");

    xe_free(NULL, reference_sigma, XE_MEM_HOST);
    xe_free(NULL, reference_d, XE_MEM_HOST);
    xe_free(NULL, reference_qs, XE_MEM_HOST);
    xe_free(NULL, normalized, XE_MEM_HOST);
    xe_free(e, output, XE_MEM_SHARED);
    xe_free(e, q8.sigma, XE_MEM_SHARED);
    xe_free(e, q8.d, XE_MEM_SHARED);
    xe_free(e, q8.qs, XE_MEM_SHARED);
    xe_free(e, row_scale, XE_MEM_SHARED);
    xe_free(e, input, XE_MEM_SHARED);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
