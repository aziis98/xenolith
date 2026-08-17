#include "../xenolith.c"

static double layer_now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static double layer_rel(const float *a, const float *b, size_t n) {
    double error = 0.0;
    double reference = 0.0;
    for (size_t i = 0; i < n; i++) {
        double difference = (double)a[i] - b[i];
        error += difference * difference;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static int layer_run(xe_engine *e, int layer_index, int rows) {
    xe_prefill_workspace workspace;
    size_t workspace_size = xe_prefill_workspace_layout(&workspace, NULL);
    void *memory = xe_alloc(e, workspace_size, XE_MEM_SHARED);
    xe_prefill_workspace_layout(&workspace, memory);
    for (int row = 0; row < rows; row++)
        xe_embed_decode(e, 2 + row,
                        workspace.hidden[0] + (size_t)row * XE_EMBD);
    int dimension = XE_IS_GLOBAL(layer_index)
                    ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
    int q_width = XE_Q_HEADS * dimension;
    float *q_snapshot = xe_alloc(
        NULL, (size_t)rows * q_width * sizeof(*q_snapshot), XE_MEM_HOST);
    float *heads_snapshot = xe_alloc(
        NULL, (size_t)rows * q_width * sizeof(*heads_snapshot), XE_MEM_HOST);
    float *projection_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EMBD * sizeof(*projection_snapshot),
        XE_MEM_HOST);
    float *attention_snapshot = xe_alloc(
        NULL, (size_t)rows * XE_EMBD * sizeof(*attention_snapshot),
        XE_MEM_HOST);
    double start = layer_now();
    xe_prefill_attention_initial_append(e, layer_index, &workspace, rows);
    xe_ze_check("zeCommandListHostSynchronize prefill layer attention",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    memcpy(q_snapshot, workspace.q_heads,
           (size_t)rows * q_width * sizeof(*q_snapshot));
    memcpy(heads_snapshot, workspace.attention_heads,
           (size_t)rows * q_width * sizeof(*heads_snapshot));
    memcpy(projection_snapshot, workspace.attention_projection,
           (size_t)rows * XE_EMBD * sizeof(*projection_snapshot));
    memcpy(attention_snapshot, workspace.attention_output,
           (size_t)rows * XE_EMBD * sizeof(*attention_snapshot));
    xe_prefill_ffn_append(e, layer_index, &workspace, rows);
    xe_ze_check("zeCommandListHostSynchronize prefill layer FFN",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double cold_ms = (layer_now() - start) * 1000.0;
    const int repetitions = rows == 512 ? 5 : 10;
    start = layer_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        xe_prefill_layer_initial_append(e, layer_index, &workspace, rows);
    xe_ze_check("zeCommandListHostSynchronize prefill layer timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double warm_ms = (layer_now() - start) * 1000.0 / repetitions;
    int finite = 1;
    for (size_t i = 0; i < (size_t)rows * XE_EMBD; i++)
        finite &= isfinite(workspace.hidden[1][i]);
    double error = 0.0;
    double attention_error = 0.0;
    double router_input_error = 0.0;
    double q_error = 0.0;
    double heads_error = 0.0;
    double projection_error = 0.0;
    double q_projection_error = 0.0;
    double q_post_error = 0.0;
    double q_reference_error = 0.0;
    double q_reference_row0_error = 0.0;
    double reference_projection_error = 0.0;
    int input_q8_mismatches = 0;
    int input_q8_delta = 0;
    int route_mismatches = 0;
    int route_set_mismatches = 0;
    float mismatched_margin = 0.0f;
    if (rows == 32) {
        ref_state reference_state = {0};
        ref_state_init(&reference_state, rows);
        reference_state.q8 = 2;
        float *reference_output = xe_alloc(
            NULL, (size_t)rows * XE_EMBD * sizeof(*reference_output),
            XE_MEM_HOST);
        float *reference_attention = xe_alloc(
            NULL, (size_t)rows * XE_EMBD * sizeof(*reference_attention),
            XE_MEM_HOST);
        float *reference_router = xe_alloc(
            NULL, (size_t)rows * XE_EMBD * sizeof(*reference_router),
            XE_MEM_HOST);
        float *reference_q = xe_alloc(
            NULL, (size_t)rows * q_width * sizeof(*reference_q), XE_MEM_HOST);
        float *q_post_reference = xe_alloc(
            NULL, (size_t)rows * q_width * sizeof(*q_post_reference),
            XE_MEM_HOST);
        float *reference_heads = xe_alloc(
            NULL, (size_t)rows * q_width * sizeof(*reference_heads),
            XE_MEM_HOST);
        float *reference_projection = xe_alloc(
            NULL, (size_t)rows * XE_EMBD * sizeof(*reference_projection),
            XE_MEM_HOST);
        int8_t reference_qs[XE_EMBD];
        _Float16 reference_d[XE_EMBD / 32];
        int16_t reference_sigma[XE_EMBD / 32];
        float normalized[XE_EMBD];
        float q_gpu_samples[256];
        float q_cpu_samples[256];
        float q_ref_samples[256];
        int q_samples = 0;
        const int q_sample_rows[8] = { 0, 1, 7, 8, 15, 16, 24, 31 };
        for (int sample_row = 0; sample_row < 8; sample_row++) {
            int row = q_sample_rows[sample_row];
            const float *input = workspace.hidden[0]
                                 + (size_t)row * XE_EMBD;
            float scale = xe_rms_scale_scalar(input, XE_EMBD);
            xe_rmsnorm_scale(input, e->layers[layer_index].attn_norm,
                             XE_EMBD, scale, normalized);
            xe_q8 reference_q8 = {
                reference_qs, reference_d, reference_sigma, XE_EMBD
            };
            xe_quantize_q8(normalized, XE_EMBD, &reference_q8);
            float raw_reference[XE_Q_HEADS * XE_GLOBAL_HEAD_DIM];
            ref_matvec_q4_q8(&e->layers[layer_index].attn_q, 0, XE_EMBD,
                             q_width, normalized, 1, raw_reference);
            xe_q8 gpu_q8 = {
                workspace.attention_input.qs + (size_t)row * XE_EMBD,
                workspace.attention_input.d
                    + (size_t)row * (XE_EMBD / 32),
                workspace.attention_input.sigma
                    + (size_t)row * (XE_EMBD / 32), XE_EMBD
            };
            for (int column = 0; column < XE_EMBD; column++) {
                int delta = abs((int)gpu_q8.qs[column]
                                - (int)reference_qs[column]);
                input_q8_mismatches += delta != 0;
                if (delta > input_q8_delta) input_q8_delta = delta;
            }
            for (int column = 0; column < q_width; column += 256) {
                q_gpu_samples[q_samples] = workspace.q_projection[
                    (size_t)row * q_width + column];
                q_cpu_samples[q_samples++] = xe_q4_q8_dot(
                    &e->layers[layer_index].attn_q, column, XE_EMBD,
                    &gpu_q8);
                q_ref_samples[q_samples - 1] = raw_reference[column];
            }
        }
        q_projection_error = layer_rel(q_gpu_samples, q_cpu_samples,
                                       q_samples);
        reference_projection_error = layer_rel(q_gpu_samples, q_ref_samples,
                                               q_samples);
        for (int row = 0; row < rows; row++)
            for (int head = 0; head < XE_Q_HEADS; head++) {
                const float *source = workspace.q_projection
                    + ((size_t)row * XE_Q_HEADS + head) * dimension;
                float scale = xe_rms_scale_scalar(source, dimension);
                size_t base = ((size_t)head * rows + row) * dimension;
                for (int d = 0; d < dimension / 2; d++) {
                    float lo = source[d] * scale
                               * e->layers[layer_index].q_norm[d];
                    float hi = source[d + dimension / 2] * scale
                               * e->layers[layer_index]
                                  .q_norm[d + dimension / 2];
                    const float *rope_cos = XE_IS_GLOBAL(layer_index)
                        ? workspace.rope_global_cos : workspace.rope_swa_cos;
                    const float *rope_sin = XE_IS_GLOBAL(layer_index)
                        ? workspace.rope_global_sin : workspace.rope_swa_sin;
                    float cosine = rope_cos[
                        (size_t)row * (dimension / 2) + d];
                    float sine = rope_sin[
                        (size_t)row * (dimension / 2) + d];
                    q_post_reference[base + d] = lo * cosine - hi * sine;
                    q_post_reference[base + d + dimension / 2] =
                        lo * sine + hi * cosine;
                }
            }
        for (int row = 0; row < rows; row++) {
            ref_embed(e, 2 + row, reference_state.hidden);
            ref_attention_half(e, layer_index, row, &reference_state);
            for (int head = 0; head < XE_Q_HEADS; head++) {
                memcpy(reference_q + ((size_t)head * rows + row) * dimension,
                       reference_state.q + (size_t)head * dimension,
                       (size_t)dimension * sizeof(float));
                memcpy(reference_heads
                           + ((size_t)head * rows + row) * dimension,
                       reference_state.attn_heads + (size_t)head * dimension,
                       (size_t)dimension * sizeof(float));
            }
            memcpy(reference_projection + (size_t)row * XE_EMBD,
                   reference_state.attn_proj, XE_EMBD * sizeof(float));
            memcpy(reference_attention + (size_t)row * XE_EMBD,
                   reference_state.attn_out, XE_EMBD * sizeof(float));
            ref_ffn_half(e, layer_index, &reference_state);
            memcpy(reference_output + (size_t)row * XE_EMBD,
                   reference_state.hidden, XE_EMBD * sizeof(float));
            memcpy(reference_router + (size_t)row * XE_EMBD,
                   reference_state.router_in, XE_EMBD * sizeof(float));
            float best_value[XE_EXPERTS_USED + 1];
            int best_index[XE_EXPERTS_USED + 1];
            for (int slot = 0; slot <= XE_EXPERTS_USED; slot++) {
                best_value[slot] = -INFINITY;
                best_index[slot] = -1;
            }
            for (int expert = 0; expert < XE_EXPERTS; expert++) {
                float value = reference_state.router_probs[expert];
                if (value <= best_value[XE_EXPERTS_USED]) continue;
                int slot = XE_EXPERTS_USED;
                while (slot > 0 && best_value[slot - 1] < value) {
                    best_value[slot] = best_value[slot - 1];
                    best_index[slot] = best_index[slot - 1];
                    slot--;
                }
                best_value[slot] = value;
                best_index[slot] = expert;
            }
            int row_mismatch = 0;
            int row_set_mismatch = 0;
            for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
                int gpu_expert = workspace.route_expert[
                    row * XE_EXPERTS_USED + slot];
                int mismatch = gpu_expert != best_index[slot];
                route_mismatches += mismatch;
                row_mismatch |= mismatch;
                int found = 0;
                for (int reference_slot = 0;
                     reference_slot < XE_EXPERTS_USED; reference_slot++)
                    found |= gpu_expert == best_index[reference_slot];
                route_set_mismatches += !found;
                row_set_mismatch |= !found;
            }
            if (row_mismatch && row_set_mismatch) {
                float margin = best_value[XE_EXPERTS_USED - 1]
                               - best_value[XE_EXPERTS_USED];
                if (margin > mismatched_margin) mismatched_margin = margin;
            }
        }
        error = layer_rel(workspace.hidden[1], reference_output,
                          (size_t)rows * XE_EMBD);
        attention_error = layer_rel(attention_snapshot,
                                    reference_attention,
                                    (size_t)rows * XE_EMBD);
        router_input_error = layer_rel(workspace.router_input,
                                       reference_router,
                                       (size_t)rows * XE_EMBD);
        q_error = layer_rel(q_snapshot, reference_q,
                            (size_t)rows * q_width);
        q_post_error = layer_rel(q_snapshot, q_post_reference,
                                 (size_t)rows * q_width);
        q_reference_error = layer_rel(q_post_reference, reference_q,
                                      (size_t)rows * q_width);
        float q_post_row0[XE_Q_HEADS * XE_GLOBAL_HEAD_DIM];
        float q_reference_row0[XE_Q_HEADS * XE_GLOBAL_HEAD_DIM];
        for (int head = 0; head < XE_Q_HEADS; head++) {
            memcpy(q_post_row0 + (size_t)head * dimension,
                   q_post_reference + (size_t)head * rows * dimension,
                   (size_t)dimension * sizeof(float));
            memcpy(q_reference_row0 + (size_t)head * dimension,
                   reference_q + (size_t)head * rows * dimension,
                   (size_t)dimension * sizeof(float));
        }
        q_reference_row0_error = layer_rel(q_post_row0, q_reference_row0,
                                           q_width);
        heads_error = layer_rel(heads_snapshot, reference_heads,
                                (size_t)rows * q_width);
        double projection_difference = 0.0;
        double projection_reference = 0.0;
        for (size_t i = 0; i < (size_t)rows * XE_EMBD; i++) {
            double got = (double)attention_snapshot[i]
                         - workspace.hidden[0][i];
            double want = reference_projection[i];
            double difference = got - want;
            projection_difference += difference * difference;
            projection_reference += want * want;
        }
        projection_error = sqrt(projection_difference
                                / (projection_reference + 1e-30));
        xe_free(NULL, reference_projection, XE_MEM_HOST);
        xe_free(NULL, reference_heads, XE_MEM_HOST);
        xe_free(NULL, reference_q, XE_MEM_HOST);
        xe_free(NULL, q_post_reference, XE_MEM_HOST);
        xe_free(NULL, reference_router, XE_MEM_HOST);
        xe_free(NULL, reference_attention, XE_MEM_HOST);
        xe_free(NULL, reference_output, XE_MEM_HOST);
        ref_state_free(&reference_state);
    }
    int routes_ok = route_set_mismatches == 0 || mismatched_margin < 5e-4f;
    int ok = finite && (rows != 32 || (routes_ok && error < 5e-3));
    printf("prefill-layer: L%d %s M%d arena %.3f MiB cold %.6f ms warm %.6f ms input-q8 %d delta %d q-projection %.3e ref-projection %.3e q-post %.3e q-reference %.3e/%.3e q %.3e heads %.3e projection %.3e attention %.3e router-input %.3e routes %d/%d margin %.3e rel %.3e %s\n",
           layer_index, XE_IS_GLOBAL(layer_index) ? "global" : "swa", rows,
           (double)workspace_size / (1024.0 * 1024.0), cold_ms, warm_ms,
           input_q8_mismatches, input_q8_delta, q_projection_error,
           reference_projection_error, q_post_error, q_reference_error,
           q_reference_row0_error, q_error, heads_error,
           projection_error, attention_error,
           router_input_error, route_mismatches, route_set_mismatches,
           (double)mismatched_margin,
           error,
           ok ? "PASS" : "FAIL");
    xe_free(NULL, attention_snapshot, XE_MEM_HOST);
    xe_free(NULL, projection_snapshot, XE_MEM_HOST);
    xe_free(NULL, heads_snapshot, XE_MEM_HOST);
    xe_free(NULL, q_snapshot, XE_MEM_HOST);
    xe_free(e, memory, XE_MEM_SHARED);
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [rows]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int rows = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 32;
    if (rows != 32 && rows != 512)
        xe_fatal("prefill layer test supports M32 or M512");
    xe_engine *e = xe_engine_open(model);
    int ok = layer_run(e, 0, rows) & layer_run(e, 5, rows);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
