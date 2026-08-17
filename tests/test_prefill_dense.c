#include "../xenolith.c"

static double dense_now(void) {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static double dense_rel(const float *a, const float *b, size_t n) {
    double error = 0.0;
    double reference = 0.0;
    for (size_t i = 0; i < n; i++) {
        double difference = (double)a[i] - b[i];
        error += difference * difference;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static xe_q8 dense_q8_alloc(xe_engine *e, int rows, int width) {
    int blocks = width / 32;
    return (xe_q8) {
        xe_alloc(e, (size_t)rows * width, XE_MEM_SHARED),
        xe_alloc(e, (size_t)rows * blocks * sizeof(_Float16), XE_MEM_SHARED),
        xe_alloc(e, (size_t)rows * blocks * sizeof(int16_t), XE_MEM_SHARED),
        rows * width
    };
}

static void dense_q8_free(xe_engine *e, xe_q8 *q) {
    xe_free(e, q->sigma, XE_MEM_SHARED);
    xe_free(e, q->d, XE_MEM_SHARED);
    xe_free(e, q->qs, XE_MEM_SHARED);
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [rows]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int rows = argc > 2 ? (int)strtol(argv[2], NULL, 10) : 512;
    if (rows < 1 || rows > 512) xe_fatal("prefill dense rows out of range");
    xe_engine *e = xe_engine_open(model);
    const xe_layer *layer = &e->layers[0];
    size_t input_n = (size_t)rows * XE_EMBD;
    size_t hidden_n = (size_t)rows * XE_DENSE_FFN;
    float *input = xe_alloc(e, input_n * sizeof(*input), XE_MEM_SHARED);
    float *row_scale = xe_alloc(e, rows * sizeof(*row_scale), XE_MEM_SHARED);
    xe_q8 dense_input = dense_q8_alloc(e, rows, XE_EMBD);
    xe_q8 moe_input = dense_q8_alloc(e, rows, XE_EMBD);
    float *router_input = xe_alloc(e, input_n * sizeof(*router_input),
                                   XE_MEM_SHARED);
    float *router_logits = xe_alloc(
        e, (size_t)rows * XE_EXPERTS * sizeof(*router_logits), XE_MEM_SHARED);
    int *route_expert = xe_alloc(
        e, (size_t)rows * XE_EXPERTS_USED * sizeof(*route_expert),
        XE_MEM_SHARED);
    float *route_weight = xe_alloc(
        e, (size_t)rows * XE_EXPERTS_USED * sizeof(*route_weight),
        XE_MEM_SHARED);
    int routes_count = rows * XE_EXPERTS_USED;
    xe_q8 packed_moe = dense_q8_alloc(e, routes_count, XE_EMBD);
    xe_prefill_routes routes = {
        xe_alloc(e, XE_EXPERTS * sizeof(int), XE_MEM_SHARED),
        xe_alloc(e, (XE_EXPERTS + 1) * sizeof(int), XE_MEM_SHARED),
        xe_alloc(e, XE_EXPERTS * sizeof(int), XE_MEM_SHARED),
        xe_alloc(e, 256 * sizeof(int), XE_MEM_SHARED),
        xe_alloc(e, 256 * sizeof(int), XE_MEM_SHARED),
        xe_alloc(e, routes_count * sizeof(int), XE_MEM_SHARED),
        xe_alloc(e, routes_count * sizeof(int), XE_MEM_SHARED)
    };
    float *expert_gate_up = xe_alloc(
        e, (size_t)routes_count * 2 * XE_EXPERT_FFN
           * sizeof(*expert_gate_up), XE_MEM_SHARED);
    xe_q8 expert_activation = dense_q8_alloc(
        e, routes_count, XE_EXPERT_FFN);
    float *expert_down = xe_alloc(
        e, (size_t)routes_count * XE_EMBD * sizeof(*expert_down),
        XE_MEM_SHARED);
    float *moe_output = xe_alloc(
        e, input_n * sizeof(*moe_output), XE_MEM_SHARED);
    float *layer_output = xe_alloc(
        e, input_n * sizeof(*layer_output), XE_MEM_SHARED);
    float *gate = xe_alloc(e, hidden_n * sizeof(*gate), XE_MEM_SHARED);
    float *up = xe_alloc(e, hidden_n * sizeof(*up), XE_MEM_SHARED);
    xe_q8 activation = dense_q8_alloc(e, rows, XE_DENSE_FFN);
    float *down = xe_alloc(e, input_n * sizeof(*down), XE_MEM_SHARED);
    for (int row = 0; row < rows; row++)
        xe_embed_decode(e, 2 + row, input + (size_t)row * XE_EMBD);
    xe_prefill_rms_append(e, input, row_scale, rows, XE_EMBD);
    xe_prefill_ffn_input_append(e, layer, input, row_scale, &dense_input,
                                &moe_input, router_input, rows);
    xe_prefill_router_append(e, layer, router_input, router_logits,
                             route_expert, route_weight, rows);
    xe_prefill_route_append(e, &moe_input, &packed_moe, route_expert,
                            &routes, rows);
    xe_prefill_grouped_projection_append(e, &layer->gate_up_exps,
                                          &packed_moe, expert_gate_up,
                                          &routes, 2 * XE_EXPERT_FFN);
    xe_prefill_expert_geglu_append(e, expert_gate_up, &expert_activation,
                                    routes_count);
    xe_prefill_grouped_projection_append(e, &layer->down_exps,
                                          &expert_activation, expert_down,
                                          &routes, XE_EMBD);
    xe_prefill_route_reduce_append(e, layer, expert_down, route_weight,
                                    route_expert, &routes, moe_output, rows);
    xe_prefill_ffn_finish_append(e, layer, down, moe_output, input,
                                 layer_output, rows);
    xe_prefill_projection_append(e, &layer->ffn_gate, &dense_input, gate,
                                 rows, XE_DENSE_FFN, rows > 32);
    xe_prefill_projection_append(e, &layer->ffn_up, &dense_input, up,
                                 rows, XE_DENSE_FFN, rows > 32);
    xe_prefill_geglu_q8_append(e, gate, up, &activation, rows, XE_DENSE_FFN);
    xe_prefill_projection_append(e, &layer->ffn_down, &activation, down,
                                 rows, XE_EMBD, rows > 32);
    xe_ze_check("zeCommandListHostSynchronize prefill dense",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    const int repetitions = 10;
    double start = dense_now();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        xe_prefill_rms_append(e, input, row_scale, rows, XE_EMBD);
        xe_prefill_ffn_input_append(e, layer, input, row_scale, &dense_input,
                                    &moe_input, router_input, rows);
        xe_prefill_router_append(e, layer, router_input, router_logits,
                                 route_expert, route_weight, rows);
        xe_prefill_route_append(e, &moe_input, &packed_moe, route_expert,
                                &routes, rows);
        xe_prefill_grouped_projection_append(e, &layer->gate_up_exps,
                                              &packed_moe, expert_gate_up,
                                              &routes, 2 * XE_EXPERT_FFN);
        xe_prefill_expert_geglu_append(e, expert_gate_up,
                                        &expert_activation, routes_count);
        xe_prefill_grouped_projection_append(e, &layer->down_exps,
                                              &expert_activation, expert_down,
                                              &routes, XE_EMBD);
        xe_prefill_route_reduce_append(e, layer, expert_down, route_weight,
                                        route_expert, &routes, moe_output,
                                        rows);
        xe_prefill_ffn_finish_append(e, layer, down, moe_output, input,
                                     layer_output, rows);
        xe_prefill_projection_append(e, &layer->ffn_gate, &dense_input, gate,
                                     rows, XE_DENSE_FFN, rows > 32);
        xe_prefill_projection_append(e, &layer->ffn_up, &dense_input, up,
                                     rows, XE_DENSE_FFN, rows > 32);
        xe_prefill_geglu_q8_append(e, gate, up, &activation, rows,
                                   XE_DENSE_FFN);
        xe_prefill_projection_append(e, &layer->ffn_down, &activation, down,
                                     rows, XE_EMBD, rows > 32);
    }
    xe_ze_check("zeCommandListHostSynchronize prefill dense timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double elapsed_ms = (dense_now() - start) * 1000.0 / repetitions;
    const int router_repetitions = 30;
    start = dense_now();
    for (int repetition = 0; repetition < router_repetitions; repetition++)
        xe_prefill_router_append(e, layer, router_input, router_logits,
                                 route_expert, route_weight, rows);
    xe_ze_check("zeCommandListHostSynchronize prefill router timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double router_ms = (dense_now() - start) * 1000.0 / router_repetitions;
    const int routing_repetitions = 30;
    start = dense_now();
    for (int repetition = 0; repetition < routing_repetitions; repetition++)
        xe_prefill_route_append(e, &moe_input, &packed_moe, route_expert,
                                &routes, rows);
    xe_ze_check("zeCommandListHostSynchronize prefill routing timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double routing_ms = (dense_now() - start) * 1000.0 / routing_repetitions;
    const int moe_repetitions = 5;
    start = dense_now();
    for (int repetition = 0; repetition < moe_repetitions; repetition++) {
        xe_prefill_grouped_projection_append(e, &layer->gate_up_exps,
                                              &packed_moe, expert_gate_up,
                                              &routes, 2 * XE_EXPERT_FFN);
        xe_prefill_expert_geglu_append(e, expert_gate_up,
                                        &expert_activation, routes_count);
        xe_prefill_grouped_projection_append(e, &layer->down_exps,
                                              &expert_activation, expert_down,
                                              &routes, XE_EMBD);
        xe_prefill_route_reduce_append(e, layer, expert_down, route_weight,
                                        route_expert, &routes, moe_output,
                                        rows);
    }
    xe_ze_check("zeCommandListHostSynchronize prefill moe timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double moe_ms = (dense_now() - start) * 1000.0 / moe_repetitions;
    const int finish_repetitions = 30;
    start = dense_now();
    for (int repetition = 0; repetition < finish_repetitions; repetition++)
        xe_prefill_ffn_finish_append(e, layer, down, moe_output, input,
                                     layer_output, rows);
    xe_ze_check("zeCommandListHostSynchronize prefill finish timing",
                zeCommandListHostSynchronize(e->gpu.commands, UINT64_MAX));
    double finish_ms = (dense_now() - start) * 1000.0 / finish_repetitions;
    int8_t *reference_qs = xe_alloc(NULL, input_n, XE_MEM_HOST);
    _Float16 *reference_d = xe_alloc(
        NULL, (size_t)rows * (XE_EMBD / 32) * sizeof(*reference_d), XE_MEM_HOST);
    int16_t *reference_sigma = xe_alloc(
        NULL, (size_t)rows * (XE_EMBD / 32) * sizeof(*reference_sigma),
        XE_MEM_HOST);
    float *semantic = xe_alloc(NULL, XE_EMBD * sizeof(*semantic), XE_MEM_HOST);
    float *router_reference = xe_alloc(NULL, input_n * sizeof(*router_reference),
                                       XE_MEM_HOST);
    int dense_mismatch = 0;
    int dense_max_delta = 0;
    int moe_mismatch = 0;
    int moe_max_delta = 0;
    for (int row = 0; row < rows; row++) {
        float scale = xe_rms_scale_scalar(input + (size_t)row * XE_EMBD,
                                          XE_EMBD);
        xe_rmsnorm_scale(input + (size_t)row * XE_EMBD, layer->ffn_norm,
                         XE_EMBD, scale, semantic);
        xe_q8 reference_q8 = {
            reference_qs + (size_t)row * XE_EMBD,
            reference_d + (size_t)row * (XE_EMBD / 32),
            reference_sigma + (size_t)row * (XE_EMBD / 32), XE_EMBD
        };
        xe_quantize_q8(semantic, XE_EMBD, &reference_q8);
        for (int column = 0; column < XE_EMBD; column++) {
            size_t index = (size_t)row * XE_EMBD + column;
            int delta = abs((int)dense_input.qs[index]
                            - (int)reference_q8.qs[column]);
            dense_mismatch += delta != 0;
            if (delta > dense_max_delta) dense_max_delta = delta;
        }
        xe_rmsnorm_scale(input + (size_t)row * XE_EMBD,
                         layer->pre_ffw_norm2, XE_EMBD, scale, semantic);
        xe_quantize_q8(semantic, XE_EMBD, &reference_q8);
        for (int column = 0; column < XE_EMBD; column++) {
            size_t index = (size_t)row * XE_EMBD + column;
            int delta = abs((int)moe_input.qs[index]
                            - (int)reference_q8.qs[column]);
            moe_mismatch += delta != 0;
            if (delta > moe_max_delta) moe_max_delta = delta;
            router_reference[index] = input[index] * scale
                * (1.0f / sqrtf((float)XE_EMBD)) * layer->router_scale[column];
        }
    }
    double router_error = dense_rel(router_input, router_reference, input_n);
    float *router_logits_reference = xe_alloc(
        NULL, (size_t)rows * XE_EXPERTS * sizeof(*router_logits_reference),
        XE_MEM_HOST);
    int route_mismatch = 0;
    double route_weight_error = 0.0;
    double route_weight_reference = 0.0;
    float minimum_margin = INFINITY;
    for (int row = 0; row < rows; row++) {
        float best_value[9];
        int best_index[9];
        for (int slot = 0; slot < 9; slot++) {
            best_value[slot] = -INFINITY;
            best_index[slot] = -1;
        }
        for (int expert = 0; expert < XE_EXPERTS; expert++) {
            float value = xe_dot_f32_avx(
                layer->router_w + (size_t)expert * XE_EMBD,
                router_input + (size_t)row * XE_EMBD, XE_EMBD);
            router_logits_reference[(size_t)row * XE_EXPERTS + expert] = value;
            if (value <= best_value[8]) continue;
            int slot = 8;
            while (slot > 0 && best_value[slot - 1] < value) {
                best_value[slot] = best_value[slot - 1];
                best_index[slot] = best_index[slot - 1];
                slot--;
            }
            best_value[slot] = value;
            best_index[slot] = expert;
        }
        float margin = best_value[7] - best_value[8];
        if (margin < minimum_margin) minimum_margin = margin;
        float weight_sum = 0.0f;
        float weights[8];
        for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
            weights[slot] = expf(best_value[slot] - best_value[0]);
            weight_sum += weights[slot];
        }
        for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
            size_t index = (size_t)row * XE_EXPERTS_USED + slot;
            route_mismatch += route_expert[index] != best_index[slot];
            double want = weights[slot] / weight_sum;
            double difference = (double)route_weight[index] - want;
            route_weight_error += difference * difference;
            route_weight_reference += want * want;
        }
    }
    double router_logits_error = dense_rel(
        router_logits, router_logits_reference, (size_t)rows * XE_EXPERTS);
    double route_weights_rel = sqrt(route_weight_error
                                    / (route_weight_reference + 1e-30));
    int routing_errors = routes.token_offset[0] != 0
                         || routes.token_offset[XE_EXPERTS] != routes_count;
    int tiles = 0;
    for (int expert = 0; expert < XE_EXPERTS; expert++) {
        routing_errors += routes.token_offset[expert + 1]
                          - routes.token_offset[expert]
                          != routes.expert_count[expert];
        tiles += (routes.expert_count[expert] + 31) / 32;
    }
    for (int tile = 0; tile < 256; tile++) {
        if (tile < tiles) {
            int expert = routes.tile_expert[tile];
            routing_errors += expert < 0 || expert >= XE_EXPERTS;
            routing_errors += routes.tile_m0[tile] < 0
                              || (routes.tile_m0[tile] & 31) != 0;
        } else {
            routing_errors += routes.tile_expert[tile] != -1;
        }
    }
    int blocks = XE_EMBD / 32;
    for (int packed = 0; packed < routes_count; packed++) {
        int route = routes.packed_route[packed];
        routing_errors += route < 0 || route >= routes_count;
        if (route < 0 || route >= routes_count) continue;
        routing_errors += routes.route_packed[route] != packed;
        int token = route >> 3;
        routing_errors += memcmp(
            packed_moe.qs + (size_t)packed * XE_EMBD,
            moe_input.qs + (size_t)token * XE_EMBD, XE_EMBD) != 0;
        routing_errors += memcmp(
            packed_moe.d + (size_t)packed * blocks,
            moe_input.d + (size_t)token * blocks,
            (size_t)blocks * sizeof(*packed_moe.d)) != 0;
        routing_errors += memcmp(
            packed_moe.sigma + (size_t)packed * blocks,
            moe_input.sigma + (size_t)token * blocks,
            (size_t)blocks * sizeof(*packed_moe.sigma)) != 0;
    }
    int8_t *activation_reference_qs = xe_alloc(NULL, hidden_n, XE_MEM_HOST);
    _Float16 *activation_reference_d = xe_alloc(
        NULL, (size_t)rows * (XE_DENSE_FFN / 32)
              * sizeof(*activation_reference_d), XE_MEM_HOST);
    int16_t *activation_reference_sigma = xe_alloc(
        NULL, (size_t)rows * (XE_DENSE_FFN / 32)
              * sizeof(*activation_reference_sigma), XE_MEM_HOST);
    float *activated = xe_alloc(NULL, XE_DENSE_FFN * sizeof(*activated),
                                XE_MEM_HOST);
    int activation_mismatch = 0;
    int activation_max_delta = 0;
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < XE_DENSE_FFN; column++)
            activated[column] = xe_gelu_fp16_value(
                gate[(size_t)row * XE_DENSE_FFN + column])
                * up[(size_t)row * XE_DENSE_FFN + column];
        xe_q8 reference_q8 = {
            activation_reference_qs + (size_t)row * XE_DENSE_FFN,
            activation_reference_d + (size_t)row * (XE_DENSE_FFN / 32),
            activation_reference_sigma + (size_t)row * (XE_DENSE_FFN / 32),
            XE_DENSE_FFN
        };
        xe_quantize_q8(activated, XE_DENSE_FFN, &reference_q8);
        for (int column = 0; column < XE_DENSE_FFN; column++) {
            size_t index = (size_t)row * XE_DENSE_FFN + column;
            int delta = abs((int)activation.qs[index]
                            - (int)reference_q8.qs[column]);
            activation_mismatch += delta != 0;
            if (delta > activation_max_delta) activation_max_delta = delta;
        }
    }
    float gpu_samples[256];
    float cpu_samples[256];
    int samples = 0;
    int sample_rows = rows < 4 ? rows : 4;
    for (int row = 0; row < sample_rows; row++) {
        xe_q8 row_q8 = {
            dense_input.qs + (size_t)row * XE_EMBD,
            dense_input.d + (size_t)row * (XE_EMBD / 32),
            dense_input.sigma + (size_t)row * (XE_EMBD / 32), XE_EMBD
        };
        for (int column = 0; column < XE_DENSE_FFN; column += 128) {
            gpu_samples[samples] = gate[(size_t)row * XE_DENSE_FFN + column];
            cpu_samples[samples++] = xe_q4_q8_dot(
                &layer->ffn_gate, column, XE_EMBD, &row_q8);
            gpu_samples[samples] = up[(size_t)row * XE_DENSE_FFN + column];
            cpu_samples[samples++] = xe_q4_q8_dot(
                &layer->ffn_up, column, XE_EMBD, &row_q8);
        }
    }
    double gate_up_error = dense_rel(gpu_samples, cpu_samples, samples);
    samples = 0;
    for (int row = 0; row < sample_rows; row++) {
        xe_q8 row_q8 = {
            activation.qs + (size_t)row * XE_DENSE_FFN,
            activation.d + (size_t)row * (XE_DENSE_FFN / 32),
            activation.sigma + (size_t)row * (XE_DENSE_FFN / 32),
            XE_DENSE_FFN
        };
        for (int column = 0; column < XE_EMBD; column += 128) {
            gpu_samples[samples] = down[(size_t)row * XE_EMBD + column];
            cpu_samples[samples++] = xe_q4_q8_dot(
                &layer->ffn_down, column, XE_DENSE_FFN, &row_q8);
        }
    }
    double down_error = dense_rel(gpu_samples, cpu_samples, samples);
    samples = 0;
    int expert_sample_rows = routes_count < 8 ? routes_count : 8;
    for (int packed = 0; packed < expert_sample_rows; packed++) {
        int expert = route_expert[routes.packed_route[packed]];
        xe_q8 row_q8 = {
            packed_moe.qs + (size_t)packed * XE_EMBD,
            packed_moe.d + (size_t)packed * (XE_EMBD / 32),
            packed_moe.sigma + (size_t)packed * (XE_EMBD / 32), XE_EMBD
        };
        for (int column = 0; column < 2 * XE_EXPERT_FFN; column += 128) {
            gpu_samples[samples] = expert_gate_up[
                (size_t)packed * 2 * XE_EXPERT_FFN + column];
            cpu_samples[samples++] = xe_q4_q8_dot(
                &layer->gate_up_exps,
                (uint64_t)expert * 2 * XE_EXPERT_FFN + column,
                XE_EMBD, &row_q8);
        }
    }
    double expert_gate_error = dense_rel(gpu_samples, cpu_samples, samples);
    int expert_activation_mismatch = 0;
    int expert_activation_delta = 0;
    int expert_activation_rows = routes_count < 64 ? routes_count : 64;
    for (int packed = 0; packed < expert_activation_rows; packed++) {
        for (int column = 0; column < XE_EXPERT_FFN; column++)
            activated[column] = xe_gelu_fp16_value(
                expert_gate_up[(size_t)packed * 2 * XE_EXPERT_FFN + column])
                * expert_gate_up[(size_t)packed * 2 * XE_EXPERT_FFN
                                 + XE_EXPERT_FFN + column];
        xe_q8 reference_q8 = {
            activation_reference_qs,
            activation_reference_d,
            activation_reference_sigma, XE_EXPERT_FFN
        };
        xe_quantize_q8(activated, XE_EXPERT_FFN, &reference_q8);
        for (int column = 0; column < XE_EXPERT_FFN; column++) {
            int delta = abs((int)expert_activation.qs[
                                (size_t)packed * XE_EXPERT_FFN + column]
                            - (int)reference_q8.qs[column]);
            expert_activation_mismatch += delta != 0;
            if (delta > expert_activation_delta)
                expert_activation_delta = delta;
        }
    }
    samples = 0;
    for (int packed = 0; packed < expert_sample_rows; packed++) {
        int expert = route_expert[routes.packed_route[packed]];
        xe_q8 row_q8 = {
            expert_activation.qs + (size_t)packed * XE_EXPERT_FFN,
            expert_activation.d
                + (size_t)packed * (XE_EXPERT_FFN / 32),
            expert_activation.sigma
                + (size_t)packed * (XE_EXPERT_FFN / 32), XE_EXPERT_FFN
        };
        for (int column = 0; column < XE_EMBD; column += 256) {
            gpu_samples[samples] = expert_down[
                (size_t)packed * XE_EMBD + column];
            cpu_samples[samples++] = xe_q4_q8_dot(
                &layer->down_exps,
                (uint64_t)expert * XE_EMBD + column,
                XE_EXPERT_FFN, &row_q8);
        }
    }
    double expert_down_error = dense_rel(gpu_samples, cpu_samples, samples);
    float *moe_reference = xe_alloc(NULL, input_n * sizeof(*moe_reference),
                                    XE_MEM_HOST);
    for (int row = 0; row < rows; row++)
        for (int column = 0; column < XE_EMBD; column++) {
            float sum = 0.0f;
            for (int slot = 0; slot < XE_EXPERTS_USED; slot++) {
                int route = row * XE_EXPERTS_USED + slot;
                int packed = routes.route_packed[route];
                int expert = route_expert[route];
                sum += route_weight[route] * layer->down_exps_scale[expert]
                       * expert_down[(size_t)packed * XE_EMBD + column];
            }
            moe_reference[(size_t)row * XE_EMBD + column] = sum;
        }
    double moe_reduce_error = dense_rel(moe_output, moe_reference, input_n);
    float *layer_reference = xe_alloc(
        NULL, input_n * sizeof(*layer_reference), XE_MEM_HOST);
    float *combined = xe_alloc(NULL, XE_EMBD * sizeof(*combined), XE_MEM_HOST);
    for (int row = 0; row < rows; row++) {
        const float *dense_row = down + (size_t)row * XE_EMBD;
        const float *moe_row = moe_output + (size_t)row * XE_EMBD;
        float dense_scale = xe_rms_scale_scalar(dense_row, XE_EMBD);
        float moe_scale = xe_rms_scale_scalar(moe_row, XE_EMBD);
        for (int column = 0; column < XE_EMBD; column++)
            combined[column] = dense_row[column] * dense_scale
                               * layer->post_ffw_norm1[column]
                               + moe_row[column] * moe_scale
                                 * layer->post_ffw_norm2[column];
        float combine_scale = xe_rms_scale_scalar(combined, XE_EMBD);
        for (int column = 0; column < XE_EMBD; column++) {
            size_t index = (size_t)row * XE_EMBD + column;
            layer_reference[index] =
                (combined[column] * combine_scale
                 * layer->post_ffw_norm[column] + input[index])
                * layer->layer_out_scale[0];
        }
    }
    double finish_error = dense_rel(layer_output, layer_reference, input_n);
    int ok = dense_max_delta <= 1 && moe_max_delta <= 1
             && activation_max_delta <= 1 && router_error < 3e-6
             && router_logits_error < 2e-5 && route_mismatch == 0
             && route_weights_rel < 2e-5
             && routing_errors == 0 && tiles <= 256
             && expert_gate_error < 2e-5 && expert_activation_delta <= 1
             && expert_down_error < 2e-5 && moe_reduce_error < 2e-5
             && finish_error < 3e-6
             && gate_up_error < 2e-5 && down_error < 2e-5;
    printf("prefill-dense: M%d chain %.6f ms router %.6f ms routing %.6f ms/%d tiles moe %.6f ms finish %.6f ms dense-q8 %d delta %d moe-q8 %d delta %d router-input %.3e logits %.3e routes %d weights %.3e min-margin %.3e activation-q8 %d delta %d gate-up %.3e down %.3e expert-gate %.3e expert-act %d delta %d expert-down %.3e reduce %.3e finish-error %.3e %s\n",
           rows, elapsed_ms, router_ms, routing_ms, tiles, moe_ms, finish_ms,
           dense_mismatch, dense_max_delta, moe_mismatch,
           moe_max_delta, router_error, router_logits_error, route_mismatch,
           route_weights_rel, (double)minimum_margin, activation_mismatch,
           activation_max_delta, gate_up_error, down_error,
           expert_gate_error, expert_activation_mismatch,
           expert_activation_delta, expert_down_error, moe_reduce_error,
           finish_error,
           ok ? "PASS" : "FAIL");
    xe_free(NULL, combined, XE_MEM_HOST);
    xe_free(NULL, layer_reference, XE_MEM_HOST);
    xe_free(NULL, moe_reference, XE_MEM_HOST);
    xe_free(NULL, activated, XE_MEM_HOST);
    xe_free(NULL, activation_reference_sigma, XE_MEM_HOST);
    xe_free(NULL, activation_reference_d, XE_MEM_HOST);
    xe_free(NULL, activation_reference_qs, XE_MEM_HOST);
    xe_free(NULL, router_logits_reference, XE_MEM_HOST);
    xe_free(NULL, router_reference, XE_MEM_HOST);
    xe_free(NULL, semantic, XE_MEM_HOST);
    xe_free(NULL, reference_sigma, XE_MEM_HOST);
    xe_free(NULL, reference_d, XE_MEM_HOST);
    xe_free(NULL, reference_qs, XE_MEM_HOST);
    xe_free(e, down, XE_MEM_SHARED);
    dense_q8_free(e, &activation);
    xe_free(e, up, XE_MEM_SHARED);
    xe_free(e, gate, XE_MEM_SHARED);
    xe_free(e, layer_output, XE_MEM_SHARED);
    xe_free(e, moe_output, XE_MEM_SHARED);
    xe_free(e, expert_down, XE_MEM_SHARED);
    dense_q8_free(e, &expert_activation);
    xe_free(e, expert_gate_up, XE_MEM_SHARED);
    xe_free(e, route_weight, XE_MEM_SHARED);
    xe_free(e, route_expert, XE_MEM_SHARED);
    xe_free(e, routes.route_packed, XE_MEM_SHARED);
    xe_free(e, routes.packed_route, XE_MEM_SHARED);
    xe_free(e, routes.tile_m0, XE_MEM_SHARED);
    xe_free(e, routes.tile_expert, XE_MEM_SHARED);
    xe_free(e, routes.cursor, XE_MEM_SHARED);
    xe_free(e, routes.token_offset, XE_MEM_SHARED);
    xe_free(e, routes.expert_count, XE_MEM_SHARED);
    dense_q8_free(e, &packed_moe);
    xe_free(e, router_logits, XE_MEM_SHARED);
    xe_free(e, router_input, XE_MEM_SHARED);
    dense_q8_free(e, &moe_input);
    dense_q8_free(e, &dense_input);
    xe_free(e, row_scale, XE_MEM_SHARED);
    xe_free(e, input, XE_MEM_SHARED);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
