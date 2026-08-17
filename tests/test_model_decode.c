#include "../xenolith.c"

static double model_rel_rms(const float *a, const float *b, int n) {
    double error = 0.0;
    double reference = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        error += d * d;
        reference += (double)b[i] * b[i];
    }
    return sqrt(error / (reference + 1e-30));
}

static double model_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int model_shared_allocation(const xe_engine *e, const void *p) {
    ze_memory_allocation_properties_t properties = {
        .stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES
    };
    ze_device_handle_t device = NULL;
    return zeMemGetAllocProperties(e->gpu.context, p, &properties, &device) ==
               ZE_RESULT_SUCCESS &&
           properties.type == ZE_MEMORY_TYPE_SHARED && device == e->gpu.device;
}

static int model_gpu_memory(const xe_engine *e) {
    uint64_t total = 0;
    int ok = e->repack_part_count > 0 && e->repack_part_count <= XE_REPACK_PARTS;
    for (int i = 0; i < e->repack_part_count; i++) {
        ok = ok && e->repack_part_sizes[i] > 0;
        ok = ok && e->repack_part_sizes[i] <= XE_REPACK_PART_LIMIT;
        ok = ok && model_shared_allocation(e, e->repack_parts[i]);
        total += e->repack_part_sizes[i];
    }
    ok = ok && total == e->repack_slab_size;
    ok = ok && e->f32_slab_size > 0;
    ok = ok && e->f32_slab_size <= XE_REPACK_PART_LIMIT;
    ok = ok && model_shared_allocation(e, e->f32_slab);
    printf("model: GPU memory Q4 parts %d F32 %.2f MiB %s\n",
           e->repack_part_count, e->f32_slab_size / 1048576.0,
           ok ? "PASS" : "FAIL");
    return ok;
}

static int model_same_expert_order(const int *a, const int *b) {
    for (int i = 0; i < XE_EXPERTS_USED; i++) if (a[i] != b[i]) return 0;
    return 1;
}

typedef struct {
    int qs;
    int scales;
} model_q8_delta;

static model_q8_delta model_compare_q8(const xe_q8 *got, const float *x, int n) {
    int8_t qs[REF_Q8_MAX_IN];
    float ds[REF_Q8_MAX_IN / 32];
    ref_quantize_q8(x, n, qs, ds);
    model_q8_delta d = { 0, 0 };
    for (int i = 0; i < n; i++) d.qs += got->qs[i] != qs[i];
    for (int b = 0; b < n / 32; b++) d.scales += (float)got->d[b] != ds[b];
    return d;
}

static int model_contains(const int *experts, int expert) {
    for (int i = 0; i < XE_EXPERTS_USED; i++) if (experts[i] == expert) return 1;
    return 0;
}

static int model_same_expert_set(const int *a, const int *b) {
    for (int i = 0; i < XE_EXPERTS_USED; i++) if (!model_contains(b, a[i])) return 0;
    return 1;
}

static int model_router_attribution(const ref_state *ref, const xe_session *s,
                                    const int *experts, int layer) {
    int removed = -1;
    int added = -1;
    for (int i = XE_EXPERTS_USED - 1; i >= 0; i--) {
        if (removed < 0 && !model_contains(s->experts, experts[i])) removed = experts[i];
        if (added < 0 && !model_contains(experts, s->experts[i])) added = s->experts[i];
    }
    if (removed < 0 && added < 0) {
        double router_rel = model_rel_rms(s->router_logits, ref->router_probs, XE_EXPERTS);
        printf("model: L%02d router same set reordered rel %.3e ATTRIBUTED\n", layer, router_rel);
        return 1;
    }
    if (removed < 0 || added < 0) return 0;
    float ref_gap = ref->router_probs[removed] - ref->router_probs[added];
    float got_gap = s->router_logits[removed] - s->router_logits[added];
    float perturbation = fabsf(ref_gap - got_gap);
    double router_rel = model_rel_rms(s->router_logits, ref->router_probs, XE_EXPERTS);
    int ok = ref_gap <= perturbation + 1e-9f;
    printf("model: L%02d router removed %d added %d gap %.3e perturb %.3e rel %.3e %s\n",
           layer, removed, added, (double)ref_gap, (double)perturbation, router_rel,
           ok ? "ATTRIBUTED" : "FAIL");
    return ok;
}

static double model_forced_experts(xe_session *s, int layer, const int *experts,
                                   const float *reference) {
    const xe_layer *l = &s->engine->layers[layer];
    float hidden[XE_EMBD];
    int normal_experts[XE_EXPERTS_USED];
    float normal_weights[XE_EXPERTS_USED];
    memcpy(hidden, s->hidden, sizeof hidden);
    memcpy(normal_experts, s->experts, sizeof normal_experts);
    memcpy(normal_weights, s->expert_weights, sizeof normal_weights);

    float common_scale = xe_rms_scale_engine(s->engine, s->attn_out, XE_EMBD);
    xe_ffn_input_arg input = { l, common_scale };
    xe_dispatch(s->engine, s, xe_ffn_input_phase, &input, 1);
    xe_router(s, l, layer);
    float sum = 0.0f;
    for (int i = 0; i < XE_EXPERTS_USED; i++) sum += s->router_logits[experts[i]];
    if (sum < ref_moe_weight_sum_min) sum = ref_moe_weight_sum_min;
    for (int i = 0; i < XE_EXPERTS_USED; i++) {
        s->experts[i] = experts[i];
        s->expert_weights[i] = s->router_logits[experts[i]] / sum;
    }
    xe_moe_run(s, l, XE_MOE_GENERIC_BATCHED);
    xe_rmsnorm_engine(s->engine, s->moe_out, l->post_ffw_norm2, XE_EMBD, s->moe_out);
    for (int i = 0; i < XE_EMBD; i++) s->combined[i] = s->dense_out[i] + s->moe_out[i];
    xe_rmsnorm_engine(s->engine, s->combined, l->post_ffw_norm, XE_EMBD, s->combined);
    float out_scale = l->layer_out_scale[0];
    for (int i = 0; i < XE_EMBD; i++)
        s->hidden[i] = (s->combined[i] + s->attn_out[i]) * out_scale;
    double rel = model_rel_rms(s->hidden, reference, XE_EMBD);

    memcpy(s->hidden, hidden, sizeof hidden);
    memcpy(s->experts, normal_experts, sizeof normal_experts);
    memcpy(s->expert_weights, normal_weights, sizeof normal_weights);
    return rel;
}

static int model_dense_boundary(const xe_layer *l, const ref_state *v1, ref_state *scalar,
                                int layer) {
    memcpy(scalar->attn_out, v1->attn_out, XE_EMBD * sizeof(*scalar->attn_out));
    ref_dense_branch(l, scalar);

    int8_t scalar_qs[REF_Q8_MAX_IN];
    int8_t v1_qs[REF_Q8_MAX_IN];
    float scalar_ds[REF_Q8_MAX_IN / 32];
    float v1_ds[REF_Q8_MAX_IN / 32];
    ref_quantize_q8(scalar->dense_act, XE_DENSE_FFN, scalar_qs, scalar_ds);
    ref_quantize_q8(v1->dense_act, XE_DENSE_FFN, v1_qs, v1_ds);

    int q_differences = 0;
    int scale_differences = 0;
    for (int i = 0; i < XE_DENSE_FFN; i++)
        q_differences += scalar_qs[i] != v1_qs[i];
    for (int b = 0; b < XE_DENSE_FFN / 32; b++)
        scale_differences += scalar_ds[b] != v1_ds[b];

    float *scalar_pack_v1 = xe_alloc(NULL, XE_EMBD * sizeof(*scalar_pack_v1), XE_MEM_HOST);
    float *v1_pack_v1 = xe_alloc(NULL, XE_EMBD * sizeof(*v1_pack_v1), XE_MEM_HOST);
    ref_matvec_q4_q8_packed(&l->ffn_down, 0, XE_DENSE_FFN, XE_EMBD,
                            scalar_qs, scalar_ds, 1, scalar_pack_v1);
    ref_matvec_q4_q8_packed(&l->ffn_down, 0, XE_DENSE_FFN, XE_EMBD,
                            v1_qs, v1_ds, 1, v1_pack_v1);
    ref_rmsnorm(scalar_pack_v1, l->post_ffw_norm1, XE_EMBD, scalar_pack_v1);
    ref_rmsnorm(v1_pack_v1, l->post_ffw_norm1, XE_EMBD, v1_pack_v1);

    double accumulation = model_rel_rms(scalar_pack_v1, scalar->dense_out, XE_EMBD);
    double boundary = model_rel_rms(v1_pack_v1, scalar_pack_v1, XE_EMBD);
    double total = model_rel_rms(v1->dense_out, scalar->dense_out, XE_EMBD);
    double reconstruction = model_rel_rms(v1_pack_v1, v1->dense_out, XE_EMBD);
    int attributed = total <= 1e-4 || q_differences > 0 || scale_differences > 0;
    int ok = accumulation < 2e-6 && reconstruction < 1e-7 && attributed;
    if (total > 1e-5 || q_differences || scale_differences)
        printf("model: L%02d Q8-boundary qs %d scales %d accumulation %.2e boundary %.2e total %.2e %s\n",
               layer, q_differences, scale_differences, accumulation, boundary, total,
               ok ? "ATTRIBUTED" : "FAIL");

    xe_free(NULL, v1_pack_v1, XE_MEM_HOST);
    xe_free(NULL, scalar_pack_v1, XE_MEM_HOST);
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf> [token [scalar_rms [max_layers [isolated [ref_q8]]]]]\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    int32_t token = argc > 2 ? (int32_t)strtol(argv[2], NULL, 10) : 2;
    xe_engine *e = xe_engine_open(model);
    if (argc > 3) e->scalar_rms = (int)strtol(argv[3], NULL, 10);
    int max_layers = argc > 4 ? (int)strtol(argv[4], NULL, 10) : XE_LAYERS;
    int isolated = argc > 5 ? (int)strtol(argv[5], NULL, 10) : 1;
    int ref_q8 = argc > 6 ? (int)strtol(argv[6], NULL, 10) : 2;
    int gpu_memory_ok = model_gpu_memory(e);
    xe_session *s = xe_session_new(e);
    ref_state ref;
    memset(&ref, 0, sizeof ref);
    ref.q8 = ref_q8;
    ref_state_init(&ref, 1);
    ref_state scalar;
    memset(&scalar, 0, sizeof scalar);
    scalar.q8 = 1;
    ref_state_init(&scalar, 1);
    ref_embed(e, token, ref.hidden);
    xe_embed_decode(e, token, s->hidden);
    printf("model: embed %.3e\n", model_rel_rms(s->hidden, ref.hidden, XE_EMBD));

    xe_rope_prepare(s, 0);
    xe_workers_begin(e);
    double optimized_seconds = 0.0;
    int ok = gpu_memory_ok;
    int cumulative_attributed = 0;
    double previous_hidden = 0.0;
    for (int layer = 0; layer < max_layers; layer++) {
        if (isolated) memcpy(s->hidden, ref.hidden, XE_EMBD * sizeof(*s->hidden));
        ref_attention_half(e, layer, 0, &ref);
        double opt_start = model_now();
        xe_attention_half_decode(s, layer, 0);
        optimized_seconds += model_now() - opt_start;
        double q = model_rel_rms(s->q, ref.q, XE_Q_HEADS * (XE_IS_GLOBAL(layer) ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM));
        double k = model_rel_rms(s->k, ref.k, ref_kv_dim(layer));
        double heads = model_rel_rms(s->attn_heads, ref.attn_heads,
                                     XE_Q_HEADS * (XE_IS_GLOBAL(layer) ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM));
        double attention = model_rel_rms(s->attn_out, ref.attn_out, XE_EMBD);

        int experts[XE_EXPERTS_USED];
        float weights[XE_EXPERTS_USED];
        ref_dense_branch(&e->layers[layer], &ref);
        if (ref_q8 == 2 && !model_dense_boundary(&e->layers[layer], &ref, &scalar, layer)) ok = 0;
        ref_router(&e->layers[layer], &ref, experts, weights);
        ref_moe_branch(&e->layers[layer], &ref, experts, weights);
        for (int i = 0; i < XE_EMBD; i++) ref.combined[i] = ref.dense_out[i] + ref.moe_out[i];
        ref_rmsnorm(ref.combined, e->layers[layer].post_ffw_norm, XE_EMBD, ref.combined);
        float out_scale = e->layers[layer].layer_out_scale[0];
        for (int i = 0; i < XE_EMBD; i++)
            ref.hidden[i] = (ref.combined[i] + ref.attn_out[i]) * out_scale;

        opt_start = model_now();
        xe_ffn_half_decode(s, layer, XE_MOE_GENERIC_BATCHED, 1);
        optimized_seconds += model_now() - opt_start;
        int same_order = model_same_expert_order(s->experts, experts);
        int same_set = model_same_expert_set(s->experts, experts);
        const char *route = same_order ? "same" : (same_set ? "reorder" : "FLIP");
        double dense = model_rel_rms(s->dense_out, ref.dense_out, XE_EMBD);
        double moe = same_set ? model_rel_rms(s->moe_out, ref.moe_out, XE_EMBD) : NAN;
        double hidden = model_rel_rms(s->hidden, ref.hidden, XE_EMBD);
        model_q8_delta dense_in = model_compare_q8(&s->q8_dense_in, ref.dense_in, XE_EMBD);
        model_q8_delta dense_act = model_compare_q8(&s->q8_dense_act, ref.dense_act, XE_DENSE_FFN);
        model_q8_delta expert_act = { 0, 0 };
        if (same_set)
            expert_act = model_compare_q8(&s->q8_expert[XE_EXPERTS_USED - 1],
                                          ref.expert_act, XE_EXPERT_FFN);
        printf("model: L%02d q %.2e k %.2e heads %.2e attn %.2e route %s dense %.2e moe %.2e out %.2e\n",
               layer, q, k, heads, attention, route, dense, moe, hidden);
        if (isolated) {
            if (q > 2e-5 || k > 2e-5 || heads > 2e-5 || attention > 1e-4 || dense > 1e-4 ||
                (same_set && (moe > 1e-4 || hidden > 1e-4)) || dense_in.qs || dense_in.scales ||
                dense_act.qs || dense_act.scales || expert_act.qs || expert_act.scales) ok = 0;
        } else {
            int already_attributed = cumulative_attributed;
            if (!cumulative_attributed && (dense > 1e-4 || hidden > 1e-4)) {
                int boundary = dense_in.qs || dense_in.scales || dense_act.qs || dense_act.scales;
                printf("model: L%02d cumulative boundary dense-in %d/%d dense-act %d/%d %s\n",
                       layer, dense_in.qs, dense_in.scales, dense_act.qs, dense_act.scales,
                       boundary ? "ATTRIBUTED" : "FAIL");
                cumulative_attributed = boundary;
                if (!boundary) ok = 0;
            }
            if (!same_order) {
                int route_ok = model_router_attribution(&ref, s, experts, layer);
                double forced = model_forced_experts(s, layer, experts, ref.hidden);
                double bound = fmax(1e-4, attention * 4.0 + 1e-4);
                printf("model: L%02d forced-route out %.3e bound %.3e %s\n",
                       layer, forced, bound, forced <= bound ? "ATTRIBUTED" : "FAIL");
                if (!cumulative_attributed || !route_ok || forced > bound) ok = 0;
            }
            double smooth_bound = fmax(1e-4, previous_hidden * (same_order ? 4.0 : 8.0) + 1e-4);
            if (already_attributed && hidden > smooth_bound) ok = 0;
            previous_hidden = hidden;
        }
    }
    if (max_layers == XE_LAYERS) {
        double opt_start = model_now();
        xe_output_decode(s, XE_SOFTCAP_SECOND_LOOP);
        optimized_seconds += model_now() - opt_start;
    }
    xe_workers_end(e);
    if (max_layers == XE_LAYERS) {
        ref_output(e, &ref);
        double logits = model_rel_rms(s->logits, ref.logits, XE_VOCAB);
        printf("model: logits %.3e optimized %.3f ms %s\n", logits, optimized_seconds * 1000.0,
               ok ? "PASS" : "FAIL");
    } else {
        printf("model: %d layers optimized %.3f ms %s\n", max_layers,
               optimized_seconds * 1000.0, ok ? "PASS" : "FAIL");
    }

    ref_state_free(&scalar);
    ref_state_free(&ref);
    xe_session_free(s);
    xe_engine_close(e);
    return ok ? 0 : 1;
}
