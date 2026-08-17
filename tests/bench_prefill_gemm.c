#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <fcntl.h>
#include <level_zero/ze_api.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BENCH_ROUNDS_MAX 15
#define BENCH_DEVICE_ID 0xa7a0
#define BENCH_VENDOR_ID 0x8086

typedef struct {
    ze_context_handle_t context;
    ze_device_handle_t device;
    ze_module_handle_t module;
    ze_command_list_handle_t commands;
    ze_kernel_handle_t fused;
    ze_kernel_handle_t tn48;
    ze_kernel_handle_t tn64;
    ze_kernel_handle_t tm64_tn64;
    ze_kernel_handle_t tn64_alt1;
    ze_kernel_handle_t tn64_alt2;
    ze_kernel_handle_t tn64_alt3;
    ze_kernel_handle_t tn64_wg256;
    ze_kernel_handle_t tn128_wg256;
    ze_kernel_handle_t row;
    ze_kernel_handle_t raw;
    ze_kernel_handle_t correction;
    ze_kernel_handle_t grouped;
    ze_kernel_handle_t grouped64;
    ze_kernel_handle_t grouped_tn48;
    ze_kernel_handle_t grouped_tn64;
    ze_kernel_handle_t grouped_tn64_direct;
    ze_kernel_handle_t grouped_tn64_coalesced;
    ze_kernel_handle_t grouped_tn64_slmacc;
    ze_kernel_handle_t grouped_tn64_wg256;
    ze_kernel_handle_t grouped_tm64_tn64_wg256;
    ze_kernel_handle_t grouped_tm24_tn64;
    ze_kernel_handle_t grouped_tn64_signed;
    ze_kernel_handle_t grouped_tn64_kb128;
    ze_kernel_handle_t grouped_tm16_tn64;
    ze_kernel_handle_t grouped_tn128_wg256;
    ze_kernel_handle_t grouped_expert;
    ze_kernel_handle_t grouped_gather;
    ze_kernel_handle_t tail;
    ze_kernel_handle_t route_reset;
    ze_kernel_handle_t route_count;
    ze_kernel_handle_t route_prefix;
    ze_kernel_handle_t route_scatter;
    ze_kernel_handle_t route_pack;
    ze_kernel_handle_t route_reduce;
    ze_kernel_handle_t route_reduce_scaled;
    ze_kernel_handle_t geglu_q8;
    ze_kernel_handle_t geglu_q8_pair;
    ze_kernel_handle_t router_gemm;
    ze_kernel_handle_t router_gemm_tiled;
    ze_kernel_handle_t router_top8;
    ze_kernel_handle_t ffn_input_q8;
    ze_kernel_handle_t rms_scale;
    ze_kernel_handle_t norm_q8;
    ze_kernel_handle_t ffn_input_q8_rowscale;
    ze_kernel_handle_t qkv_post;
    ze_kernel_handle_t heads_q8;
    ze_kernel_handle_t rms_residual;
    ze_kernel_handle_t ffn_finish;
    ze_kernel_handle_t attn_qk;
    ze_kernel_handle_t attn_softmax;
    ze_kernel_handle_t attn_pv;
    ze_kernel_handle_t attn_pv4;
    ze_kernel_handle_t attn_online;
    ze_kernel_handle_t attn_online_b4;
    ze_kernel_handle_t attn_online_b8;
    ze_kernel_handle_t attn_online_b8_ring;
    ze_kernel_handle_t swa_stage;
    ze_kernel_handle_t swa_commit;
    ze_kernel_handle_t attn_long_init;
    ze_kernel_handle_t attn_partial_b8;
    ze_kernel_handle_t attn_partial_merge;
    ze_kernel_handle_t attn_gqa8;
    uint64_t timer_resolution;
    uint32_t timestamp_bits;
} bench_gpu;

typedef struct {
    atomic_int ready;
    atomic_int stop;
    pthread_t thread;
    long samples;
    long actual_sum;
    long actual_min;
    long actual_max;
    long requested_sum;
    long pl1;
    long pl2;
    long thermal;
} bench_telemetry;

typedef struct {
    double seconds;
    double actual_mhz;
    double requested_mhz;
    long actual_min;
    long actual_max;
    long pl1;
    long pl2;
    long thermal;
} bench_sample;

typedef struct {
    const char *name;
    int m;
    int n;
    int blocks;
} bench_shape;

typedef struct {
    uint8_t *weight_x8;
    uint8_t *weight_row;
    _Float16 *weight_scale_x8;
    _Float16 *weight_scale_row;
    int8_t *activation;
    _Float16 *activation_scale;
    int16_t *activation_sigma;
    float *output;
} bench_data;

enum {
    BENCH_FUSED,
    BENCH_ROW,
    BENCH_SPLIT,
    BENCH_TN48,
    BENCH_TN64,
    BENCH_TN64_WG256,
    BENCH_TN128_WG256,
    BENCH_VARIANTS
};

#define BENCH_TM64_TN64 BENCH_VARIANTS

#define BENCH_MOE_EXPERTS 128
#define BENCH_MOE_ROWS 4096
#define BENCH_MOE_TOKENS 512
#ifndef BENCH_MOE_N
#define BENCH_MOE_N 1408
#endif
#ifndef BENCH_MOE_BLOCKS
#define BENCH_MOE_BLOCKS 88
#endif
#define BENCH_MOE_PADDED_TILES 256
#define BENCH_MOE_PADDED16_TILES 384
#define BENCH_MOE_PADDED24_TILES 320
#define BENCH_MOE_FULL_TILES 128
#define BENCH_MOE_TAIL_TILES 128

typedef struct {
    uint8_t *weight;
    _Float16 *weight_scale;
    int8_t *activation;
    _Float16 *activation_scale;
    int16_t *activation_sigma;
    int8_t *source_activation;
    _Float16 *source_scale;
    int16_t *source_sigma;
    float *output;
    int *expert_count;
    int *token_offset;
    int *padded_expert;
    int *padded_m0;
    int *padded64_expert;
    int *padded64_m0;
    int *padded16_expert;
    int *padded16_m0;
    int *padded24_expert;
    int *padded24_m0;
    int *full_expert;
    int *full_m0;
    int *tail_expert;
    int *tail_m0;
    ze_group_count_t *padded_launch;
    int *route_expert;
    int *route_token;
    int *packed_route;
    int *route_packed;
    int *route_cursor;
    float *route_weight;
    float *reduced;
    int padded_valid;
    int padded64_valid;
    int padded16_valid;
    int padded24_valid;
    int full_valid;
    int tail_valid;
} bench_moe_data;

enum {
    BENCH_MOE_PADDED,
    BENCH_MOE_PADDED_FIXED,
    BENCH_MOE_PADDED_INDIRECT,
    BENCH_MOE_PADDED64,
    BENCH_MOE_TN48,
    BENCH_MOE_TN64,
    BENCH_MOE_TN64_WG256,
    BENCH_MOE_TM64_TN64_WG256,
    BENCH_MOE_TM24_TN64,
    BENCH_MOE_TN64_SIGNED,
    BENCH_MOE_EXPERT,
    BENCH_MOE_HYBRID,
    BENCH_MOE_VARIANTS
};

#define BENCH_MOE_TM16_TN64 BENCH_MOE_VARIANTS
#define BENCH_MOE_TN128_WG256 (BENCH_MOE_VARIANTS + 1)
#define BENCH_MOE_KB128 (BENCH_MOE_VARIANTS + 2)
#define BENCH_MOE_TN64_SLMACC (BENCH_MOE_VARIANTS + 3)
#define BENCH_MOE_TN64_DIRECT (BENCH_MOE_VARIANTS + 4)
#define BENCH_MOE_TN64_COALESCED (BENCH_MOE_VARIANTS + 5)

enum {
    BENCH_ROUTE_METADATA,
    BENCH_ROUTE_PACK,
    BENCH_ROUTE_BASELINE,
    BENCH_ROUTE_PACK_GEMM,
    BENCH_ROUTE_GATHER_GEMM,
    BENCH_ROUTE_VARIANTS
};

typedef struct {
    const char *name;
    int m;
    int n;
    int dimension;
    int heads;
    int kv_heads;
    int query_offset;
    int window;
} bench_attention_shape;

typedef struct {
    float *q;
    _Float16 *k;
    _Float16 *v;
    _Float16 *ring_k;
    _Float16 *ring_v;
    _Float16 *physical_k;
    _Float16 *physical_v;
    _Float16 *batch_k;
    _Float16 *batch_v;
    float *scores;
    float *materialized;
    float *online;
    float *online_b4;
    float *online_b8;
    float *online_b8_ring;
    float *gqa8;
} bench_attention_data;

typedef struct {
    uint8_t *weight;
    _Float16 *weight_scale;
    float *output;
    int n;
    int blocks;
    int experts;
} bench_layer_matrix;

typedef struct {
    int rows;
    int dimension;
    int kv_heads;
    int has_v;
    float *hidden;
    float *ones;
    float *row_scale;
    int8_t *attn_q8;
    _Float16 *attn_d;
    int16_t *attn_s;
    bench_layer_matrix q;
    bench_layer_matrix k;
    bench_layer_matrix v_projection;
    bench_layer_matrix o;
    bench_layer_matrix dense_gate;
    bench_layer_matrix dense_up;
    bench_layer_matrix dense_down;
    bench_layer_matrix expert_gate_up;
    bench_layer_matrix expert_down;
    float *q_weight;
    float *k_weight;
    float *rope_cos;
    float *rope_sin;
    float *q_heads;
    _Float16 *k_heads;
    _Float16 *v_heads;
    float *attn_heads;
    int8_t *heads_q8;
    _Float16 *heads_d;
    int16_t *heads_s;
    float *attn_out;
    int8_t *dense_q8;
    _Float16 *dense_d;
    int16_t *dense_s;
    int8_t *moe_q8;
    _Float16 *moe_d;
    int16_t *moe_s;
    float *router_input;
    float *router_weight;
    float *router_logits;
    int *route_expert;
    float *route_weight;
    int *route_token;
    int *expert_count;
    int *token_offset;
    int *cursor;
    int *tile_expert;
    int *tile_m0;
    int *packed_route;
    int *route_packed;
    int8_t *packed_q8;
    _Float16 *packed_d;
    int16_t *packed_s;
    int8_t *dense_act_q8;
    _Float16 *dense_act_d;
    int16_t *dense_act_s;
    int8_t *expert_act_q8;
    _Float16 *expert_act_d;
    int16_t *expert_act_s;
    float *moe_reduced;
    float *expert_scale;
    float *finish_output;
    float *layer_scale;
    ze_kernel_handle_t ffn_rms;
    ze_kernel_handle_t kernels[9];
    void *allocations[96];
    int allocation_count;
} bench_layer_data;

enum {
    BENCH_ATTN_MATERIALIZED,
    BENCH_ATTN_ONLINE,
    BENCH_ATTN_ONLINE_B4,
    BENCH_ATTN_ONLINE_B8,
    BENCH_ATTN_ONLINE_B8_RING,
    BENCH_ATTN_GQA8,
    BENCH_ATTN_VARIANTS
};

static _Noreturn void bench_fatal(const char *format, ...) {
    va_list args;
    fprintf(stderr, "prefill-gemm: ");
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(1);
}

static void bench_ze_check(const char *operation, ze_result_t result) {
    if (result != ZE_RESULT_SUCCESS)
        bench_fatal("%s failed: 0x%x", operation, result);
}

static double bench_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        bench_fatal("clock_gettime: %s", strerror(errno));
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void *bench_alloc(size_t bytes) {
    if (bytes > SIZE_MAX - 63) bench_fatal("allocation size overflow");
    size_t rounded = (bytes + 63) / 64 * 64;
    void *pointer = aligned_alloc(64, rounded);
    if (!pointer) bench_fatal("out of memory allocating %zu bytes", rounded);
    return pointer;
}

static void *bench_read_file(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file) bench_fatal("%s: %s", path, strerror(errno));
    if (fseek(file, 0, SEEK_END) != 0) bench_fatal("%s: seek failed", path);
    long length = ftell(file);
    if (length <= 0) bench_fatal("%s: invalid size", path);
    if (fseek(file, 0, SEEK_SET) != 0) bench_fatal("%s: seek failed", path);
    void *data = bench_alloc((size_t)length);
    if (fread(data, 1, (size_t)length, file) != (size_t)length)
        bench_fatal("%s: short read", path);
    fclose(file);
    *size = (size_t)length;
    return data;
}

static long bench_read_fd(int fd) {
    char buffer[64];
    ssize_t length = pread(fd, buffer, sizeof buffer - 1, 0);
    if (length <= 0) return -1;
    buffer[length] = 0;
    char *end;
    long value = strtol(buffer, &end, 10);
    return end == buffer ? -1 : value;
}

static int bench_open_telemetry(const char *name) {
    const char *base = "/sys/class/drm/card1/gt/gt0/";
    char path[256];
    int length = snprintf(path, sizeof path, "%s%s", base, name);
    if (length < 0 || (size_t)length >= sizeof path)
        bench_fatal("telemetry path overflow");
    int fd = open(path, O_RDONLY);
    if (fd < 0) bench_fatal("%s: %s", path, strerror(errno));
    return fd;
}

static void *bench_telemetry_main(void *opaque) {
    bench_telemetry *telemetry = opaque;
    int requested_fd = bench_open_telemetry("punit_req_freq_mhz");
    int actual_fd = bench_open_telemetry("rps_act_freq_mhz");
    int pl1_fd = bench_open_telemetry("throttle_reason_pl1");
    int pl2_fd = bench_open_telemetry("throttle_reason_pl2");
    int thermal_fd = bench_open_telemetry("throttle_reason_thermal");
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(13, &cpus);
    if (pthread_setaffinity_np(pthread_self(), sizeof cpus, &cpus) != 0)
        bench_fatal("cannot pin telemetry thread to CPU 13");
    atomic_store_explicit(&telemetry->ready, 1, memory_order_release);

    while (!atomic_load_explicit(&telemetry->stop, memory_order_relaxed)) {
        long requested = bench_read_fd(requested_fd);
        long actual = bench_read_fd(actual_fd);
        long pl1 = bench_read_fd(pl1_fd);
        long pl2 = bench_read_fd(pl2_fd);
        long thermal = bench_read_fd(thermal_fd);
        if (requested >= 0 && actual >= 0) {
            telemetry->requested_sum += requested;
            telemetry->actual_sum += actual;
            if (telemetry->actual_min < 0 || actual < telemetry->actual_min)
                telemetry->actual_min = actual;
            if (actual > telemetry->actual_max) telemetry->actual_max = actual;
            telemetry->samples++;
        }
        if (pl1 > 0) telemetry->pl1++;
        if (pl2 > 0) telemetry->pl2++;
        if (thermal > 0) telemetry->thermal++;
        struct timespec delay = { 0, 20000000 };
        nanosleep(&delay, NULL);
    }

    close(thermal_fd);
    close(pl2_fd);
    close(pl1_fd);
    close(actual_fd);
    close(requested_fd);
    return NULL;
}

static void bench_telemetry_start(bench_telemetry *telemetry) {
    memset(telemetry, 0, sizeof *telemetry);
    telemetry->actual_min = -1;
    atomic_init(&telemetry->ready, 0);
    atomic_init(&telemetry->stop, 0);
    if (pthread_create(&telemetry->thread, NULL, bench_telemetry_main, telemetry) != 0)
        bench_fatal("cannot create telemetry thread");
    while (!atomic_load_explicit(&telemetry->ready, memory_order_acquire))
        sched_yield();
}

static void bench_telemetry_stop(bench_telemetry *telemetry) {
    atomic_store_explicit(&telemetry->stop, 1, memory_order_relaxed);
    if (pthread_join(telemetry->thread, NULL) != 0)
        bench_fatal("cannot join telemetry thread");
    if (!telemetry->samples) bench_fatal("GPU frequency telemetry has no samples");
}

static void *bench_gpu_alloc(bench_gpu *gpu, size_t bytes) {
    ze_device_mem_alloc_desc_t device_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC
    };
    ze_host_mem_alloc_desc_t host_desc = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC
    };
    void *pointer = NULL;
    bench_ze_check("zeMemAllocShared",
                   zeMemAllocShared(gpu->context, &device_desc, &host_desc,
                                    bytes, 64, gpu->device, &pointer));
    return pointer;
}

static ze_kernel_handle_t bench_kernel_create_size(bench_gpu *gpu,
                                                    const char *name,
                                                    uint32_t group_size) {
    ze_kernel_desc_t desc = {
        .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
        .pKernelName = name
    };
    ze_kernel_handle_t kernel = NULL;
    bench_ze_check("zeKernelCreate", zeKernelCreate(gpu->module, &desc, &kernel));
    bench_ze_check("zeKernelSetGroupSize",
                   zeKernelSetGroupSize(kernel, group_size, 1, 1));
    ze_kernel_properties_t properties = {
        .stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES
    };
    bench_ze_check("zeKernelGetProperties", zeKernelGetProperties(kernel, &properties));
    printf("prefill-gemm: kernel %s local %u private %u spill %u\n",
           name, properties.localMemSize, properties.privateMemSize,
           properties.spillMemSize);
    return kernel;
}

static ze_kernel_handle_t bench_kernel_create(bench_gpu *gpu, const char *name) {
    return bench_kernel_create_size(gpu, name, 128);
}

static void bench_gpu_init(bench_gpu *gpu, const char *spv_path) {
    memset(gpu, 0, sizeof *gpu);
    bench_ze_check("zeInit", zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t driver_count = 0;
    bench_ze_check("zeDriverGet count", zeDriverGet(&driver_count, NULL));
    if (!driver_count) bench_fatal("no Level Zero driver");
    ze_driver_handle_t *drivers = bench_alloc(driver_count * sizeof *drivers);
    bench_ze_check("zeDriverGet", zeDriverGet(&driver_count, drivers));
    ze_driver_handle_t driver = drivers[0];
    free(drivers);

    uint32_t device_count = 0;
    bench_ze_check("zeDeviceGet count", zeDeviceGet(driver, &device_count, NULL));
    if (!device_count) bench_fatal("no Level Zero GPU");
    ze_device_handle_t *devices = bench_alloc(device_count * sizeof *devices);
    bench_ze_check("zeDeviceGet", zeDeviceGet(driver, &device_count, devices));
    gpu->device = devices[0];
    free(devices);

    ze_device_properties_t properties = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2
    };
    bench_ze_check("zeDeviceGetProperties",
                   zeDeviceGetProperties(gpu->device, &properties));
    if (properties.vendorId != BENCH_VENDOR_ID || properties.deviceId != BENCH_DEVICE_ID)
        bench_fatal("expected %04x:%04x, found %04x:%04x", BENCH_VENDOR_ID,
                    BENCH_DEVICE_ID, properties.vendorId, properties.deviceId);
    printf("prefill-gemm: device %s %04x:%04x max-clock %u MHz\n",
           properties.name, properties.vendorId, properties.deviceId,
           properties.coreClockRate);
    gpu->timer_resolution = properties.timerResolution;
    gpu->timestamp_bits = properties.kernelTimestampValidBits;

    ze_context_desc_t context_desc = {
        .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC
    };
    bench_ze_check("zeContextCreate",
                   zeContextCreate(driver, &context_desc, &gpu->context));
    ze_command_queue_desc_t queue_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .ordinal = 0,
        .index = 0,
        .flags = ZE_COMMAND_QUEUE_FLAG_IN_ORDER,
        .mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    };
    bench_ze_check("zeCommandListCreateImmediate",
                   zeCommandListCreateImmediate(gpu->context, gpu->device,
                                                &queue_desc, &gpu->commands));

    size_t spv_size = 0;
    void *spv = bench_read_file(spv_path, &spv_size);
    size_t path_length = strlen(spv_path);
    int native = path_length >= 4 && !strcmp(spv_path + path_length - 4, ".bin");
    const char *build_flags = getenv("BENCH_BUILD_FLAGS");
    if (!build_flags) build_flags = "";
    ze_module_desc_t module_desc = {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .format = native ? ZE_MODULE_FORMAT_NATIVE : ZE_MODULE_FORMAT_IL_SPIRV,
        .inputSize = spv_size,
        .pInputModule = spv,
        .pBuildFlags = build_flags
    };
    ze_module_build_log_handle_t log = NULL;
    double module_start = bench_now();
    ze_result_t result = zeModuleCreate(gpu->context, gpu->device, &module_desc,
                                        &gpu->module, &log);
    double module_seconds = bench_now() - module_start;
    if (result != ZE_RESULT_SUCCESS && log) {
        size_t length = 0;
        zeModuleBuildLogGetString(log, &length, NULL);
        char *message = bench_alloc(length + 1);
        zeModuleBuildLogGetString(log, &length, message);
        message[length] = 0;
        fprintf(stderr, "%s\n", message);
        free(message);
    }
    if (log) zeModuleBuildLogDestroy(log);
    free(spv);
    bench_ze_check("zeModuleCreate", result);
    printf("prefill-gemm: module create %.6f s format %s bytes %zu flags '%s'\n",
           module_seconds, native ? "native" : "SPIR-V", spv_size,
           build_flags);

    gpu->fused = bench_kernel_create(gpu, "prefill_q4q8_x8_fused");
    gpu->tn48 = bench_kernel_create(gpu, "prefill_q4q8_x8_tn48");
    gpu->tn64 = bench_kernel_create(gpu, "prefill_q4q8_x8_tn64");
    gpu->tm64_tn64 =
        bench_kernel_create(gpu, "prefill_q4q8_x8_tm64_tn64");
    gpu->tn64_alt1 = bench_kernel_create(gpu, "prefill_q4q8_x8_tn64");
    gpu->tn64_alt2 = bench_kernel_create(gpu, "prefill_q4q8_x8_tn64");
    gpu->tn64_alt3 = bench_kernel_create(gpu, "prefill_q4q8_x8_tn64");
    gpu->tn64_wg256 = bench_kernel_create_size(
        gpu, "prefill_q4q8_x8_tn64_wg256", 256);
    gpu->tn128_wg256 = bench_kernel_create_size(
        gpu, "prefill_q4q8_x8_tn128_wg256", 256);
    gpu->row = bench_kernel_create(gpu, "prefill_q4q8_row_fused");
    gpu->raw = bench_kernel_create(gpu, "prefill_q4q8_x8_raw");
    gpu->correction = bench_kernel_create(gpu, "prefill_q4q8_correction");
    gpu->grouped = bench_kernel_create(gpu, "prefill_q4q8_grouped32");
    gpu->grouped64 = bench_kernel_create(gpu, "prefill_q4q8_grouped64");
    gpu->grouped_tn48 =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_tn48");
    gpu->grouped_tn64 =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_tn64");
    gpu->grouped_tn64_direct =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_tn64_direct");
    gpu->grouped_tn64_coalesced =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_tn64_coalesced");
    gpu->grouped_tn64_slmacc =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_tn64_slmacc");
    gpu->grouped_tn64_wg256 = bench_kernel_create_size(
        gpu, "prefill_q4q8_grouped_tn64_wg256", 256);
    gpu->grouped_tm64_tn64_wg256 = bench_kernel_create_size(
        gpu, "prefill_q4q8_grouped_tm64_tn64_wg256", 256);
    gpu->grouped_tm24_tn64 =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_tm24_tn64");
    gpu->grouped_tn64_signed =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_tn64_signed");
    gpu->grouped_tn64_kb128 =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_tn64_kb128");
    gpu->grouped_tm16_tn64 =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_tm16_tn64");
    gpu->grouped_tn128_wg256 = bench_kernel_create_size(
        gpu, "prefill_q4q8_grouped_tn128_wg256", 256);
    gpu->grouped_expert =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_expert");
    gpu->grouped_gather =
        bench_kernel_create(gpu, "prefill_q4q8_grouped_gather");
    gpu->tail = bench_kernel_create(gpu, "prefill_q4q8_tail32");
    gpu->route_reset = bench_kernel_create(gpu, "prefill_route_reset");
    gpu->route_count = bench_kernel_create(gpu, "prefill_route_count");
    gpu->route_prefix = bench_kernel_create(gpu, "prefill_route_prefix");
    gpu->route_scatter = bench_kernel_create(gpu, "prefill_route_scatter");
    gpu->route_pack = bench_kernel_create(gpu, "prefill_route_pack");
    gpu->route_reduce = bench_kernel_create(gpu, "prefill_route_reduce");
    gpu->route_reduce_scaled =
        bench_kernel_create(gpu, "prefill_route_reduce_scaled");
    gpu->geglu_q8 = bench_kernel_create(gpu, "prefill_geglu_q8");
    gpu->geglu_q8_pair = bench_kernel_create(gpu, "prefill_geglu_q8_pair");
    gpu->router_gemm = bench_kernel_create(gpu, "prefill_router_gemm");
    gpu->router_gemm_tiled =
        bench_kernel_create(gpu, "prefill_router_gemm_tiled");
    gpu->router_top8 = bench_kernel_create(gpu, "prefill_router_top8");
    gpu->ffn_input_q8 = bench_kernel_create(gpu, "prefill_ffn_input_q8");
    gpu->rms_scale = bench_kernel_create(gpu, "prefill_rms_scale");
    gpu->norm_q8 = bench_kernel_create(gpu, "prefill_norm_q8");
    gpu->ffn_input_q8_rowscale =
        bench_kernel_create(gpu, "prefill_ffn_input_q8_rowscale");
    gpu->qkv_post = bench_kernel_create(gpu, "prefill_qkv_post");
    gpu->heads_q8 = bench_kernel_create(gpu, "prefill_heads_q8");
    gpu->rms_residual = bench_kernel_create(gpu, "prefill_rms_residual");
    gpu->ffn_finish = bench_kernel_create(gpu, "prefill_ffn_finish");
    gpu->attn_qk = bench_kernel_create(gpu, "prefill_attn_qk");
    gpu->attn_softmax = bench_kernel_create(gpu, "prefill_attn_softmax");
    gpu->attn_pv = bench_kernel_create(gpu, "prefill_attn_pv");
    gpu->attn_pv4 = bench_kernel_create(gpu, "prefill_attn_pv4");
    gpu->attn_online = bench_kernel_create(gpu, "prefill_attn_online");
    gpu->attn_online_b4 = bench_kernel_create(gpu, "prefill_attn_online_b4");
    gpu->attn_online_b8 = bench_kernel_create(gpu, "prefill_attn_online_b8");
    gpu->attn_online_b8_ring =
        bench_kernel_create(gpu, "prefill_attn_online_b8_stage");
    gpu->swa_stage = bench_kernel_create_size(gpu, "prefill_swa_stage", 256);
    gpu->swa_commit = bench_kernel_create_size(gpu, "prefill_swa_commit", 256);
    gpu->attn_long_init =
        bench_kernel_create_size(gpu, "prefill_attn_long_init", 256);
    gpu->attn_partial_b8 =
        bench_kernel_create_size(gpu, "prefill_attn_partial_b8", 128);
    gpu->attn_partial_merge =
        bench_kernel_create_size(gpu, "prefill_attn_partial_merge", 128);
    gpu->attn_gqa8 = bench_kernel_create(gpu, "prefill_attn_gqa8");
}

static void bench_pointer_arg(ze_kernel_handle_t kernel, uint32_t index,
                              const void *pointer) {
    bench_ze_check("zeKernelSetArgumentValue pointer",
                   zeKernelSetArgumentValue(kernel, index, sizeof pointer, &pointer));
}

static void bench_int_arg(ze_kernel_handle_t kernel, uint32_t index, int value) {
    bench_ze_check("zeKernelSetArgumentValue int",
                   zeKernelSetArgumentValue(kernel, index, sizeof value, &value));
}

static void bench_float_arg(ze_kernel_handle_t kernel, uint32_t index,
                            float value) {
    bench_ze_check("zeKernelSetArgumentValue float",
                   zeKernelSetArgumentValue(kernel, index, sizeof value, &value));
}

static void bench_set_gemm(ze_kernel_handle_t kernel, const void *weight,
                           const void *weight_scale, const bench_data *data,
                           const bench_shape *shape) {
    bench_pointer_arg(kernel, 0, weight);
    bench_pointer_arg(kernel, 1, weight_scale);
    bench_pointer_arg(kernel, 2, data->activation);
    bench_pointer_arg(kernel, 3, data->activation_scale);
    bench_pointer_arg(kernel, 4, data->activation_sigma);
    bench_pointer_arg(kernel, 5, data->output);
    bench_int_arg(kernel, 6, shape->m);
    bench_int_arg(kernel, 7, shape->n);
    bench_int_arg(kernel, 8, shape->blocks);
}

static void bench_set_correction(ze_kernel_handle_t kernel, const bench_data *data,
                                 const bench_shape *shape) {
    bench_pointer_arg(kernel, 0, data->weight_scale_x8);
    bench_pointer_arg(kernel, 1, data->activation_scale);
    bench_pointer_arg(kernel, 2, data->activation_sigma);
    bench_pointer_arg(kernel, 3, data->output);
    bench_int_arg(kernel, 4, shape->m);
    bench_int_arg(kernel, 5, shape->n);
    bench_int_arg(kernel, 6, shape->blocks);
}

static uint32_t bench_random(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static void bench_data_init(bench_gpu *gpu, bench_data *data,
                            const bench_shape *shape) {
    size_t weight_bytes = (size_t)shape->n * shape->blocks * 16;
    size_t weight_scale_bytes = (size_t)shape->n * shape->blocks * sizeof(_Float16);
    size_t activation_bytes = (size_t)shape->m * shape->blocks * 32;
    size_t activation_scale_bytes =
        (size_t)shape->m * shape->blocks * sizeof(_Float16);
    size_t output_bytes = (size_t)shape->m * shape->n * sizeof(float);
    data->weight_x8 = bench_gpu_alloc(gpu, weight_bytes);
    data->weight_row = bench_gpu_alloc(gpu, weight_bytes);
    data->weight_scale_x8 = bench_gpu_alloc(gpu, weight_scale_bytes);
    data->weight_scale_row = bench_gpu_alloc(gpu, weight_scale_bytes);
    data->activation = bench_gpu_alloc(gpu, activation_bytes);
    data->activation_scale = bench_gpu_alloc(gpu, activation_scale_bytes);
    data->activation_sigma = bench_gpu_alloc(gpu,
                                             (size_t)shape->m * shape->blocks
                                             * sizeof(int16_t));
    data->output = bench_gpu_alloc(gpu, output_bytes);
    uint32_t random = UINT32_C(0x6a09e667) ^ (uint32_t)shape->m
                      ^ ((uint32_t)shape->n << 8) ^ ((uint32_t)shape->blocks << 20);

    for (int group = 0; group < shape->n / 8; group++) {
        for (int block = 0; block < shape->blocks; block++) {
            for (int row = 0; row < 8; row++) {
                size_t x8_scale = ((size_t)group * shape->blocks + block) * 8 + row;
                float scale = (float)(bench_random(&random) % 200 + 1) / 1000.0f;
                data->weight_scale_x8[x8_scale] = (_Float16)scale;
                data->weight_scale_row[((size_t)group * 8 + row) * shape->blocks
                                       + block] = (_Float16)scale;
                for (int chunk = 0; chunk < 4; chunk++) {
                    for (int byte = 0; byte < 4; byte++) {
                        uint8_t value = (uint8_t)bench_random(&random);
                        size_t x8 = ((size_t)group * shape->blocks + block) * 128
                                    + chunk * 32 + row * 4 + byte;
                        size_t native = (((size_t)group * 8 + row) * shape->blocks
                                         + block) * 16 + chunk * 4 + byte;
                        data->weight_x8[x8] = value;
                        data->weight_row[native] = value;
                    }
                }
            }
        }
    }
    for (int row = 0; row < shape->m; row++) {
        for (int block = 0; block < shape->blocks; block++) {
            size_t base = ((size_t)row * shape->blocks + block) * 32;
            int sigma = 0;
            for (int k = 0; k < 32; k++) {
                int value = (int)(bench_random(&random) % 255) - 127;
                data->activation[base + k] = (int8_t)value;
                sigma += value;
            }
            size_t index = (size_t)row * shape->blocks + block;
            data->activation_scale[index] =
                (_Float16)((float)(bench_random(&random) % 200 + 1) / 1000.0f);
            data->activation_sigma[index] = (int16_t)sigma;
        }
    }

    bench_set_gemm(gpu->fused, data->weight_x8, data->weight_scale_x8,
                   data, shape);
    bench_set_gemm(gpu->tn48, data->weight_x8, data->weight_scale_x8,
                   data, shape);
    bench_set_gemm(gpu->tn64, data->weight_x8, data->weight_scale_x8,
                   data, shape);
    bench_set_gemm(gpu->tm64_tn64, data->weight_x8,
                   data->weight_scale_x8, data, shape);
    bench_set_gemm(gpu->tn64_wg256, data->weight_x8,
                   data->weight_scale_x8, data, shape);
    bench_set_gemm(gpu->tn128_wg256, data->weight_x8,
                   data->weight_scale_x8, data, shape);
    bench_set_gemm(gpu->row, data->weight_row, data->weight_scale_row,
                   data, shape);
    bench_set_gemm(gpu->raw, data->weight_x8, data->weight_scale_x8,
                   data, shape);
    bench_set_correction(gpu->correction, data, shape);
}

static void bench_data_destroy(bench_gpu *gpu, bench_data *data) {
    bench_ze_check("zeMemFree output", zeMemFree(gpu->context, data->output));
    bench_ze_check("zeMemFree activation sigma",
                   zeMemFree(gpu->context, data->activation_sigma));
    bench_ze_check("zeMemFree activation scale",
                   zeMemFree(gpu->context, data->activation_scale));
    bench_ze_check("zeMemFree activation",
                   zeMemFree(gpu->context, data->activation));
    bench_ze_check("zeMemFree row scale",
                   zeMemFree(gpu->context, data->weight_scale_row));
    bench_ze_check("zeMemFree x8 scale",
                   zeMemFree(gpu->context, data->weight_scale_x8));
    bench_ze_check("zeMemFree row weight",
                   zeMemFree(gpu->context, data->weight_row));
    bench_ze_check("zeMemFree x8 weight",
                   zeMemFree(gpu->context, data->weight_x8));
}

static double bench_launch(bench_gpu *gpu, int variant,
                           const bench_shape *shape, int repetitions) {
    ze_group_count_t groups = {
        (uint32_t)(variant == BENCH_TM64_TN64
                   ? (shape->m + 63) / 64 : (shape->m + 31) / 32),
        (uint32_t)(variant == BENCH_TN48 ? (shape->n + 47) / 48
                   : variant == BENCH_TN128_WG256
                     ? (shape->n + 127) / 128
                   : shape->n
                     / (variant == BENCH_TN64
                        || variant == BENCH_TM64_TN64
                        || variant == BENCH_TN64_WG256 ? 64 : 32)),
        1
    };
    ze_kernel_handle_t first = variant == BENCH_ROW ? gpu->row
                              : variant == BENCH_SPLIT ? gpu->raw
                              : variant == BENCH_TN48 ? gpu->tn48
                              : variant == BENCH_TN64 ? gpu->tn64
                              : variant == BENCH_TM64_TN64 ? gpu->tm64_tn64
                              : variant == BENCH_TN64_WG256
                                ? gpu->tn64_wg256
                              : variant == BENCH_TN128_WG256
                                ? gpu->tn128_wg256 : gpu->fused;
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        bench_ze_check("zeCommandListAppendLaunchKernel GEMM",
                       zeCommandListAppendLaunchKernel(gpu->commands, first,
                                                       &groups, NULL, 0, NULL));
        if (variant == BENCH_SPLIT)
            bench_ze_check("zeCommandListAppendLaunchKernel correction",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->correction,
                                                           &groups, NULL, 0, NULL));
    }
    bench_ze_check("zeCommandListHostSynchronize",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static bench_sample bench_measure(bench_gpu *gpu, int variant,
                                  const bench_shape *shape, int repetitions) {
    bench_telemetry telemetry;
    bench_telemetry_start(&telemetry);
    double seconds = bench_launch(gpu, variant, shape, repetitions);
    bench_telemetry_stop(&telemetry);
    return (bench_sample) {
        .seconds = seconds / repetitions,
        .actual_mhz = (double)telemetry.actual_sum / telemetry.samples,
        .requested_mhz = (double)telemetry.requested_sum / telemetry.samples,
        .actual_min = telemetry.actual_min,
        .actual_max = telemetry.actual_max,
        .pl1 = telemetry.pl1,
        .pl2 = telemetry.pl2,
        .thermal = telemetry.thermal
    };
}

static float bench_reference(const bench_data *data, const bench_shape *shape,
                             int output_row, int output_column) {
    int group = output_column >> 3;
    int row = output_column & 7;
    float sum = 0.0f;
    for (int block = 0; block < shape->blocks; block++) {
        size_t weight_base = ((size_t)group * shape->blocks + block) * 128
                             + row * 4;
        size_t activation_base =
            ((size_t)output_row * shape->blocks + block) * 32;
        int integer = 0;
        for (int chunk = 0; chunk < 4; chunk++) {
            for (int byte = 0; byte < 4; byte++) {
                uint8_t packed = data->weight_x8[weight_base + chunk * 32 + byte];
                integer += (packed & 15)
                           * data->activation[activation_base + chunk * 4 + byte];
                integer += (packed >> 4)
                           * data->activation[activation_base + 16
                                              + chunk * 4 + byte];
            }
        }
        size_t activation_scale = (size_t)output_row * shape->blocks + block;
        size_t weight_scale = ((size_t)group * shape->blocks + block) * 8 + row;
        integer -= 8 * data->activation_sigma[activation_scale];
        sum += (float)integer * (float)data->activation_scale[activation_scale]
               * (float)data->weight_scale_x8[weight_scale];
    }
    return sum;
}

static void bench_verify(bench_gpu *gpu, int variant, const char *name,
                         const bench_data *data, const bench_shape *shape) {
    bench_launch(gpu, variant, shape, 1);
    int probes = shape->m * shape->n < 256 ? shape->m * shape->n : 256;
    double error = 0.0;
    double reference = 0.0;
    double max_abs = 0.0;
    for (int probe = 0; probe < probes; probe++) {
        int row = (probe * 37 + 11) % shape->m;
        int column = (probe * 101 + 17) % shape->n;
        double expected = bench_reference(data, shape, row, column);
        double difference = (double)data->output[(size_t)row * shape->n + column]
                            - expected;
        error += difference * difference;
        reference += expected * expected;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    double rel_rms = sqrt(error / (reference + 1e-30));
    printf("prefill-gemm: correctness %s rel-rms %.9g max-abs %.9g probes %d\n",
           name, rel_rms, max_abs, probes);
    if (!isfinite(rel_rms) || rel_rms > 5e-5)
        bench_fatal("%s correctness failed", name);
}

static int bench_double_compare(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double bench_median(const double *values, int count) {
    double copy[BENCH_ROUNDS_MAX];
    memcpy(copy, values, (size_t)count * sizeof *values);
    qsort(copy, (size_t)count, sizeof *copy, bench_double_compare);
    return copy[count / 2];
}

static void bench_shape_run(bench_gpu *gpu, const bench_shape *shape,
                            int rounds, double target_seconds) {
    static const char *variant_names[BENCH_VARIANTS] = {
        "x8-fused", "row", "split", "tn48", "tn64", "tn64-wg256",
        "tn128-wg256"
    };
    bench_data data = {0};
    bench_data_init(gpu, &data, shape);
    printf("prefill-gemm: shape %s M %d N %d K %d\n",
           shape->name, shape->m, shape->n, shape->blocks * 32);
    bench_verify(gpu, BENCH_FUSED, "x8-fused", &data, shape);
    bench_verify(gpu, BENCH_ROW, "row", &data, shape);
    bench_verify(gpu, BENCH_SPLIT, "split", &data, shape);
    bench_verify(gpu, BENCH_TN48, "tn48", &data, shape);
    bench_verify(gpu, BENCH_TN64, "tn64", &data, shape);
    bench_verify(gpu, BENCH_TN64_WG256, "tn64-wg256", &data, shape);
    bench_verify(gpu, BENCH_TN128_WG256, "tn128-wg256", &data, shape);
    double probe = bench_launch(gpu, BENCH_FUSED, shape, 3) / 3.0;
    int repetitions = (int)ceil(target_seconds / probe);
    if (repetitions < 3) repetitions = 3;
    printf("prefill-gemm: repetitions %d target %.3f s/variant\n",
           repetitions, target_seconds);

    bench_sample samples[BENCH_VARIANTS][BENCH_ROUNDS_MAX];
    double times[BENCH_VARIANTS][BENCH_ROUNDS_MAX];
    double frequencies[BENCH_VARIANTS][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        for (int position = 0; position < BENCH_VARIANTS; position++) {
            int variant = (round + position) % BENCH_VARIANTS;
            samples[variant][round] =
                bench_measure(gpu, variant, shape, repetitions);
            times[variant][round] = samples[variant][round].seconds;
            frequencies[variant][round] = samples[variant][round].actual_mhz;
        }
        printf("prefill-gemm: round %d", round + 1);
        for (int variant = 0; variant < BENCH_VARIANTS; variant++) {
            bench_sample *sample = &samples[variant][round];
            printf(" %s %.6f ms@%.0fMHz[%ld,%ld] throttle=%ld/%ld/%ld",
                   variant_names[variant], sample->seconds * 1e3,
                   sample->actual_mhz, sample->actual_min, sample->actual_max,
                   sample->pl1, sample->pl2, sample->thermal);
        }
        printf("\n");
    }

    double median_time[BENCH_VARIANTS];
    double median_frequency[BENCH_VARIANTS];
    for (int variant = 0; variant < BENCH_VARIANTS; variant++) {
        median_time[variant] = bench_median(times[variant], rounds);
        median_frequency[variant] = bench_median(frequencies[variant], rounds);
    }
    double operations = 2.0 * shape->m * shape->n * shape->blocks * 32;
    double row_cost = median_time[BENCH_ROW] / median_time[BENCH_FUSED];
    double split_cost = median_time[BENCH_SPLIT] / median_time[BENCH_FUSED];
    double tn48_cost = median_time[BENCH_TN48] / median_time[BENCH_FUSED];
    double tn64_cost = median_time[BENCH_TN64] / median_time[BENCH_FUSED];
    double tn64_wg256_cost = median_time[BENCH_TN64_WG256]
                             / median_time[BENCH_FUSED];
    double tn128_wg256_cost = median_time[BENCH_TN128_WG256]
                              / median_time[BENCH_FUSED];
    double minimum_frequency = median_frequency[0];
    double maximum_frequency = median_frequency[0];
    for (int variant = 1; variant < BENCH_VARIANTS; variant++) {
        minimum_frequency = fmin(minimum_frequency, median_frequency[variant]);
        maximum_frequency = fmax(maximum_frequency, median_frequency[variant]);
    }
    double frequency_span = maximum_frequency / minimum_frequency;
    printf("prefill-gemm: median");
    for (int variant = 0; variant < BENCH_VARIANTS; variant++)
        printf(" %s %.6f ms %.6f TFLOP/s @%.0fMHz",
               variant_names[variant], median_time[variant] * 1e3,
               operations / median_time[variant] / 1e12,
               median_frequency[variant]);
    printf(" row-cost %.6fx split-cost %.6fx tn48-cost %.6fx tn64-cost %.6fx tn64-wg256-cost %.6fx tn128-wg256-cost %.6fx frequency-span %.6fx\n",
           row_cost, split_cost, tn48_cost, tn64_cost, tn64_wg256_cost,
           tn128_wg256_cost, frequency_span);
    printf("prefill-gemm: decision layout %s correction %s tile %s frequency %s\n",
           row_cost >= 1.03 ? "x8-go" : "open",
           split_cost <= 0.97 ? "split-go"
           : split_cost >= 1.03 ? "fused-go" : "open",
           tn128_wg256_cost <= fmin(tn48_cost, tn64_cost) * 0.97
             ? "tn128-wg256-go"
           : tn64_wg256_cost <= fmin(tn48_cost, tn64_cost) * 0.97
             ? "tn64-wg256-go"
           : tn64_cost <= tn48_cost * 0.97 && tn64_cost <= 0.97 ? "tn64-go"
           : tn48_cost <= tn64_cost * 0.97 && tn48_cost <= 0.97 ? "tn48-go"
           : tn64_cost >= 1.03 && tn48_cost >= 1.03
             && tn64_wg256_cost >= 1.03
             ? "tn32-go" : "open",
           frequency_span <= 1.05 && median_frequency[BENCH_FUSED] >= 1000.0
           ? "comparable" : "reject");
    bench_data_destroy(gpu, &data);
}

static void bench_shape_tile_run(bench_gpu *gpu, const bench_shape *shape,
                                 int rounds, double target_seconds) {
    bench_data data = {0};
    bench_data_init(gpu, &data, shape);
    bench_verify(gpu, BENCH_FUSED, "tn32-control", &data, shape);
    bench_verify(gpu, BENCH_TN48, "tn48", &data, shape);
    bench_verify(gpu, BENCH_TN64, "tn64", &data, shape);
    bench_verify(gpu, BENCH_TN64_WG256, "tn64-wg256", &data, shape);
    bench_verify(gpu, BENCH_TN128_WG256, "tn128-wg256", &data, shape);
    double probe32 = bench_launch(gpu, BENCH_FUSED, shape, 3) / 3.0;
    double probe48 = bench_launch(gpu, BENCH_TN48, shape, 3) / 3.0;
    double probe64 = bench_launch(gpu, BENCH_TN64, shape, 3) / 3.0;
    double probe256 = bench_launch(gpu, BENCH_TN64_WG256, shape, 3) / 3.0;
    double probe128 = bench_launch(gpu, BENCH_TN128_WG256, shape, 3) / 3.0;
    int repetitions[5] = {
        (int)ceil(target_seconds / probe32),
        (int)ceil(target_seconds / probe48),
        (int)ceil(target_seconds / probe64),
        (int)ceil(target_seconds / probe256),
        (int)ceil(target_seconds / probe128)
    };
    for (int variant = 0; variant < 5; variant++)
        if (repetitions[variant] < 3) repetitions[variant] = 3;
    printf("prefill-tile: shape %s M %d N %d K %d repetitions tn32/tn48/tn64/tn64-wg256/tn128-wg256 %d/%d/%d/%d/%d target %.3f s\n",
           shape->name, shape->m, shape->n, shape->blocks * 32,
           repetitions[0], repetitions[1], repetitions[2], repetitions[3],
           repetitions[4],
           target_seconds);
    bench_sample samples[5][BENCH_ROUNDS_MAX];
    double times[5][BENCH_ROUNDS_MAX];
    double frequencies[5][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        for (int position = 0; position < 5; position++) {
            int variant = (round + position) % 5;
            int kernel_variant = variant == 0 ? BENCH_FUSED
                                 : variant == 1 ? BENCH_TN48
                                 : variant == 2 ? BENCH_TN64
                                 : variant == 3 ? BENCH_TN64_WG256
                                 : BENCH_TN128_WG256;
            samples[variant][round] = bench_measure(
                gpu, kernel_variant, shape, repetitions[variant]);
            times[variant][round] = samples[variant][round].seconds;
            frequencies[variant][round] = samples[variant][round].actual_mhz;
        }
        printf("prefill-tile: round %d tn32 %.6f ms@%.0fMHz tn48 %.6f ms@%.0fMHz tn64 %.6f ms@%.0fMHz tn64-wg256 %.6f ms@%.0fMHz tn128-wg256 %.6f ms@%.0fMHz\n",
               round + 1, samples[0][round].seconds * 1e3,
               samples[0][round].actual_mhz, samples[1][round].seconds * 1e3,
               samples[1][round].actual_mhz, samples[2][round].seconds * 1e3,
               samples[2][round].actual_mhz, samples[3][round].seconds * 1e3,
               samples[3][round].actual_mhz, samples[4][round].seconds * 1e3,
               samples[4][round].actual_mhz);
    }
    double tn32 = bench_median(times[0], rounds);
    double tn48 = bench_median(times[1], rounds);
    double tn64 = bench_median(times[2], rounds);
    double tn64_wg256 = bench_median(times[3], rounds);
    double tn128_wg256 = bench_median(times[4], rounds);
    double frequency32 = bench_median(frequencies[0], rounds);
    double frequency48 = bench_median(frequencies[1], rounds);
    double frequency64 = bench_median(frequencies[2], rounds);
    double frequency256 = bench_median(frequencies[3], rounds);
    double frequency128 = bench_median(frequencies[4], rounds);
    double frequency_span = fmax(fmax(frequency32, frequency48),
                                 fmax(frequency64,
                                      fmax(frequency256, frequency128)))
                            / fmin(fmin(frequency32, frequency48),
                                   fmin(frequency64,
                                        fmin(frequency256, frequency128)));
    printf("prefill-tile: median tn32 %.6f ms @%.0fMHz tn48 %.6f ms @%.0fMHz tn64 %.6f ms @%.0fMHz tn64-wg256 %.6f ms @%.0fMHz tn128-wg256 %.6f ms @%.0fMHz speedup-64-vs-128 %.6fx frequency-span %.6fx decision %s\n",
           tn32 * 1e3, frequency32, tn48 * 1e3, frequency48,
           tn64 * 1e3, frequency64,
           tn64_wg256 * 1e3, frequency256,
           tn128_wg256 * 1e3, frequency128, tn64 / tn128_wg256,
           frequency_span,
           frequency_span > 1.05 ? "frequency-reject"
           : tn128_wg256 <= tn64 * 0.97 ? "tn128-wg256-go"
           : tn64_wg256 <= fmin(tn48, tn64) * 0.97 ? "tn64-wg256-go"
           : tn64 <= tn48 * 0.97 && tn64 <= tn32 * 0.97 ? "tn64-go"
           : tn48 <= tn64 * 0.97 && tn48 <= tn32 * 0.97 ? "tn48-go"
           : "tn32-go");
    bench_data_destroy(gpu, &data);
}

static void bench_b5_run(bench_gpu *gpu, int rounds, double target_seconds) {
    const bench_shape shape = { "b5-dense", 512, 4096, 88 };
    bench_data data = {0};
    bench_data_init(gpu, &data, &shape);
    bench_verify(gpu, BENCH_TN64, "tm32-tn64", &data, &shape);
    bench_verify(gpu, BENCH_TM64_TN64, "tm64-tn64", &data, &shape);
    double probe = bench_launch(gpu, BENCH_TN64, &shape, 3) / 3.0;
    int repetitions = (int)ceil(target_seconds / probe);
    if (repetitions < 3) repetitions = 3;
    double times[2][BENCH_ROUNDS_MAX];
    double frequencies[2][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        printf("prefill-b5: round %d", round + 1);
        for (int position = 0; position < 2; position++) {
            int variant = (round + position) & 1;
            int kernel = variant ? BENCH_TM64_TN64 : BENCH_TN64;
            bench_sample sample = bench_measure(
                gpu, kernel, &shape, repetitions);
            times[variant][round] = sample.seconds;
            frequencies[variant][round] = sample.actual_mhz;
            printf(" %s %.6f ms@%.0fMHz[%ld,%ld]",
                   variant ? "tm64-tn64" : "tm32-tn64",
                   sample.seconds * 1e3, sample.actual_mhz,
                   sample.actual_min, sample.actual_max);
        }
        printf("\n");
    }
    double tm32 = bench_median(times[0], rounds);
    double tm64 = bench_median(times[1], rounds);
    double f32 = bench_median(frequencies[0], rounds);
    double f64 = bench_median(frequencies[1], rounds);
    double operations = 2.0 * shape.m * shape.n * shape.blocks * 32;
    printf("prefill-b5: median tm32-tn64 %.6f ms %.6f TFLOP/s @%.0fMHz tm64-tn64 %.6f ms %.6f TFLOP/s @%.0fMHz speedup %.6fx frequency-span %.6fx decision %s\n",
           tm32 * 1e3, operations / tm32 / 1e12, f32,
           tm64 * 1e3, operations / tm64 / 1e12, f64, tm32 / tm64,
           fmax(f32, f64) / fmin(f32, f64),
           tm64 <= tm32 * 0.90 ? "tm64-go" : "tm32-go");
    bench_data_destroy(gpu, &data);
}

static double bench_fusion_launch(bench_gpu *gpu, const bench_shape *combined,
                                  const bench_shape *parts, int part_count,
                                  int fused, int repetitions) {
    ze_group_count_t combined_groups = {
        (uint32_t)((combined->m + 31) / 32),
        (uint32_t)((combined->n + 63) / 64), 1
    };
    ze_kernel_handle_t kernels[3] = {
        gpu->tn64_alt1, gpu->tn64_alt2, gpu->tn64_alt3
    };
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        if (fused) {
            bench_ze_check("zeCommandListAppendLaunchKernel fused projection",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->tn64, &combined_groups,
                               NULL, 0, NULL));
        } else {
            for (int part = 0; part < part_count; part++) {
                ze_group_count_t groups = {
                    (uint32_t)((parts[part].m + 31) / 32),
                    (uint32_t)((parts[part].n + 63) / 64), 1
                };
                bench_ze_check("zeCommandListAppendLaunchKernel split projection",
                               zeCommandListAppendLaunchKernel(
                                   gpu->commands, kernels[part], &groups,
                                   NULL, 0, NULL));
            }
        }
    }
    bench_ze_check("zeCommandListHostSynchronize projection fusion",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static void bench_fusion_run(bench_gpu *gpu, const char *name, int rounds,
                             double target_seconds) {
    bench_shape combined;
    bench_shape parts[3];
    int part_count;
    if (!name || !strcmp(name, "dense")) {
        combined = (bench_shape){ "dense-gu-pair", 512, 4224, 88 };
        parts[0] = (bench_shape){ "dense-gate", 512, 2112, 88 };
        parts[1] = (bench_shape){ "dense-up", 512, 2112, 88 };
        part_count = 2;
    } else if (!strcmp(name, "swa")) {
        combined = (bench_shape){ "swa-qkv", 512, 8192, 88 };
        parts[0] = (bench_shape){ "swa-q", 512, 4096, 88 };
        parts[1] = (bench_shape){ "swa-k", 512, 2048, 88 };
        parts[2] = (bench_shape){ "swa-v", 512, 2048, 88 };
        part_count = 3;
    } else if (!strcmp(name, "global")) {
        combined = (bench_shape){ "global-qk", 512, 9216, 88 };
        parts[0] = (bench_shape){ "global-q", 512, 8192, 88 };
        parts[1] = (bench_shape){ "global-k", 512, 1024, 88 };
        part_count = 2;
    } else {
        bench_fatal("unknown fusion shape: %s", name);
    }

    bench_data combined_data = {0};
    bench_data part_data[3] = {{0}};
    bench_data_init(gpu, &combined_data, &combined);
    for (int part = 0; part < part_count; part++)
        bench_data_init(gpu, &part_data[part], &parts[part]);
    bench_set_gemm(gpu->tn64, combined_data.weight_x8,
                   combined_data.weight_scale_x8, &combined_data, &combined);
    ze_kernel_handle_t kernels[3] = {
        gpu->tn64_alt1, gpu->tn64_alt2, gpu->tn64_alt3
    };
    for (int part = 0; part < part_count; part++)
        bench_set_gemm(kernels[part], part_data[part].weight_x8,
                       part_data[part].weight_scale_x8, &part_data[part],
                       &parts[part]);

    double probes[2] = {
        bench_fusion_launch(gpu, &combined, parts, part_count, 0, 3) / 3.0,
        bench_fusion_launch(gpu, &combined, parts, part_count, 1, 3) / 3.0
    };
    int repetitions[2] = {
        (int)ceil(target_seconds / probes[0]),
        (int)ceil(target_seconds / probes[1])
    };
    for (int variant = 0; variant < 2; variant++)
        if (repetitions[variant] < 3) repetitions[variant] = 3;
    printf("prefill-fusion: shape %s parts %d repetitions split/fused %d/%d target %.3f s\n",
           name ? name : "dense", part_count, repetitions[0], repetitions[1],
           target_seconds);

    double times[2][BENCH_ROUNDS_MAX];
    double frequencies[2][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        for (int position = 0; position < 2; position++) {
            int variant = (round + position) & 1;
            bench_telemetry telemetry;
            bench_telemetry_start(&telemetry);
            double seconds = bench_fusion_launch(
                gpu, &combined, parts, part_count, variant,
                repetitions[variant]);
            bench_telemetry_stop(&telemetry);
            times[variant][round] = seconds / repetitions[variant];
            frequencies[variant][round] =
                (double)telemetry.actual_sum / telemetry.samples;
        }
        printf("prefill-fusion: round %d split %.6f ms@%.0fMHz fused %.6f ms@%.0fMHz\n",
               round + 1, times[0][round] * 1e3, frequencies[0][round],
               times[1][round] * 1e3, frequencies[1][round]);
    }
    double split = bench_median(times[0], rounds);
    double fused = bench_median(times[1], rounds);
    double split_frequency = bench_median(frequencies[0], rounds);
    double fused_frequency = bench_median(frequencies[1], rounds);
    double frequency_span = fmax(split_frequency, fused_frequency)
                            / fmin(split_frequency, fused_frequency);
    printf("prefill-fusion: median split %.6f ms @%.0fMHz fused %.6f ms @%.0fMHz speedup %.6fx frequency-span %.6fx decision %s\n",
           split * 1e3, split_frequency, fused * 1e3, fused_frequency,
           split / fused, frequency_span,
           frequency_span > 1.05 ? "frequency-reject"
           : fused <= split * 0.97 ? "fused-go" : "split-go");

    for (int part = part_count - 1; part >= 0; part--)
        bench_data_destroy(gpu, &part_data[part]);
    bench_data_destroy(gpu, &combined_data);
}

static void bench_moe_set_kernel(ze_kernel_handle_t kernel,
                                 const bench_moe_data *data,
                                 const int *tile_expert, const int *tile_m0) {
    bench_pointer_arg(kernel, 0, data->weight);
    bench_pointer_arg(kernel, 1, data->weight_scale);
    bench_pointer_arg(kernel, 2, data->activation);
    bench_pointer_arg(kernel, 3, data->activation_scale);
    bench_pointer_arg(kernel, 4, data->activation_sigma);
    bench_pointer_arg(kernel, 5, data->output);
    bench_pointer_arg(kernel, 6, data->expert_count);
    bench_pointer_arg(kernel, 7, data->token_offset);
    bench_pointer_arg(kernel, 8, tile_expert);
    bench_pointer_arg(kernel, 9, tile_m0);
    bench_int_arg(kernel, 10, BENCH_MOE_N);
    bench_int_arg(kernel, 11, BENCH_MOE_BLOCKS);
}

static void bench_moe_set_expert_kernel(ze_kernel_handle_t kernel,
                                        const bench_moe_data *data) {
    bench_pointer_arg(kernel, 0, data->weight);
    bench_pointer_arg(kernel, 1, data->weight_scale);
    bench_pointer_arg(kernel, 2, data->activation);
    bench_pointer_arg(kernel, 3, data->activation_scale);
    bench_pointer_arg(kernel, 4, data->activation_sigma);
    bench_pointer_arg(kernel, 5, data->output);
    bench_pointer_arg(kernel, 6, data->expert_count);
    bench_pointer_arg(kernel, 7, data->token_offset);
    bench_int_arg(kernel, 8, BENCH_MOE_N);
    bench_int_arg(kernel, 9, BENCH_MOE_BLOCKS);
}

static void bench_moe_data_init(bench_gpu *gpu, bench_moe_data *data) {
    memset(data, 0, sizeof *data);
    size_t weight_bytes = (size_t)BENCH_MOE_EXPERTS * BENCH_MOE_N
                          * BENCH_MOE_BLOCKS * 16;
    size_t weight_scale_bytes = (size_t)BENCH_MOE_EXPERTS * BENCH_MOE_N
                                * BENCH_MOE_BLOCKS * sizeof(_Float16);
    size_t activation_bytes = (size_t)BENCH_MOE_ROWS * BENCH_MOE_BLOCKS * 32;
    size_t activation_scale_bytes = (size_t)BENCH_MOE_ROWS * BENCH_MOE_BLOCKS
                                    * sizeof(_Float16);
    data->weight = bench_gpu_alloc(gpu, weight_bytes);
    data->weight_scale = bench_gpu_alloc(gpu, weight_scale_bytes);
    data->activation = bench_gpu_alloc(gpu, activation_bytes);
    data->activation_scale = bench_gpu_alloc(gpu, activation_scale_bytes);
    data->activation_sigma = bench_gpu_alloc(gpu, activation_scale_bytes);
    data->source_activation = bench_gpu_alloc(
        gpu, (size_t)BENCH_MOE_TOKENS * BENCH_MOE_BLOCKS * 32);
    data->source_scale = bench_gpu_alloc(
        gpu, (size_t)BENCH_MOE_TOKENS * BENCH_MOE_BLOCKS * sizeof(_Float16));
    data->source_sigma = bench_gpu_alloc(
        gpu, (size_t)BENCH_MOE_TOKENS * BENCH_MOE_BLOCKS * sizeof(int16_t));
    data->output = bench_gpu_alloc(gpu,
                                   (size_t)BENCH_MOE_ROWS * BENCH_MOE_N
                                   * sizeof(float));
    data->expert_count = bench_gpu_alloc(gpu, BENCH_MOE_EXPERTS * sizeof(int));
    data->token_offset = bench_gpu_alloc(gpu,
                                         (BENCH_MOE_EXPERTS + 1) * sizeof(int));
    data->padded_expert = bench_gpu_alloc(gpu,
                                          BENCH_MOE_PADDED_TILES * sizeof(int));
    data->padded_m0 = bench_gpu_alloc(gpu, BENCH_MOE_PADDED_TILES * sizeof(int));
    data->padded64_expert = bench_gpu_alloc(gpu,
                                            BENCH_MOE_PADDED_TILES * sizeof(int));
    data->padded64_m0 = bench_gpu_alloc(gpu,
                                        BENCH_MOE_PADDED_TILES * sizeof(int));
    data->padded16_expert = bench_gpu_alloc(
        gpu, BENCH_MOE_PADDED16_TILES * sizeof(int));
    data->padded16_m0 = bench_gpu_alloc(
        gpu, BENCH_MOE_PADDED16_TILES * sizeof(int));
    data->padded24_expert = bench_gpu_alloc(
        gpu, BENCH_MOE_PADDED24_TILES * sizeof(int));
    data->padded24_m0 = bench_gpu_alloc(
        gpu, BENCH_MOE_PADDED24_TILES * sizeof(int));
    data->full_expert = bench_gpu_alloc(gpu, BENCH_MOE_FULL_TILES * sizeof(int));
    data->full_m0 = bench_gpu_alloc(gpu, BENCH_MOE_FULL_TILES * sizeof(int));
    data->tail_expert = bench_gpu_alloc(gpu, BENCH_MOE_TAIL_TILES * sizeof(int));
    data->tail_m0 = bench_gpu_alloc(gpu, BENCH_MOE_TAIL_TILES * sizeof(int));
    data->padded_launch = bench_gpu_alloc(gpu, sizeof *data->padded_launch);
    data->route_expert = bench_gpu_alloc(gpu, BENCH_MOE_ROWS * sizeof(int));
    data->route_token = bench_gpu_alloc(gpu, BENCH_MOE_ROWS * sizeof(int));
    data->packed_route = bench_gpu_alloc(gpu, BENCH_MOE_ROWS * sizeof(int));
    data->route_packed = bench_gpu_alloc(gpu, BENCH_MOE_ROWS * sizeof(int));
    data->route_cursor = bench_gpu_alloc(gpu,
                                         BENCH_MOE_EXPERTS * sizeof(int));
    data->route_weight = bench_gpu_alloc(gpu, BENCH_MOE_ROWS * sizeof(float));
    data->reduced = bench_gpu_alloc(gpu,
                                    (size_t)BENCH_MOE_TOKENS * BENCH_MOE_N
                                    * sizeof(float));

    uint32_t random = UINT32_C(0x510e527f);
    for (int expert = 0; expert < BENCH_MOE_EXPERTS; expert++) {
        for (int group = 0; group < BENCH_MOE_N / 8; group++) {
            for (int block = 0; block < BENCH_MOE_BLOCKS; block++) {
                for (int row = 0; row < 8; row++) {
                    size_t scale = (((size_t)expert * (BENCH_MOE_N / 8) + group)
                                    * BENCH_MOE_BLOCKS + block) * 8 + row;
                    data->weight_scale[scale] =
                        (_Float16)((float)(bench_random(&random) % 200 + 1)
                                   / 1000.0f);
                    for (int chunk = 0; chunk < 4; chunk++) {
                        for (int byte = 0; byte < 4; byte++) {
                            size_t index = (((size_t)expert * (BENCH_MOE_N / 8)
                                             + group) * BENCH_MOE_BLOCKS + block)
                                           * 128 + chunk * 32 + row * 4 + byte;
                            data->weight[index] = (uint8_t)bench_random(&random);
                        }
                    }
                }
            }
        }
    }
    for (int row = 0; row < BENCH_MOE_ROWS; row++) {
        for (int block = 0; block < BENCH_MOE_BLOCKS; block++) {
            size_t base = ((size_t)row * BENCH_MOE_BLOCKS + block) * 32;
            int sigma = 0;
            for (int k = 0; k < 32; k++) {
                int value = (int)(bench_random(&random) % 255) - 127;
                data->activation[base + k] = (int8_t)value;
                sigma += value;
            }
            size_t index = (size_t)row * BENCH_MOE_BLOCKS + block;
            data->activation_scale[index] =
                (_Float16)((float)(bench_random(&random) % 200 + 1) / 1000.0f);
            data->activation_sigma[index] = (int16_t)sigma;
        }
    }
    for (int row = 0; row < BENCH_MOE_TOKENS; row++) {
        for (int block = 0; block < BENCH_MOE_BLOCKS; block++) {
            size_t base = ((size_t)row * BENCH_MOE_BLOCKS + block) * 32;
            int sigma = 0;
            for (int k = 0; k < 32; k++) {
                int value = (int)(bench_random(&random) % 255) - 127;
                data->source_activation[base + k] = (int8_t)value;
                sigma += value;
            }
            size_t index = (size_t)row * BENCH_MOE_BLOCKS + block;
            data->source_scale[index] =
                (_Float16)((float)(bench_random(&random) % 200 + 1) / 1000.0f);
            data->source_sigma[index] = (int16_t)sigma;
        }
    }
}

static void bench_moe_data_destroy(bench_gpu *gpu, bench_moe_data *data) {
    bench_ze_check("zeMemFree reduced", zeMemFree(gpu->context, data->reduced));
    bench_ze_check("zeMemFree route weight",
                   zeMemFree(gpu->context, data->route_weight));
    bench_ze_check("zeMemFree route cursor",
                   zeMemFree(gpu->context, data->route_cursor));
    bench_ze_check("zeMemFree packed route",
                   zeMemFree(gpu->context, data->packed_route));
    bench_ze_check("zeMemFree route packed",
                   zeMemFree(gpu->context, data->route_packed));
    bench_ze_check("zeMemFree route token",
                   zeMemFree(gpu->context, data->route_token));
    bench_ze_check("zeMemFree route expert",
                   zeMemFree(gpu->context, data->route_expert));
    bench_ze_check("zeMemFree padded launch",
                   zeMemFree(gpu->context, data->padded_launch));
    bench_ze_check("zeMemFree tail m0", zeMemFree(gpu->context, data->tail_m0));
    bench_ze_check("zeMemFree tail expert",
                   zeMemFree(gpu->context, data->tail_expert));
    bench_ze_check("zeMemFree full m0", zeMemFree(gpu->context, data->full_m0));
    bench_ze_check("zeMemFree full expert",
                   zeMemFree(gpu->context, data->full_expert));
    bench_ze_check("zeMemFree padded m0",
                   zeMemFree(gpu->context, data->padded_m0));
    bench_ze_check("zeMemFree padded expert",
                   zeMemFree(gpu->context, data->padded_expert));
    bench_ze_check("zeMemFree padded64 m0",
                   zeMemFree(gpu->context, data->padded64_m0));
    bench_ze_check("zeMemFree padded64 expert",
                   zeMemFree(gpu->context, data->padded64_expert));
    bench_ze_check("zeMemFree padded16 m0",
                   zeMemFree(gpu->context, data->padded16_m0));
    bench_ze_check("zeMemFree padded16 expert",
                   zeMemFree(gpu->context, data->padded16_expert));
    bench_ze_check("zeMemFree padded24 m0",
                   zeMemFree(gpu->context, data->padded24_m0));
    bench_ze_check("zeMemFree padded24 expert",
                   zeMemFree(gpu->context, data->padded24_expert));
    bench_ze_check("zeMemFree token offset",
                   zeMemFree(gpu->context, data->token_offset));
    bench_ze_check("zeMemFree expert count",
                   zeMemFree(gpu->context, data->expert_count));
    bench_ze_check("zeMemFree MoE output", zeMemFree(gpu->context, data->output));
    bench_ze_check("zeMemFree MoE sigma",
                   zeMemFree(gpu->context, data->activation_sigma));
    bench_ze_check("zeMemFree MoE source sigma",
                   zeMemFree(gpu->context, data->source_sigma));
    bench_ze_check("zeMemFree MoE source scale",
                   zeMemFree(gpu->context, data->source_scale));
    bench_ze_check("zeMemFree MoE source activation",
                   zeMemFree(gpu->context, data->source_activation));
    bench_ze_check("zeMemFree MoE activation scale",
                   zeMemFree(gpu->context, data->activation_scale));
    bench_ze_check("zeMemFree MoE activation",
                   zeMemFree(gpu->context, data->activation));
    bench_ze_check("zeMemFree MoE weight scale",
                   zeMemFree(gpu->context, data->weight_scale));
    bench_ze_check("zeMemFree MoE weight", zeMemFree(gpu->context, data->weight));
}

static void bench_moe_schedule(bench_moe_data *data, const char *distribution) {
    memset(data->expert_count, 0, BENCH_MOE_EXPERTS * sizeof(int));
    if (!strcmp(distribution, "uniform")) {
        for (int expert = 0; expert < BENCH_MOE_EXPERTS; expert++)
            data->expert_count[expert] = BENCH_MOE_ROWS / BENCH_MOE_EXPERTS;
    } else if (!strcmp(distribution, "multinomial")) {
        uint32_t random = UINT32_C(0x1f83d9ab);
        for (int row = 0; row < BENCH_MOE_ROWS; row++)
            data->expert_count[(bench_random(&random) >> 16)
                               % BENCH_MOE_EXPERTS]++;
    } else if (!strcmp(distribution, "adversarial")) {
        for (int expert = 0; expert < BENCH_MOE_EXPERTS; expert++)
            data->expert_count[expert] = expert < BENCH_MOE_EXPERTS / 2 ? 1 : 63;
    } else {
        bench_fatal("unknown MoE distribution: %s", distribution);
    }

    data->token_offset[0] = 0;
    for (int expert = 0; expert < BENCH_MOE_EXPERTS; expert++)
        data->token_offset[expert + 1] = data->token_offset[expert]
                                         + data->expert_count[expert];
    if (data->token_offset[BENCH_MOE_EXPERTS] != BENCH_MOE_ROWS)
        bench_fatal("MoE distribution has %d rows",
                    data->token_offset[BENCH_MOE_EXPERTS]);

    int route = 0;
    for (int expert = 0; expert < BENCH_MOE_EXPERTS; expert++)
        for (int index = 0; index < data->expert_count[expert]; index++)
            data->route_expert[route++] = expert;
    uint32_t route_random = UINT32_C(0x3c6ef372);
    for (int index = BENCH_MOE_ROWS - 1; index > 0; index--) {
        int other = (int)(bench_random(&route_random) % (uint32_t)(index + 1));
        int temporary = data->route_expert[index];
        data->route_expert[index] = data->route_expert[other];
        data->route_expert[other] = temporary;
    }
    memcpy(data->route_cursor, data->token_offset,
           BENCH_MOE_EXPERTS * sizeof(int));
    for (int index = 0; index < BENCH_MOE_ROWS; index++) {
        int expert = data->route_expert[index];
        data->route_token[index] = index / 8;
        int packed = data->route_cursor[expert]++;
        data->packed_route[packed] = index;
        data->route_packed[index] = packed;
        data->route_weight[index] = (float)(index % 8 + 1) / 36.0f;
    }
    for (int packed = 0; packed < BENCH_MOE_ROWS; packed++) {
        int token = data->route_token[data->packed_route[packed]];
        memcpy(data->activation + (size_t)packed * BENCH_MOE_BLOCKS * 32,
               data->source_activation + (size_t)token * BENCH_MOE_BLOCKS * 32,
               BENCH_MOE_BLOCKS * 32);
        memcpy(data->activation_scale + (size_t)packed * BENCH_MOE_BLOCKS,
               data->source_scale + (size_t)token * BENCH_MOE_BLOCKS,
               BENCH_MOE_BLOCKS * sizeof(_Float16));
        memcpy(data->activation_sigma + (size_t)packed * BENCH_MOE_BLOCKS,
               data->source_sigma + (size_t)token * BENCH_MOE_BLOCKS,
               BENCH_MOE_BLOCKS * sizeof(int16_t));
    }

    for (int tile = 0; tile < BENCH_MOE_PADDED_TILES; tile++) {
        data->padded_expert[tile] = -1;
        data->padded_m0[tile] = 0;
        data->padded64_expert[tile] = -1;
        data->padded64_m0[tile] = 0;
    }
    for (int tile = 0; tile < BENCH_MOE_PADDED16_TILES; tile++) {
        data->padded16_expert[tile] = -1;
        data->padded16_m0[tile] = 0;
    }
    for (int tile = 0; tile < BENCH_MOE_PADDED24_TILES; tile++) {
        data->padded24_expert[tile] = -1;
        data->padded24_m0[tile] = 0;
    }
    for (int tile = 0; tile < BENCH_MOE_FULL_TILES; tile++) {
        data->full_expert[tile] = -1;
        data->full_m0[tile] = 0;
    }
    for (int tile = 0; tile < BENCH_MOE_TAIL_TILES; tile++) {
        data->tail_expert[tile] = -1;
        data->tail_m0[tile] = 0;
    }

    data->padded_valid = 0;
    data->padded64_valid = 0;
    data->padded16_valid = 0;
    data->padded24_valid = 0;
    data->full_valid = 0;
    data->tail_valid = 0;
    for (int expert = 0; expert < BENCH_MOE_EXPERTS; expert++) {
        int count = data->expert_count[expert];
        for (int m0 = 0; m0 < count; m0 += 16) {
            if (data->padded16_valid >= BENCH_MOE_PADDED16_TILES)
                bench_fatal("too many padded16 MoE tiles");
            data->padded16_expert[data->padded16_valid] = expert;
            data->padded16_m0[data->padded16_valid++] = m0;
        }
        for (int m0 = 0; m0 < count; m0 += 24) {
            if (data->padded24_valid >= BENCH_MOE_PADDED24_TILES)
                bench_fatal("too many padded24 MoE tiles");
            data->padded24_expert[data->padded24_valid] = expert;
            data->padded24_m0[data->padded24_valid++] = m0;
        }
        for (int m0 = 0; m0 < count; m0 += 32) {
            if (data->padded_valid >= BENCH_MOE_PADDED_TILES)
                bench_fatal("too many padded MoE tiles");
            data->padded_expert[data->padded_valid] = expert;
            data->padded_m0[data->padded_valid++] = m0;
        }
        for (int m0 = 0; m0 < count; m0 += 64) {
            if (data->padded64_valid >= BENCH_MOE_PADDED_TILES)
                bench_fatal("too many padded64 MoE tiles");
            data->padded64_expert[data->padded64_valid] = expert;
            data->padded64_m0[data->padded64_valid++] = m0;
        }
        int full_rows = count / 32 * 32;
        for (int m0 = 0; m0 < full_rows; m0 += 32) {
            if (data->full_valid >= BENCH_MOE_FULL_TILES)
                bench_fatal("too many full MoE tiles");
            data->full_expert[data->full_valid] = expert;
            data->full_m0[data->full_valid++] = m0;
        }
        if (full_rows < count) {
            if (data->tail_valid >= BENCH_MOE_TAIL_TILES)
                bench_fatal("too many tail MoE tiles");
            data->tail_expert[data->tail_valid] = expert;
            data->tail_m0[data->tail_valid++] = full_rows;
        }
    }
    *data->padded_launch = (ze_group_count_t) {
        (uint32_t)data->padded_valid, BENCH_MOE_N / 32, 1
    };
}

static double bench_moe_launch(bench_gpu *gpu, int variant,
                               const bench_moe_data *data, int repetitions) {
    ze_group_count_t padded_groups = { (uint32_t)data->padded_valid,
                                       BENCH_MOE_N / 32, 1 };
    ze_group_count_t padded_fixed_groups = { BENCH_MOE_PADDED_TILES,
                                             BENCH_MOE_N / 32, 1 };
    ze_group_count_t padded64_groups = { (uint32_t)data->padded64_valid,
                                         BENCH_MOE_N / 32, 1 };
    ze_group_count_t tn64_groups = { (uint32_t)data->padded_valid,
                                     BENCH_MOE_N / 64, 1 };
    ze_group_count_t tm16_tn64_groups = { (uint32_t)data->padded16_valid,
                                          BENCH_MOE_N / 64, 1 };
    ze_group_count_t tm24_tn64_groups = { (uint32_t)data->padded24_valid,
                                          BENCH_MOE_N / 64, 1 };
    ze_group_count_t tn128_groups = { (uint32_t)data->padded_valid,
                                      (BENCH_MOE_N + 127) / 128, 1 };
    ze_group_count_t expert_groups = { BENCH_MOE_EXPERTS,
                                       BENCH_MOE_N / 32, 1 };
    ze_group_count_t full_groups = {
        (uint32_t)(data->full_valid ? data->full_valid : 1),
        BENCH_MOE_N / 32, 1
    };
    ze_group_count_t tail_groups = {
        (uint32_t)(data->tail_valid ? data->tail_valid : 1),
        BENCH_MOE_N / 32, 1
    };
    if (variant == BENCH_MOE_PADDED64) {
        bench_moe_set_kernel(gpu->grouped64, data, data->padded64_expert,
                             data->padded64_m0);
    } else if (variant == BENCH_MOE_TN48) {
        bench_moe_set_kernel(gpu->grouped_tn48, data, data->padded_expert,
                             data->padded_m0);
    } else if (variant == BENCH_MOE_TN64) {
        bench_moe_set_kernel(gpu->grouped_tn64, data, data->padded_expert,
                             data->padded_m0);
    } else if (variant == BENCH_MOE_TN64_DIRECT) {
        bench_moe_set_kernel(gpu->grouped_tn64_direct, data,
                             data->padded_expert, data->padded_m0);
    } else if (variant == BENCH_MOE_TN64_COALESCED) {
        bench_moe_set_kernel(gpu->grouped_tn64_coalesced, data,
                             data->padded_expert, data->padded_m0);
    } else if (variant == BENCH_MOE_TN64_WG256) {
        bench_moe_set_kernel(gpu->grouped_tn64_wg256, data,
                             data->padded_expert, data->padded_m0);
    } else if (variant == BENCH_MOE_TM64_TN64_WG256) {
        bench_moe_set_kernel(gpu->grouped_tm64_tn64_wg256, data,
                             data->padded64_expert, data->padded64_m0);
    } else if (variant == BENCH_MOE_TM24_TN64) {
        bench_moe_set_kernel(gpu->grouped_tm24_tn64, data,
                             data->padded24_expert, data->padded24_m0);
    } else if (variant == BENCH_MOE_TN64_SIGNED) {
        bench_moe_set_kernel(gpu->grouped_tn64_signed, data,
                             data->padded_expert, data->padded_m0);
    } else if (variant == BENCH_MOE_TN64_SLMACC) {
        bench_moe_set_kernel(gpu->grouped_tn64_slmacc, data,
                             data->padded_expert, data->padded_m0);
    } else if (variant == BENCH_MOE_KB128) {
        bench_moe_set_kernel(gpu->grouped_tn64_kb128, data,
                             data->padded_expert, data->padded_m0);
    } else if (variant == BENCH_MOE_TM16_TN64) {
        bench_moe_set_kernel(gpu->grouped_tm16_tn64, data,
                             data->padded16_expert, data->padded16_m0);
    } else if (variant == BENCH_MOE_TN128_WG256) {
        bench_moe_set_kernel(gpu->grouped_tn128_wg256, data,
                             data->padded_expert, data->padded_m0);
    } else if (variant == BENCH_MOE_EXPERT) {
        bench_moe_set_expert_kernel(gpu->grouped_expert, data);
    } else if (variant != BENCH_MOE_HYBRID) {
        bench_moe_set_kernel(gpu->grouped, data, data->padded_expert,
                             data->padded_m0);
    } else {
        bench_moe_set_kernel(gpu->grouped, data, data->full_expert,
                             data->full_m0);
        bench_moe_set_kernel(gpu->tail, data, data->tail_expert, data->tail_m0);
    }
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        if (variant == BENCH_MOE_PADDED || variant == BENCH_MOE_PADDED_FIXED) {
            const ze_group_count_t *groups = variant == BENCH_MOE_PADDED
                                             ? &padded_groups
                                             : &padded_fixed_groups;
            bench_ze_check("zeCommandListAppendLaunchKernel padded MoE",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->grouped,
                                                           groups,
                                                           NULL, 0, NULL));
        } else if (variant == BENCH_MOE_PADDED_INDIRECT) {
            bench_ze_check("zeCommandListAppendLaunchKernelIndirect padded MoE",
                           zeCommandListAppendLaunchKernelIndirect(
                               gpu->commands, gpu->grouped, data->padded_launch,
                               NULL, 0, NULL));
        } else if (variant == BENCH_MOE_PADDED64) {
            bench_ze_check("zeCommandListAppendLaunchKernel padded64 MoE",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->grouped64,
                                                           &padded64_groups,
                                                           NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TN48) {
            ze_group_count_t tn48_groups = {
                (uint32_t)data->padded_valid,
                (uint32_t)((BENCH_MOE_N + 47) / 48), 1
            };
            bench_ze_check("zeCommandListAppendLaunchKernel tn48 MoE",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->grouped_tn48,
                                                           &tn48_groups,
                                                           NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TN64) {
            bench_ze_check("zeCommandListAppendLaunchKernel tn64 MoE",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->grouped_tn64,
                                                           &tn64_groups,
                                                           NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TN64_DIRECT) {
            bench_ze_check("zeCommandListAppendLaunchKernel tn64-direct MoE",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->grouped_tn64_direct,
                               &tn64_groups, NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TN64_COALESCED) {
            bench_ze_check("zeCommandListAppendLaunchKernel tn64-coalesced MoE",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->grouped_tn64_coalesced,
                               &tn64_groups, NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TN64_WG256) {
            bench_ze_check("zeCommandListAppendLaunchKernel tn64-wg256 MoE",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->grouped_tn64_wg256,
                               &tn64_groups, NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TM64_TN64_WG256) {
            ze_group_count_t groups = { (uint32_t)data->padded64_valid,
                                        BENCH_MOE_N / 64, 1 };
            bench_ze_check("zeCommandListAppendLaunchKernel tm64-tn64-wg256 MoE",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands,
                               gpu->grouped_tm64_tn64_wg256,
                               &groups, NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TM24_TN64) {
            bench_ze_check("zeCommandListAppendLaunchKernel tm24-tn64 MoE",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->grouped_tm24_tn64,
                               &tm24_tn64_groups, NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TN64_SIGNED) {
            bench_ze_check("zeCommandListAppendLaunchKernel tn64-signed MoE",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->grouped_tn64_signed,
                               &tn64_groups, NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TN64_SLMACC) {
            bench_ze_check("zeCommandListAppendLaunchKernel tn64-slmacc MoE",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->grouped_tn64_slmacc,
                               &tn64_groups, NULL, 0, NULL));
        } else if (variant == BENCH_MOE_KB128) {
            bench_ze_check("zeCommandListAppendLaunchKernel tn64-kb128 MoE",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->grouped_tn64_kb128,
                               &tn64_groups, NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TM16_TN64) {
            bench_ze_check("zeCommandListAppendLaunchKernel tm16-tn64 MoE",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->grouped_tm16_tn64,
                               &tm16_tn64_groups, NULL, 0, NULL));
        } else if (variant == BENCH_MOE_TN128_WG256) {
            bench_ze_check("zeCommandListAppendLaunchKernel tn128-wg256 MoE",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->grouped_tn128_wg256,
                               &tn128_groups, NULL, 0, NULL));
        } else if (variant == BENCH_MOE_EXPERT) {
            bench_ze_check("zeCommandListAppendLaunchKernel expert MoE",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->grouped_expert,
                                                           &expert_groups,
                                                           NULL, 0, NULL));
        } else {
            bench_ze_check("zeCommandListAppendLaunchKernel full MoE",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->grouped,
                                                           &full_groups,
                                                           NULL, 0, NULL));
            bench_ze_check("zeCommandListAppendLaunchKernel tail MoE",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->tail,
                                                           &tail_groups,
                                                           NULL, 0, NULL));
        }
    }
    bench_ze_check("zeCommandListHostSynchronize MoE",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static bench_sample bench_moe_measure(bench_gpu *gpu, int variant,
                                      const bench_moe_data *data,
                                      int repetitions) {
    bench_telemetry telemetry;
    bench_telemetry_start(&telemetry);
    double seconds = bench_moe_launch(gpu, variant, data, repetitions);
    bench_telemetry_stop(&telemetry);
    return (bench_sample) {
        .seconds = seconds / repetitions,
        .actual_mhz = (double)telemetry.actual_sum / telemetry.samples,
        .requested_mhz = (double)telemetry.requested_sum / telemetry.samples,
        .actual_min = telemetry.actual_min,
        .actual_max = telemetry.actual_max,
        .pl1 = telemetry.pl1,
        .pl2 = telemetry.pl2,
        .thermal = telemetry.thermal
    };
}

static float bench_moe_reference(const bench_moe_data *data, int expert,
                                 int packed_row, int output_column) {
    int group = output_column >> 3;
    int row = output_column & 7;
    float sum = 0.0f;
    for (int block = 0; block < BENCH_MOE_BLOCKS; block++) {
        size_t weight_base = (((size_t)expert * (BENCH_MOE_N / 8) + group)
                              * BENCH_MOE_BLOCKS + block) * 128 + row * 4;
        size_t activation_base =
            ((size_t)packed_row * BENCH_MOE_BLOCKS + block) * 32;
        int integer = 0;
        for (int chunk = 0; chunk < 4; chunk++) {
            for (int byte = 0; byte < 4; byte++) {
                uint8_t packed = data->weight[weight_base + chunk * 32 + byte];
                integer += (packed & 15)
                           * data->activation[activation_base + chunk * 4 + byte];
                integer += (packed >> 4)
                           * data->activation[activation_base + 16
                                              + chunk * 4 + byte];
            }
        }
        size_t activation_scale = (size_t)packed_row * BENCH_MOE_BLOCKS + block;
        size_t weight_scale = (((size_t)expert * (BENCH_MOE_N / 8) + group)
                               * BENCH_MOE_BLOCKS + block) * 8 + row;
        integer -= 8 * data->activation_sigma[activation_scale];
        sum += (float)integer * (float)data->activation_scale[activation_scale]
               * (float)data->weight_scale[weight_scale];
    }
    return sum;
}

static void bench_moe_verify(bench_gpu *gpu, int variant, const char *name,
                             const bench_moe_data *data) {
    bench_moe_launch(gpu, variant, data, 1);
    double error = 0.0;
    double reference = 0.0;
    double max_abs = 0.0;
    for (int probe = 0; probe < 256; probe++) {
        int expert = (probe * 37 + 11) % BENCH_MOE_EXPERTS;
        int local_row = (probe * 17 + 5) % data->expert_count[expert];
        int packed_row = data->token_offset[expert] + local_row;
        int column = (probe * 101 + 17) % BENCH_MOE_N;
        double expected = bench_moe_reference(data, expert, packed_row, column);
        double difference = data->output[(size_t)packed_row * BENCH_MOE_N + column]
                            - expected;
        error += difference * difference;
        reference += expected * expected;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    double rel_rms = sqrt(error / (reference + 1e-30));
    printf("prefill-moe: correctness %s rel-rms %.9g max-abs %.9g probes 256\n",
           name, rel_rms, max_abs);
    if (!isfinite(rel_rms) || rel_rms > 5e-5)
        bench_fatal("MoE %s correctness failed", name);
}

static void bench_moe_distribution(bench_gpu *gpu, bench_moe_data *data,
                                   const char *distribution, int rounds,
                                   double target_seconds) {
    static const char *variant_names[BENCH_MOE_VARIANTS] = {
        "padded", "padded-fixed", "padded-indirect", "padded64", "tn48", "tn64",
        "tn64-wg256", "tm64-tn64-wg256", "tm24-tn64", "tn64-signed",
        "expert-slm", "full-tail"
    };
    bench_moe_schedule(data, distribution);
    int minimum = BENCH_MOE_ROWS;
    int maximum = 0;
    for (int expert = 0; expert < BENCH_MOE_EXPERTS; expert++) {
        if (data->expert_count[expert] < minimum)
            minimum = data->expert_count[expert];
        if (data->expert_count[expert] > maximum)
            maximum = data->expert_count[expert];
    }
    double padding = (double)data->padded_valid * 32 / BENCH_MOE_ROWS;
    printf("prefill-moe: distribution %s count %d..%d padded24/padded32/padded64/full/tail tiles %d/%d/%d/%d/%d padding24 %.6fx padding32 %.6fx padding64 %.6fx\n",
           distribution, minimum, maximum, data->padded24_valid,
           data->padded_valid, data->padded64_valid, data->full_valid,
           data->tail_valid,
           (double)data->padded24_valid * 24 / BENCH_MOE_ROWS, padding,
           (double)data->padded64_valid * 64 / BENCH_MOE_ROWS);
    bench_moe_verify(gpu, BENCH_MOE_PADDED, "padded", data);
    bench_moe_verify(gpu, BENCH_MOE_PADDED_INDIRECT, "padded-indirect", data);
    bench_moe_verify(gpu, BENCH_MOE_PADDED64, "padded64", data);
    bench_moe_verify(gpu, BENCH_MOE_TN48, "tn48", data);
    bench_moe_verify(gpu, BENCH_MOE_TN64, "tn64", data);
    bench_moe_verify(gpu, BENCH_MOE_TN64_WG256, "tn64-wg256", data);
    bench_moe_verify(gpu, BENCH_MOE_TM64_TN64_WG256,
                     "tm64-tn64-wg256", data);
    bench_moe_verify(gpu, BENCH_MOE_TM24_TN64, "tm24-tn64", data);
    bench_moe_verify(gpu, BENCH_MOE_TN64_SIGNED, "tn64-signed", data);
    bench_moe_verify(gpu, BENCH_MOE_EXPERT, "expert-slm", data);
    bench_moe_verify(gpu, BENCH_MOE_HYBRID, "full-tail", data);
    double probe = bench_moe_launch(gpu, BENCH_MOE_PADDED, data, 2) / 2.0;
    int repetitions = (int)ceil(target_seconds / probe);
    if (repetitions < 3) repetitions = 3;
    printf("prefill-moe: repetitions %d target %.3f s/variant\n",
           repetitions, target_seconds);

    bench_sample samples[BENCH_MOE_VARIANTS][BENCH_ROUNDS_MAX];
    double times[BENCH_MOE_VARIANTS][BENCH_ROUNDS_MAX];
    double frequencies[BENCH_MOE_VARIANTS][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        for (int position = 0; position < BENCH_MOE_VARIANTS; position++) {
            int variant = (round + position) % BENCH_MOE_VARIANTS;
            samples[variant][round] =
                bench_moe_measure(gpu, variant, data, repetitions);
            times[variant][round] = samples[variant][round].seconds;
            frequencies[variant][round] = samples[variant][round].actual_mhz;
        }
        printf("prefill-moe: round %d", round + 1);
        for (int variant = 0; variant < BENCH_MOE_VARIANTS; variant++) {
            bench_sample *sample = &samples[variant][round];
            printf(" %s %.6f ms@%.0fMHz[%ld,%ld] throttle=%ld/%ld/%ld",
                   variant_names[variant], sample->seconds * 1e3,
                   sample->actual_mhz, sample->actual_min, sample->actual_max,
                   sample->pl1, sample->pl2, sample->thermal);
        }
        printf("\n");
    }

    double padded_time = bench_median(times[BENCH_MOE_PADDED], rounds);
    double fixed_time = bench_median(times[BENCH_MOE_PADDED_FIXED], rounds);
    double indirect_time = bench_median(times[BENCH_MOE_PADDED_INDIRECT], rounds);
    double padded64_time = bench_median(times[BENCH_MOE_PADDED64], rounds);
    double tn48_time = bench_median(times[BENCH_MOE_TN48], rounds);
    double tn64_time = bench_median(times[BENCH_MOE_TN64], rounds);
    double tn64_wg256_time =
        bench_median(times[BENCH_MOE_TN64_WG256], rounds);
    double tm64_tn64_wg256_time =
        bench_median(times[BENCH_MOE_TM64_TN64_WG256], rounds);
    double tm24_tn64_time =
        bench_median(times[BENCH_MOE_TM24_TN64], rounds);
    double tn64_signed_time =
        bench_median(times[BENCH_MOE_TN64_SIGNED], rounds);
    double expert_time = bench_median(times[BENCH_MOE_EXPERT], rounds);
    double hybrid_time = bench_median(times[BENCH_MOE_HYBRID], rounds);
    double padded_frequency = bench_median(frequencies[BENCH_MOE_PADDED], rounds);
    double fixed_frequency =
        bench_median(frequencies[BENCH_MOE_PADDED_FIXED], rounds);
    double indirect_frequency =
        bench_median(frequencies[BENCH_MOE_PADDED_INDIRECT], rounds);
    double padded64_frequency =
        bench_median(frequencies[BENCH_MOE_PADDED64], rounds);
    double tn48_frequency = bench_median(frequencies[BENCH_MOE_TN48], rounds);
    double tn64_frequency = bench_median(frequencies[BENCH_MOE_TN64], rounds);
    double tn64_wg256_frequency =
        bench_median(frequencies[BENCH_MOE_TN64_WG256], rounds);
    double tm64_tn64_wg256_frequency =
        bench_median(frequencies[BENCH_MOE_TM64_TN64_WG256], rounds);
    double tm24_tn64_frequency =
        bench_median(frequencies[BENCH_MOE_TM24_TN64], rounds);
    double tn64_signed_frequency =
        bench_median(frequencies[BENCH_MOE_TN64_SIGNED], rounds);
    double expert_frequency =
        bench_median(frequencies[BENCH_MOE_EXPERT], rounds);
    double hybrid_frequency = bench_median(frequencies[BENCH_MOE_HYBRID], rounds);
    double useful_operations = 2.0 * BENCH_MOE_ROWS * BENCH_MOE_N
                               * BENCH_MOE_BLOCKS * 32;
    double padded_operations = useful_operations * padding;
    double padded_useful = useful_operations / padded_time / 1e12;
    double padded_executed = padded_operations / padded_time / 1e12;
    double hybrid_useful = useful_operations / hybrid_time / 1e12;
    double recovery = hybrid_useful / padded_executed;
    double speedup = padded_time / hybrid_time;
    double fixed_cost = fixed_time / padded_time;
    double indirect_cost = indirect_time / padded_time;
    double padded64_cost = padded64_time / padded_time;
    double tn48_cost = tn48_time / padded_time;
    double tn64_cost = tn64_time / padded_time;
    double tn64_wg256_cost = tn64_wg256_time / padded_time;
    double tm64_tn64_wg256_cost = tm64_tn64_wg256_time / padded_time;
    double tm24_tn64_cost = tm24_tn64_time / padded_time;
    double tn64_signed_cost = tn64_signed_time / padded_time;
    double expert_cost = expert_time / padded_time;
    double frequency_max = fmax(fmax(padded_frequency, fixed_frequency),
                                fmax(fmax(indirect_frequency,
                                          fmax(padded64_frequency,
                                               tn48_frequency)),
                                     fmax(fmax(tn64_frequency,
                                               tn64_wg256_frequency),
                                          fmax(tm64_tn64_wg256_frequency,
                                               fmax(tm24_tn64_frequency,
                                                    fmax(tn64_signed_frequency,
                                                         fmax(expert_frequency,
                                                              hybrid_frequency)))))));
    double frequency_min = fmin(fmin(padded_frequency, fixed_frequency),
                                fmin(fmin(indirect_frequency,
                                          fmin(padded64_frequency,
                                               tn48_frequency)),
                                     fmin(fmin(tn64_frequency,
                                               tn64_wg256_frequency),
                                          fmin(tm64_tn64_wg256_frequency,
                                               fmin(tm24_tn64_frequency,
                                                    fmin(tn64_signed_frequency,
                                                         fmin(expert_frequency,
                                                              hybrid_frequency)))))));
    double frequency_span = frequency_max / frequency_min;
    printf("prefill-moe: median padded %.6f ms useful %.6f TFLOP/s executed %.6f TFLOP/s @%.0fMHz padded-fixed %.6f ms @%.0fMHz padded-indirect %.6f ms @%.0fMHz padded64 %.6f ms @%.0fMHz tn48 %.6f ms @%.0fMHz tn64 %.6f ms @%.0fMHz tn64-wg256 %.6f ms @%.0fMHz tm64-tn64-wg256 %.6f ms @%.0fMHz tm24-tn64 %.6f ms @%.0fMHz tn64-signed %.6f ms @%.0fMHz expert-slm %.6f ms @%.0fMHz full-tail %.6f ms useful %.6f TFLOP/s @%.0fMHz speedup %.6fx recovery %.6fx fixed-cost %.6fx indirect-cost %.6fx padded64-cost %.6fx tn48-cost %.6fx tn64-cost %.6fx tn64-wg256-cost %.6fx tm64-tn64-wg256-cost %.6fx tm24-tn64-cost %.6fx tn64-signed-cost %.6fx expert-cost %.6fx frequency-span %.6fx\n",
           padded_time * 1e3, padded_useful, padded_executed, padded_frequency,
           fixed_time * 1e3, fixed_frequency, indirect_time * 1e3,
           indirect_frequency, padded64_time * 1e3, padded64_frequency,
           tn48_time * 1e3, tn48_frequency,
           tn64_time * 1e3, tn64_frequency,
           tn64_wg256_time * 1e3, tn64_wg256_frequency,
           tm64_tn64_wg256_time * 1e3, tm64_tn64_wg256_frequency,
           tm24_tn64_time * 1e3, tm24_tn64_frequency,
           tn64_signed_time * 1e3, tn64_signed_frequency,
           expert_time * 1e3, expert_frequency,
           hybrid_time * 1e3, hybrid_useful,
           hybrid_frequency, speedup, recovery, fixed_cost, indirect_cost,
           padded64_cost, tn48_cost, tn64_cost, tn64_wg256_cost,
           tm64_tn64_wg256_cost, tm24_tn64_cost, tn64_signed_cost,
           expert_cost, frequency_span);
    const char *scheduling = indirect_cost <= fixed_cost && indirect_cost <= 1.05
                             ? "indirect-go"
                             : fixed_cost <= 1.05 ? "fixed-go" : "no-go";
    const char *kernel_choice = tn64_signed_cost <= tn64_cost * 0.90
                                ? "tn64-signed-go"
                                : tm24_tn64_cost <= tn64_cost * 0.90
                                ? "tm24-tn64-go"
                                : tm64_tn64_wg256_cost <= tn64_cost * 0.90
                                ? "tm64-tn64-wg256-go"
                                : tn64_wg256_cost <= fmin(tn48_cost, tn64_cost) * 0.97
                                ? "tn64-wg256-go"
                                : tn48_cost <= tn64_cost * 0.97
                                  && tn48_cost <= 0.97 ? "tn48-go"
                                : tn64_cost <= 0.97 ? "tn64-go"
                                : expert_cost <= 0.90 ? "expert-slm-go"
                                : padded64_cost <= 0.90 ? "padded64-go"
                                : recovery >= 0.85 && speedup >= 1.0
                                  ? "full-tail-go" : "padded32-go";
    printf("prefill-moe: decision kernel %s scheduling %s frequency %s\n",
           kernel_choice,
           scheduling,
           frequency_span <= 1.05 && padded_frequency >= 1000.0
           && fixed_frequency >= 1000.0 && indirect_frequency >= 1000.0
           && padded64_frequency >= 1000.0
           && tn48_frequency >= 1000.0
           && tn64_frequency >= 1000.0
           && tn64_wg256_frequency >= 1000.0
           && tm64_tn64_wg256_frequency >= 1000.0
           && tm24_tn64_frequency >= 1000.0
           && tn64_signed_frequency >= 1000.0
           && expert_frequency >= 1000.0
           && hybrid_frequency >= 1000.0
           ? "comparable" : "reject");
}

static void bench_moe_run(bench_gpu *gpu, const char *selected_distribution,
                          int rounds, double target_seconds) {
    static const char *distributions[] = { "uniform", "multinomial", "adversarial" };
    bench_moe_data data;
    bench_moe_data_init(gpu, &data);
    int matched = 0;
    for (size_t i = 0; i < sizeof distributions / sizeof distributions[0]; i++) {
        if (selected_distribution
            && strcmp(selected_distribution, distributions[i])) continue;
        bench_moe_distribution(gpu, &data, distributions[i], rounds,
                               target_seconds);
        matched = 1;
    }
    if (!matched) bench_fatal("unknown MoE distribution: %s",
                              selected_distribution);
    bench_moe_data_destroy(gpu, &data);
}

static void bench_moe_pair_run(bench_gpu *gpu, int candidate,
                               const char *label, int rounds,
                               double target_seconds) {
    bench_moe_data data;
    bench_moe_data_init(gpu, &data);
    bench_moe_schedule(&data, "multinomial");
    bench_moe_verify(gpu, BENCH_MOE_TN64, "tn64", &data);
    bench_moe_verify(gpu, candidate, label, &data);
    double probe = bench_moe_launch(gpu, BENCH_MOE_TN64, &data, 2) / 2.0;
    int repetitions = (int)ceil(target_seconds / probe);
    if (repetitions < 3) repetitions = 3;
    double times[2][BENCH_ROUNDS_MAX];
    double frequencies[2][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        printf("prefill-moe-pair: round %d", round + 1);
        for (int position = 0; position < 2; position++) {
            int variant = (round + position) & 1;
            int benchmark_variant = variant ? candidate : BENCH_MOE_TN64;
            bench_sample sample = bench_moe_measure(
                gpu, benchmark_variant, &data, repetitions);
            times[variant][round] = sample.seconds;
            frequencies[variant][round] = sample.actual_mhz;
            printf(" %s %.6f ms@%.0fMHz[%ld,%ld] throttle=%ld/%ld/%ld",
                   variant ? label : "tn64", sample.seconds * 1e3,
                   sample.actual_mhz, sample.actual_min, sample.actual_max,
                   sample.pl1, sample.pl2, sample.thermal);
        }
        printf("\n");
    }
    double unsigned_time = bench_median(times[0], rounds);
    double signed_time = bench_median(times[1], rounds);
    double unsigned_frequency = bench_median(frequencies[0], rounds);
    double signed_frequency = bench_median(frequencies[1], rounds);
    double frequency_span = fmax(unsigned_frequency, signed_frequency)
                            / fmin(unsigned_frequency, signed_frequency);
    printf("prefill-moe-pair: median tn64 %.6f ms @%.0fMHz %s %.6f ms @%.0fMHz speedup %.6fx frequency-span %.6fx decision %s\n",
           unsigned_time * 1e3, unsigned_frequency, label, signed_time * 1e3,
           signed_frequency, unsigned_time / signed_time, frequency_span,
           frequency_span > 1.05 ? "frequency-reject"
           : signed_time <= unsigned_time * 0.90 ? "candidate-go"
           : "tn64-go");
    bench_moe_data_destroy(gpu, &data);
}

static void bench_moe_tile_run(bench_gpu *gpu, const char *distribution,
                               int rounds, double target_seconds) {
    bench_moe_data data;
    bench_moe_data_init(gpu, &data);
    bench_moe_schedule(&data, distribution ? distribution : "multinomial");
    bench_moe_verify(gpu, BENCH_MOE_PADDED, "tn32-control", &data);
    bench_moe_verify(gpu, BENCH_MOE_TN48, "tn48", &data);
    bench_moe_verify(gpu, BENCH_MOE_TN64, "tn64", &data);
    bench_moe_verify(gpu, BENCH_MOE_TN64_WG256, "tn64-wg256", &data);
    bench_moe_verify(gpu, BENCH_MOE_TM16_TN64, "tm16-tn64", &data);
    bench_moe_verify(gpu, BENCH_MOE_TN128_WG256, "tn128-wg256", &data);
    double probe32 = bench_moe_launch(gpu, BENCH_MOE_PADDED, &data, 3) / 3.0;
    double probe48 = bench_moe_launch(gpu, BENCH_MOE_TN48, &data, 3) / 3.0;
    double probe64 = bench_moe_launch(gpu, BENCH_MOE_TN64, &data, 3) / 3.0;
    double probe256 = bench_moe_launch(
        gpu, BENCH_MOE_TN64_WG256, &data, 3) / 3.0;
    double probe16 = bench_moe_launch(
        gpu, BENCH_MOE_TM16_TN64, &data, 3) / 3.0;
    double probe128 = bench_moe_launch(
        gpu, BENCH_MOE_TN128_WG256, &data, 3) / 3.0;
    int repetitions[6] = {
        (int)ceil(target_seconds / probe32),
        (int)ceil(target_seconds / probe48),
        (int)ceil(target_seconds / probe64),
        (int)ceil(target_seconds / probe256),
        (int)ceil(target_seconds / probe16),
        (int)ceil(target_seconds / probe128)
    };
    for (int variant = 0; variant < 6; variant++)
        if (repetitions[variant] < 3) repetitions[variant] = 3;
    printf("prefill-moe-tile: padding tm32/tm16 %.6fx/%.6fx repetitions tn32/tn48/tn64/tn64-wg256/tm16-tn64/tn128-wg256 %d/%d/%d/%d/%d/%d target %.3f s\n",
           (double)data.padded_valid * 32 / BENCH_MOE_ROWS,
           (double)data.padded16_valid * 16 / BENCH_MOE_ROWS,
           repetitions[0], repetitions[1], repetitions[2], repetitions[3],
           repetitions[4], repetitions[5], target_seconds);
    bench_sample samples[6][BENCH_ROUNDS_MAX];
    double times[6][BENCH_ROUNDS_MAX];
    double frequencies[6][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        for (int position = 0; position < 6; position++) {
            int variant = (round + position) % 6;
            int kernel_variant = variant == 0 ? BENCH_MOE_PADDED
                                 : variant == 1 ? BENCH_MOE_TN48
                                 : variant == 2 ? BENCH_MOE_TN64
                                 : variant == 3 ? BENCH_MOE_TN64_WG256
                                 : variant == 4 ? BENCH_MOE_TM16_TN64
                                 : BENCH_MOE_TN128_WG256;
            samples[variant][round] = bench_moe_measure(
                gpu, kernel_variant, &data, repetitions[variant]);
            times[variant][round] = samples[variant][round].seconds;
            frequencies[variant][round] = samples[variant][round].actual_mhz;
        }
        printf("prefill-moe-tile: round %d tn32 %.6f ms@%.0fMHz tn48 %.6f ms@%.0fMHz tn64 %.6f ms@%.0fMHz tn64-wg256 %.6f ms@%.0fMHz tm16-tn64 %.6f ms@%.0fMHz tn128-wg256 %.6f ms@%.0fMHz\n",
               round + 1, samples[0][round].seconds * 1e3,
               samples[0][round].actual_mhz, samples[1][round].seconds * 1e3,
               samples[1][round].actual_mhz, samples[2][round].seconds * 1e3,
               samples[2][round].actual_mhz, samples[3][round].seconds * 1e3,
               samples[3][round].actual_mhz, samples[4][round].seconds * 1e3,
               samples[4][round].actual_mhz, samples[5][round].seconds * 1e3,
               samples[5][round].actual_mhz);
    }
    double tn32 = bench_median(times[0], rounds);
    double tn48 = bench_median(times[1], rounds);
    double tn64 = bench_median(times[2], rounds);
    double tn64_wg256 = bench_median(times[3], rounds);
    double tm16_tn64 = bench_median(times[4], rounds);
    double tn128_wg256 = bench_median(times[5], rounds);
    double frequency32 = bench_median(frequencies[0], rounds);
    double frequency48 = bench_median(frequencies[1], rounds);
    double frequency64 = bench_median(frequencies[2], rounds);
    double frequency256 = bench_median(frequencies[3], rounds);
    double frequency16 = bench_median(frequencies[4], rounds);
    double frequency128 = bench_median(frequencies[5], rounds);
    double frequency_span = fmax(fmax(frequency32, frequency48),
                                 fmax(frequency64,
                                      fmax(frequency256,
                                           fmax(frequency16, frequency128))))
                            / fmin(fmin(frequency32, frequency48),
                                   fmin(frequency64,
                                        fmin(frequency256,
                                             fmin(frequency16, frequency128))));
    printf("prefill-moe-tile: median tn32 %.6f ms @%.0fMHz tn48 %.6f ms @%.0fMHz tn64 %.6f ms @%.0fMHz tn64-wg256 %.6f ms @%.0fMHz tm16-tn64 %.6f ms @%.0fMHz tn128-wg256 %.6f ms @%.0fMHz speedup-64-vs-128 %.6fx frequency-span %.6fx decision %s\n",
           tn32 * 1e3, frequency32, tn48 * 1e3, frequency48,
           tn64 * 1e3, frequency64, tn64_wg256 * 1e3, frequency256,
           tm16_tn64 * 1e3, frequency16,
           tn128_wg256 * 1e3, frequency128, tn64 / tn128_wg256,
           frequency_span,
           frequency_span > 1.05 ? "frequency-reject"
           : tn128_wg256 <= tn64 * 0.97 ? "tn128-wg256-go"
           : tm16_tn64 <= tn64 * 0.97 ? "tm16-tn64-go"
           : tn64_wg256 <= fmin(tn48, tn64) * 0.97 ? "tn64-wg256-go"
           : tn48 <= tn64 * 0.97 && tn48 <= tn32 * 0.97 ? "tn48-go"
           : tn64 <= tn32 * 0.97 ? "tn64-go" : "tn32-go");
    bench_moe_data_destroy(gpu, &data);
}

static void bench_route_set_args(bench_gpu *gpu, const bench_moe_data *data) {
    bench_pointer_arg(gpu->route_reset, 0, data->expert_count);
    bench_pointer_arg(gpu->route_reset, 1, data->route_cursor);
    bench_pointer_arg(gpu->route_reset, 2, data->padded_expert);
    bench_pointer_arg(gpu->route_reset, 3, data->padded_m0);

    bench_pointer_arg(gpu->route_count, 0, data->route_expert);
    bench_pointer_arg(gpu->route_count, 1, data->expert_count);
    bench_int_arg(gpu->route_count, 2, BENCH_MOE_ROWS);

    bench_pointer_arg(gpu->route_prefix, 0, data->expert_count);
    bench_pointer_arg(gpu->route_prefix, 1, data->token_offset);
    bench_pointer_arg(gpu->route_prefix, 2, data->route_cursor);
    bench_pointer_arg(gpu->route_prefix, 3, data->padded_expert);
    bench_pointer_arg(gpu->route_prefix, 4, data->padded_m0);

    bench_pointer_arg(gpu->route_scatter, 0, data->route_expert);
    bench_pointer_arg(gpu->route_scatter, 1, data->route_cursor);
    bench_pointer_arg(gpu->route_scatter, 2, data->packed_route);
    bench_pointer_arg(gpu->route_scatter, 3, data->route_packed);
    bench_int_arg(gpu->route_scatter, 4, BENCH_MOE_ROWS);

    bench_pointer_arg(gpu->route_pack, 0, data->source_activation);
    bench_pointer_arg(gpu->route_pack, 1, data->source_scale);
    bench_pointer_arg(gpu->route_pack, 2, data->source_sigma);
    bench_pointer_arg(gpu->route_pack, 3, data->activation);
    bench_pointer_arg(gpu->route_pack, 4, data->activation_scale);
    bench_pointer_arg(gpu->route_pack, 5, data->activation_sigma);
    bench_pointer_arg(gpu->route_pack, 6, data->route_token);
    bench_pointer_arg(gpu->route_pack, 7, data->packed_route);
    bench_int_arg(gpu->route_pack, 8, BENCH_MOE_BLOCKS);
    bench_int_arg(gpu->route_pack, 9, BENCH_MOE_ROWS);

    bench_pointer_arg(gpu->grouped_gather, 0, data->weight);
    bench_pointer_arg(gpu->grouped_gather, 1, data->weight_scale);
    bench_pointer_arg(gpu->grouped_gather, 2, data->source_activation);
    bench_pointer_arg(gpu->grouped_gather, 3, data->source_scale);
    bench_pointer_arg(gpu->grouped_gather, 4, data->source_sigma);
    bench_pointer_arg(gpu->grouped_gather, 5, data->output);
    bench_pointer_arg(gpu->grouped_gather, 6, data->expert_count);
    bench_pointer_arg(gpu->grouped_gather, 7, data->token_offset);
    bench_pointer_arg(gpu->grouped_gather, 8, data->padded_expert);
    bench_pointer_arg(gpu->grouped_gather, 9, data->padded_m0);
    bench_pointer_arg(gpu->grouped_gather, 10, data->route_token);
    bench_pointer_arg(gpu->grouped_gather, 11, data->packed_route);
    bench_int_arg(gpu->grouped_gather, 12, BENCH_MOE_N);
    bench_int_arg(gpu->grouped_gather, 13, BENCH_MOE_BLOCKS);

    bench_moe_set_kernel(gpu->grouped, data, data->padded_expert,
                         data->padded_m0);
}

static double bench_route_launch(bench_gpu *gpu, int variant,
                                 int repetitions) {
    ze_group_count_t reset_groups = { 2, 1, 1 };
    ze_group_count_t route_groups = { BENCH_MOE_ROWS / 128, 1, 1 };
    ze_group_count_t prefix_groups = { 1, 1, 1 };
    ze_group_count_t pack_groups = { BENCH_MOE_ROWS, 1, 1 };
    ze_group_count_t gemm_groups = { BENCH_MOE_PADDED_TILES,
                                     BENCH_MOE_N / 32, 1 };
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        if (variant != BENCH_ROUTE_BASELINE) {
            bench_ze_check("zeCommandListAppendLaunchKernel route reset",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->route_reset,
                                                           &reset_groups,
                                                           NULL, 0, NULL));
            bench_ze_check("zeCommandListAppendLaunchKernel route count",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->route_count,
                                                           &route_groups,
                                                           NULL, 0, NULL));
            bench_ze_check("zeCommandListAppendLaunchKernel route prefix",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->route_prefix,
                                                           &prefix_groups,
                                                           NULL, 0, NULL));
            bench_ze_check("zeCommandListAppendLaunchKernel route scatter",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->route_scatter,
                                                           &route_groups,
                                                           NULL, 0, NULL));
        }
        if (variant == BENCH_ROUTE_PACK || variant == BENCH_ROUTE_PACK_GEMM)
            bench_ze_check("zeCommandListAppendLaunchKernel route pack",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->route_pack,
                                                           &pack_groups,
                                                           NULL, 0, NULL));
        if (variant == BENCH_ROUTE_BASELINE || variant == BENCH_ROUTE_PACK_GEMM)
            bench_ze_check("zeCommandListAppendLaunchKernel route grouped",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->grouped,
                                                           &gemm_groups,
                                                           NULL, 0, NULL));
        if (variant == BENCH_ROUTE_GATHER_GEMM)
            bench_ze_check("zeCommandListAppendLaunchKernel route gather",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->grouped_gather,
                                                           &gemm_groups,
                                                           NULL, 0, NULL));
    }
    bench_ze_check("zeCommandListHostSynchronize route",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static float bench_route_reference(const bench_moe_data *data, int expert,
                                   int packed_row, int output_column) {
    int token = data->route_token[data->packed_route[packed_row]];
    int group = output_column >> 3;
    int row = output_column & 7;
    float sum = 0.0f;
    for (int block = 0; block < BENCH_MOE_BLOCKS; block++) {
        size_t weight_base = (((size_t)expert * (BENCH_MOE_N / 8) + group)
                              * BENCH_MOE_BLOCKS + block) * 128 + row * 4;
        size_t activation_base =
            ((size_t)token * BENCH_MOE_BLOCKS + block) * 32;
        int integer = 0;
        for (int chunk = 0; chunk < 4; chunk++) {
            for (int byte = 0; byte < 4; byte++) {
                uint8_t packed = data->weight[weight_base + chunk * 32 + byte];
                integer += (packed & 15)
                           * data->source_activation[activation_base
                                                     + chunk * 4 + byte];
                integer += (packed >> 4)
                           * data->source_activation[activation_base + 16
                                                     + chunk * 4 + byte];
            }
        }
        size_t activation_scale = (size_t)token * BENCH_MOE_BLOCKS + block;
        size_t weight_scale = (((size_t)expert * (BENCH_MOE_N / 8) + group)
                               * BENCH_MOE_BLOCKS + block) * 8 + row;
        integer -= 8 * data->source_sigma[activation_scale];
        sum += (float)integer * (float)data->source_scale[activation_scale]
               * (float)data->weight_scale[weight_scale];
    }
    return sum;
}

static void bench_route_verify(bench_gpu *gpu, bench_moe_data *data,
                               const int *expected_count, int variant,
                               const char *name) {
    bench_route_launch(gpu, variant, 1);
    int seen[BENCH_MOE_ROWS] = {0};
    for (int expert = 0; expert < BENCH_MOE_EXPERTS; expert++) {
        if (data->expert_count[expert] != expected_count[expert])
            bench_fatal("route count mismatch for expert %d", expert);
        if (data->token_offset[expert + 1] - data->token_offset[expert]
            != expected_count[expert])
            bench_fatal("route offset mismatch for expert %d", expert);
        for (int packed = data->token_offset[expert];
             packed < data->token_offset[expert + 1]; packed++) {
            int route = data->packed_route[packed];
            if (route < 0 || route >= BENCH_MOE_ROWS || seen[route]++)
                bench_fatal("route permutation invalid at %d", packed);
            if (data->route_expert[route] != expert)
                bench_fatal("route expert mismatch at %d", packed);
        }
    }
    if (variant == BENCH_ROUTE_PACK_GEMM) {
        for (int probe = 0; probe < 256; probe++) {
            int packed = (probe * 37 + 11) % BENCH_MOE_ROWS;
            int token = data->route_token[data->packed_route[packed]];
            int k = (probe * 101 + 17) % (BENCH_MOE_BLOCKS * 32);
            if (data->activation[(size_t)packed * BENCH_MOE_BLOCKS * 32 + k]
                != data->source_activation[(size_t)token * BENCH_MOE_BLOCKS * 32
                                            + k])
                bench_fatal("route pack mismatch at %d", packed);
        }
    }
    double error = 0.0;
    double reference = 0.0;
    double max_abs = 0.0;
    for (int probe = 0; probe < 256; probe++) {
        int expert = (probe * 37 + 11) % BENCH_MOE_EXPERTS;
        int local_row = (probe * 17 + 5) % data->expert_count[expert];
        int packed = data->token_offset[expert] + local_row;
        int column = (probe * 101 + 17) % BENCH_MOE_N;
        double expected = bench_route_reference(data, expert, packed, column);
        double difference = data->output[(size_t)packed * BENCH_MOE_N + column]
                            - expected;
        error += difference * difference;
        reference += expected * expected;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    double rel_rms = sqrt(error / (reference + 1e-30));
    printf("prefill-route: correctness %s rel-rms %.9g max-abs %.9g\n",
           name, rel_rms, max_abs);
    if (!isfinite(rel_rms) || rel_rms > 5e-5)
        bench_fatal("route %s correctness failed", name);
}

static bench_sample bench_route_measure(bench_gpu *gpu, int variant,
                                        int repetitions) {
    bench_telemetry telemetry;
    bench_telemetry_start(&telemetry);
    double seconds = bench_route_launch(gpu, variant, repetitions);
    bench_telemetry_stop(&telemetry);
    return (bench_sample) {
        .seconds = seconds / repetitions,
        .actual_mhz = (double)telemetry.actual_sum / telemetry.samples,
        .requested_mhz = (double)telemetry.requested_sum / telemetry.samples,
        .actual_min = telemetry.actual_min,
        .actual_max = telemetry.actual_max,
        .pl1 = telemetry.pl1,
        .pl2 = telemetry.pl2,
        .thermal = telemetry.thermal
    };
}

static void bench_route_run(bench_gpu *gpu, const char *distribution,
                            int rounds, double target_seconds) {
    static const char *names[BENCH_ROUTE_VARIANTS] = {
        "metadata", "metadata-pack", "grouped-baseline", "pack-grouped",
        "gather-grouped"
    };
    bench_moe_data data;
    bench_moe_data_init(gpu, &data);
    bench_moe_schedule(&data, distribution ? distribution : "multinomial");
    int expected_count[BENCH_MOE_EXPERTS];
    memcpy(expected_count, data.expert_count, sizeof expected_count);
    bench_route_set_args(gpu, &data);
    bench_route_verify(gpu, &data, expected_count, BENCH_ROUTE_PACK_GEMM,
                       "pack-grouped");
    bench_route_verify(gpu, &data, expected_count, BENCH_ROUTE_GATHER_GEMM,
                       "gather-grouped");

    int repetitions[BENCH_ROUTE_VARIANTS];
    for (int variant = 0; variant < BENCH_ROUTE_VARIANTS; variant++) {
        double probe = bench_route_launch(gpu, variant, 3) / 3.0;
        repetitions[variant] = (int)ceil(target_seconds / probe);
        if (repetitions[variant] < 3) repetitions[variant] = 3;
        if (repetitions[variant] > 2000) repetitions[variant] = 2000;
    }
    printf("prefill-route: repetitions metadata/pack/baseline/pack-gemm/gather-gemm %d/%d/%d/%d/%d target %.3f s\n",
           repetitions[0], repetitions[1], repetitions[2], repetitions[3],
           repetitions[4], target_seconds);

    bench_sample samples[BENCH_ROUTE_VARIANTS][BENCH_ROUNDS_MAX];
    double times[BENCH_ROUTE_VARIANTS][BENCH_ROUNDS_MAX];
    double frequencies[BENCH_ROUTE_VARIANTS][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        for (int position = 0; position < BENCH_ROUTE_VARIANTS; position++) {
            int variant = (round + position) % BENCH_ROUTE_VARIANTS;
            samples[variant][round] = bench_route_measure(
                gpu, variant, repetitions[variant]);
            times[variant][round] = samples[variant][round].seconds;
            frequencies[variant][round] = samples[variant][round].actual_mhz;
        }
        printf("prefill-route: round %d", round + 1);
        for (int variant = 0; variant < BENCH_ROUTE_VARIANTS; variant++)
            printf(" %s %.6f ms@%.0fMHz", names[variant],
                   samples[variant][round].seconds * 1e3,
                   samples[variant][round].actual_mhz);
        printf("\n");
    }
    double medians[BENCH_ROUTE_VARIANTS];
    double median_frequencies[BENCH_ROUTE_VARIANTS];
    for (int variant = 0; variant < BENCH_ROUTE_VARIANTS; variant++) {
        medians[variant] = bench_median(times[variant], rounds);
        median_frequencies[variant] = bench_median(frequencies[variant], rounds);
    }
    double pack_copy = medians[BENCH_ROUTE_PACK] - medians[BENCH_ROUTE_METADATA];
    double pack_tax = medians[BENCH_ROUTE_PACK_GEMM]
                      - medians[BENCH_ROUTE_BASELINE];
    double gather_cost = medians[BENCH_ROUTE_GATHER_GEMM]
                         / medians[BENCH_ROUTE_PACK_GEMM];
    double complete_frequency_span =
        fmax(median_frequencies[BENCH_ROUTE_PACK_GEMM],
             median_frequencies[BENCH_ROUTE_GATHER_GEMM])
        / fmin(median_frequencies[BENCH_ROUTE_PACK_GEMM],
               median_frequencies[BENCH_ROUTE_GATHER_GEMM]);
    printf("prefill-route: median metadata %.6f ms pack-total %.6f ms pack-copy %.6f ms grouped-baseline %.6f ms pack-grouped %.6f ms pack-tax %.6f ms gather-grouped %.6f ms gather-cost %.6fx frequency-span %.6fx decision %s\n",
           medians[BENCH_ROUTE_METADATA] * 1e3,
           medians[BENCH_ROUTE_PACK] * 1e3, pack_copy * 1e3,
           medians[BENCH_ROUTE_BASELINE] * 1e3,
           medians[BENCH_ROUTE_PACK_GEMM] * 1e3, pack_tax * 1e3,
           medians[BENCH_ROUTE_GATHER_GEMM] * 1e3, gather_cost,
           complete_frequency_span,
           complete_frequency_span > 1.05 ? "frequency-reject"
           : gather_cost <= 1.03 ? "gather-go" : "pack-go");
    bench_moe_data_destroy(gpu, &data);
}

static double bench_reduce_launch(bench_gpu *gpu, int repetitions) {
    ze_group_count_t groups = { BENCH_MOE_TOKENS, 1, 1 };
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        bench_ze_check("zeCommandListAppendLaunchKernel route reduce",
                       zeCommandListAppendLaunchKernel(gpu->commands,
                                                       gpu->route_reduce,
                                                       &groups,
                                                       NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize route reduce",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static bench_sample bench_reduce_measure(bench_gpu *gpu, int repetitions) {
    bench_telemetry telemetry;
    bench_telemetry_start(&telemetry);
    double seconds = bench_reduce_launch(gpu, repetitions);
    bench_telemetry_stop(&telemetry);
    return (bench_sample) {
        .seconds = seconds / repetitions,
        .actual_mhz = (double)telemetry.actual_sum / telemetry.samples,
        .requested_mhz = (double)telemetry.requested_sum / telemetry.samples,
        .actual_min = telemetry.actual_min,
        .actual_max = telemetry.actual_max,
        .pl1 = telemetry.pl1,
        .pl2 = telemetry.pl2,
        .thermal = telemetry.thermal
    };
}

static void bench_reduce_run(bench_gpu *gpu, const char *distribution,
                             int rounds, double target_seconds) {
    bench_moe_data data;
    bench_moe_data_init(gpu, &data);
    bench_moe_schedule(&data, distribution ? distribution : "multinomial");
    bench_moe_launch(gpu, BENCH_MOE_PADDED, &data, 1);
    bench_pointer_arg(gpu->route_reduce, 0, data.output);
    bench_pointer_arg(gpu->route_reduce, 1, data.route_weight);
    bench_pointer_arg(gpu->route_reduce, 2, data.route_packed);
    bench_pointer_arg(gpu->route_reduce, 3, data.reduced);
    bench_int_arg(gpu->route_reduce, 4, BENCH_MOE_N);
    bench_reduce_launch(gpu, 1);
    double error = 0.0;
    double reference = 0.0;
    double max_abs = 0.0;
    for (int probe = 0; probe < 256; probe++) {
        int token = (probe * 37 + 11) % BENCH_MOE_TOKENS;
        int column = (probe * 101 + 17) % BENCH_MOE_N;
        float expected = 0.0f;
        for (int slot = 0; slot < 8; slot++) {
            int route = token * 8 + slot;
            int packed = data.route_packed[route];
            expected += data.route_weight[route]
                        * data.output[(size_t)packed * BENCH_MOE_N + column];
        }
        double difference =
            data.reduced[(size_t)token * BENCH_MOE_N + column] - expected;
        error += difference * difference;
        reference += (double)expected * expected;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    double rel_rms = sqrt(error / (reference + 1e-30));
    printf("prefill-reduce: correctness rel-rms %.9g max-abs %.9g\n",
           rel_rms, max_abs);
    if (!isfinite(rel_rms) || rel_rms > 1e-7)
        bench_fatal("route reduce correctness failed");

    double probe = bench_reduce_launch(gpu, 5) / 5.0;
    int repetitions = (int)ceil(target_seconds / probe);
    if (repetitions < 5) repetitions = 5;
    printf("prefill-reduce: shape routes %d tokens %d N %d repetitions %d target %.3f s\n",
           BENCH_MOE_ROWS, BENCH_MOE_TOKENS, BENCH_MOE_N, repetitions,
           target_seconds);
    bench_sample samples[BENCH_ROUNDS_MAX];
    double times[BENCH_ROUNDS_MAX];
    double frequencies[BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        samples[round] = bench_reduce_measure(gpu, repetitions);
        times[round] = samples[round].seconds;
        frequencies[round] = samples[round].actual_mhz;
        printf("prefill-reduce: round %d %.6f ms@%.0fMHz[%ld,%ld] throttle=%ld/%ld/%ld\n",
               round + 1, samples[round].seconds * 1e3,
               samples[round].actual_mhz, samples[round].actual_min,
               samples[round].actual_max, samples[round].pl1,
               samples[round].pl2, samples[round].thermal);
    }
    double median = bench_median(times, rounds);
    double frequency = bench_median(frequencies, rounds);
    double bytes = (double)BENCH_MOE_ROWS * BENCH_MOE_N * sizeof(float)
                   + (double)BENCH_MOE_TOKENS * BENCH_MOE_N * sizeof(float)
                   + BENCH_MOE_ROWS * (sizeof(float) + sizeof(int));
    printf("prefill-reduce: median %.6f ms %.6f GB/s @%.0fMHz model-cost %.6f ms/token\n",
           median * 1e3, bytes / median / 1e9, frequency,
           median * 30.0 / BENCH_MOE_TOKENS * 1e3);
    bench_moe_data_destroy(gpu, &data);
}

static double bench_activate_launch(bench_gpu *gpu, int repetitions,
                                    int rows, int width) {
    ze_group_count_t groups = { (uint32_t)(rows * (width / 32) / 8), 1, 1 };
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        bench_ze_check("zeCommandListAppendLaunchKernel GeGLU Q8",
                       zeCommandListAppendLaunchKernel(gpu->commands,
                                                       gpu->geglu_q8,
                                                       &groups,
                                                       NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize GeGLU Q8",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static bench_sample bench_activate_measure(bench_gpu *gpu, int repetitions,
                                            int rows, int width) {
    bench_telemetry telemetry;
    bench_telemetry_start(&telemetry);
    double seconds = bench_activate_launch(gpu, repetitions, rows, width);
    bench_telemetry_stop(&telemetry);
    return (bench_sample) {
        .seconds = seconds / repetitions,
        .actual_mhz = (double)telemetry.actual_sum / telemetry.samples,
        .requested_mhz = (double)telemetry.requested_sum / telemetry.samples,
        .actual_min = telemetry.actual_min,
        .actual_max = telemetry.actual_max,
        .pl1 = telemetry.pl1,
        .pl2 = telemetry.pl2,
        .thermal = telemetry.thermal
    };
}

static float bench_gelu_lookup(float x) {
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return x;
    float h = (float)(_Float16)x;
    float inner = 0.79788456080286535588f
                  * (h + 0.044715f * h * h * h);
    return (float)(_Float16)(0.5f * h * (1.0f + tanhf(inner)));
}

static void bench_activate_run(bench_gpu *gpu, const char *distribution,
                               const char *activation_shape,
                               int rounds, double target_seconds) {
    if (BENCH_MOE_N != 1408 || BENCH_MOE_BLOCKS != 88)
        bench_fatal("--activate requires the gate/up benchmark binary");
    bench_moe_data data;
    bench_moe_data_init(gpu, &data);
    bench_moe_schedule(&data, distribution ? distribution : "multinomial");
    bench_moe_launch(gpu, BENCH_MOE_PADDED, &data, 1);
    int rows = activation_shape && !strcmp(activation_shape, "dense")
               ? BENCH_MOE_TOKENS : BENCH_MOE_ROWS;
    int width = activation_shape && !strcmp(activation_shape, "dense")
                ? 2112 : BENCH_MOE_N / 2;
    if (activation_shape && strcmp(activation_shape, "dense")
        && strcmp(activation_shape, "expert"))
        bench_fatal("unknown activation shape: %s", activation_shape);
    int blocks = width / 32;
    int8_t *quantized = bench_gpu_alloc(
        gpu, (size_t)rows * width);
    _Float16 *scale = bench_gpu_alloc(
        gpu, (size_t)rows * blocks * sizeof(_Float16));
    int16_t *sigma = bench_gpu_alloc(
        gpu, (size_t)rows * blocks * sizeof(int16_t));
    bench_pointer_arg(gpu->geglu_q8, 0, data.output);
    bench_pointer_arg(gpu->geglu_q8, 1, quantized);
    bench_pointer_arg(gpu->geglu_q8, 2, scale);
    bench_pointer_arg(gpu->geglu_q8, 3, sigma);
    bench_int_arg(gpu->geglu_q8, 4, rows);
    bench_int_arg(gpu->geglu_q8, 5, width);
    bench_activate_launch(gpu, 1, rows, width);

    double error = 0.0;
    double reference = 0.0;
    double max_abs = 0.0;
    int q_mismatches = 0;
    int sigma_mismatches = 0;
    for (int probe = 0; probe < 256; probe++) {
        int row = (probe * 37 + 11) % rows;
        int block = (probe * 17 + 5) % blocks;
        float values[32];
        float maximum = 0.0f;
        for (int i = 0; i < 32; i++) {
            int column = block * 32 + i;
            float gate = data.output[(size_t)row * width * 2 + column];
            float up = data.output[(size_t)row * width * 2 + width + column];
            values[i] = bench_gelu_lookup(gate) * up;
            if (fabsf(values[i]) > maximum) maximum = fabsf(values[i]);
        }
        float d = maximum / 127.0f;
        float inverse = d ? 1.0f / d : 0.0f;
        _Float16 stored_d = (_Float16)d;
        int expected_sigma = 0;
        size_t base = ((size_t)row * blocks + block) * 32;
        for (int i = 0; i < 32; i++) {
            int expected_q = (int)roundf(values[i] * inverse);
            int actual_q = quantized[base + i];
            expected_sigma += expected_q;
            if (actual_q != expected_q) q_mismatches++;
            double expected = expected_q * (float)stored_d;
            double actual = actual_q
                            * (float)scale[(size_t)row * blocks + block];
            double difference = actual - expected;
            error += difference * difference;
            reference += expected * expected;
            if (fabs(difference) > max_abs) max_abs = fabs(difference);
        }
        if (sigma[(size_t)row * blocks + block] != expected_sigma)
            sigma_mismatches++;
    }
    double rel_rms = sqrt(error / (reference + 1e-30));
    printf("prefill-activate: correctness rel-rms %.9g max-abs %.9g q-mismatch %d/8192 sigma-mismatch %d/256\n",
           rel_rms, max_abs, q_mismatches, sigma_mismatches);
    if (!isfinite(rel_rms) || rel_rms > 5e-5)
        bench_fatal("GeGLU Q8 correctness failed");

    double probe = bench_activate_launch(gpu, 5, rows, width) / 5.0;
    int repetitions = (int)ceil(target_seconds / probe);
    if (repetitions < 5) repetitions = 5;
    printf("prefill-activate: shape rows %d gate-up %d output %d repetitions %d target %.3f s\n",
           rows, width * 2, width, repetitions, target_seconds);
    bench_sample samples[BENCH_ROUNDS_MAX];
    double times[BENCH_ROUNDS_MAX];
    double frequencies[BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        samples[round] = bench_activate_measure(gpu, repetitions, rows, width);
        times[round] = samples[round].seconds;
        frequencies[round] = samples[round].actual_mhz;
        printf("prefill-activate: round %d %.6f ms@%.0fMHz[%ld,%ld] throttle=%ld/%ld/%ld\n",
               round + 1, samples[round].seconds * 1e3,
               samples[round].actual_mhz, samples[round].actual_min,
               samples[round].actual_max, samples[round].pl1,
               samples[round].pl2, samples[round].thermal);
    }
    double median = bench_median(times, rounds);
    double frequency = bench_median(frequencies, rounds);
    printf("prefill-activate: median %.6f ms @%.0fMHz model-cost %.6f ms/token\n",
           median * 1e3, frequency,
           median * 30.0 / BENCH_MOE_TOKENS * 1e3);
    bench_ze_check("zeMemFree GeGLU sigma", zeMemFree(gpu->context, sigma));
    bench_ze_check("zeMemFree GeGLU scale", zeMemFree(gpu->context, scale));
    bench_ze_check("zeMemFree GeGLU quantized",
                   zeMemFree(gpu->context, quantized));
    bench_moe_data_destroy(gpu, &data);
}

static double bench_router_launch(bench_gpu *gpu, int variant,
                                  int repetitions) {
    ze_group_count_t naive_groups = { BENCH_MOE_TOKENS * BENCH_MOE_EXPERTS / 8,
                                      1, 1 };
    ze_group_count_t tiled_groups = { BENCH_MOE_TOKENS / 16,
                                      BENCH_MOE_EXPERTS / 16, 1 };
    ze_group_count_t top8_groups = { BENCH_MOE_TOKENS / 8, 1, 1 };
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        ze_kernel_handle_t kernel = variant == 0 ? gpu->router_gemm
                                    : gpu->router_gemm_tiled;
        const ze_group_count_t *groups = variant == 0 ? &naive_groups
                                         : &tiled_groups;
        bench_ze_check("zeCommandListAppendLaunchKernel router GEMM",
                       zeCommandListAppendLaunchKernel(gpu->commands, kernel,
                                                       groups,
                                                       NULL, 0, NULL));
        if (variant == 2)
            bench_ze_check("zeCommandListAppendLaunchKernel router top8",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->router_top8,
                                                           &top8_groups,
                                                           NULL, 0, NULL));
    }
    bench_ze_check("zeCommandListHostSynchronize router",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static bench_sample bench_router_measure(bench_gpu *gpu, int variant,
                                         int repetitions) {
    bench_telemetry telemetry;
    bench_telemetry_start(&telemetry);
    double seconds = bench_router_launch(gpu, variant, repetitions);
    bench_telemetry_stop(&telemetry);
    return (bench_sample) {
        .seconds = seconds / repetitions,
        .actual_mhz = (double)telemetry.actual_sum / telemetry.samples,
        .requested_mhz = (double)telemetry.requested_sum / telemetry.samples,
        .actual_min = telemetry.actual_min,
        .actual_max = telemetry.actual_max,
        .pl1 = telemetry.pl1,
        .pl2 = telemetry.pl2,
        .thermal = telemetry.thermal
    };
}

static void bench_router_run(bench_gpu *gpu, int rounds,
                             double target_seconds) {
    const int rows = BENCH_MOE_TOKENS;
    const int width = 2816;
    const int experts = BENCH_MOE_EXPERTS;
    float *input = bench_gpu_alloc(gpu, (size_t)rows * width * sizeof(float));
    float *weight = bench_gpu_alloc(gpu,
                                    (size_t)experts * width * sizeof(float));
    float *logits = bench_gpu_alloc(gpu,
                                    (size_t)rows * experts * sizeof(float));
    int *route_expert = bench_gpu_alloc(gpu, (size_t)rows * 8 * sizeof(int));
    float *route_weight = bench_gpu_alloc(gpu,
                                          (size_t)rows * 8 * sizeof(float));
    uint32_t random = UINT32_C(0xa54ff53a);
    for (int index = 0; index < rows * width; index++)
        input[index] = ((int)(bench_random(&random) >> 8) % 2001 - 1000)
                       / 1000.0f;
    for (int index = 0; index < experts * width; index++)
        weight[index] = ((int)(bench_random(&random) >> 8) % 2001 - 1000)
                        / 32000.0f;
    bench_pointer_arg(gpu->router_gemm, 0, input);
    bench_pointer_arg(gpu->router_gemm, 1, weight);
    bench_pointer_arg(gpu->router_gemm, 2, logits);
    bench_int_arg(gpu->router_gemm, 3, rows);
    bench_int_arg(gpu->router_gemm, 4, width);
    bench_int_arg(gpu->router_gemm, 5, experts);
    bench_pointer_arg(gpu->router_gemm_tiled, 0, input);
    bench_pointer_arg(gpu->router_gemm_tiled, 1, weight);
    bench_pointer_arg(gpu->router_gemm_tiled, 2, logits);
    bench_int_arg(gpu->router_gemm_tiled, 3, rows);
    bench_int_arg(gpu->router_gemm_tiled, 4, width);
    bench_int_arg(gpu->router_gemm_tiled, 5, experts);
    bench_pointer_arg(gpu->router_top8, 0, logits);
    bench_pointer_arg(gpu->router_top8, 1, route_expert);
    bench_pointer_arg(gpu->router_top8, 2, route_weight);
    bench_int_arg(gpu->router_top8, 3, rows);
    bench_int_arg(gpu->router_top8, 4, experts);
    bench_router_launch(gpu, 2, 1);

    double error = 0.0;
    double reference = 0.0;
    double max_abs = 0.0;
    for (int probe = 0; probe < 128; probe++) {
        int row = (probe * 37 + 11) % rows;
        int expert = (probe * 101 + 17) % experts;
        double expected = 0.0;
        for (int k = 0; k < width; k++)
            expected += (double)input[(size_t)row * width + k]
                        * weight[(size_t)expert * width + k];
        double difference = logits[(size_t)row * experts + expert] - expected;
        error += difference * difference;
        reference += expected * expected;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    double rel_rms = sqrt(error / (reference + 1e-30));
    int top8_mismatches = 0;
    double weight_error = 0.0;
    double weight_reference = 0.0;
    for (int row = 0; row < rows; row++) {
        int selected[8];
        float selected_value[8];
        for (int slot = 0; slot < 8; slot++) {
            selected[slot] = -1;
            selected_value[slot] = -INFINITY;
        }
        for (int expert = 0; expert < experts; expert++) {
            float value = logits[(size_t)row * experts + expert];
            if (value <= selected_value[7]) continue;
            int slot = 7;
            while (slot > 0 && selected_value[slot - 1] < value) {
                selected_value[slot] = selected_value[slot - 1];
                selected[slot] = selected[slot - 1];
                slot--;
            }
            selected_value[slot] = value;
            selected[slot] = expert;
        }
        float sum = 0.0f;
        float expected_weight[8];
        for (int slot = 0; slot < 8; slot++) {
            expected_weight[slot] = expf(selected_value[slot]
                                         - selected_value[0]);
            sum += expected_weight[slot];
        }
        for (int slot = 0; slot < 8; slot++) {
            if (route_expert[row * 8 + slot] != selected[slot])
                top8_mismatches++;
            double expected = expected_weight[slot] / sum;
            double difference = route_weight[row * 8 + slot] - expected;
            weight_error += difference * difference;
            weight_reference += expected * expected;
        }
    }
    double weight_rel_rms = sqrt(weight_error / (weight_reference + 1e-30));
    printf("prefill-router: correctness logits rel-rms %.9g max-abs %.9g top8-mismatch %d/4096 weights rel-rms %.9g\n",
           rel_rms, max_abs, top8_mismatches, weight_rel_rms);
    if (!isfinite(rel_rms) || rel_rms > 5e-5 || top8_mismatches
        || !isfinite(weight_rel_rms) || weight_rel_rms > 5e-6)
        bench_fatal("router correctness failed");

    int repetitions[3];
    for (int variant = 0; variant < 3; variant++) {
        double probe = bench_router_launch(gpu, variant, 5) / 5.0;
        repetitions[variant] = (int)ceil(target_seconds / probe);
        if (repetitions[variant] < 5) repetitions[variant] = 5;
    }
    printf("prefill-router: shape M %d N %d K %d repetitions naive/tiled/complete %d/%d/%d target %.3f s\n",
           rows, experts, width, repetitions[0], repetitions[1],
           repetitions[2], target_seconds);
    bench_sample samples[3][BENCH_ROUNDS_MAX];
    double times[3][BENCH_ROUNDS_MAX];
    double frequencies[3][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        for (int position = 0; position < 3; position++) {
            int variant = (round + position) % 3;
            samples[variant][round] = bench_router_measure(
                gpu, variant, repetitions[variant]);
            times[variant][round] = samples[variant][round].seconds;
            frequencies[variant][round] = samples[variant][round].actual_mhz;
        }
        printf("prefill-router: round %d naive %.6f ms@%.0fMHz tiled %.6f ms@%.0fMHz complete %.6f ms@%.0fMHz\n",
               round + 1, samples[0][round].seconds * 1e3,
               samples[0][round].actual_mhz, samples[1][round].seconds * 1e3,
               samples[1][round].actual_mhz, samples[2][round].seconds * 1e3,
               samples[2][round].actual_mhz);
    }
    double naive = bench_median(times[0], rounds);
    double tiled = bench_median(times[1], rounds);
    double complete = bench_median(times[2], rounds);
    double naive_frequency = bench_median(frequencies[0], rounds);
    double tiled_frequency = bench_median(frequencies[1], rounds);
    double complete_frequency = bench_median(frequencies[2], rounds);
    double operations = 2.0 * rows * experts * width;
    printf("prefill-router: median naive %.6f ms %.6f TFLOP/s @%.0fMHz tiled %.6f ms %.6f TFLOP/s @%.0fMHz speedup %.6fx complete %.6f ms @%.0fMHz top8-tax %.6f ms model-cost %.6f ms/token\n",
           naive * 1e3, operations / naive / 1e12, naive_frequency,
           tiled * 1e3, operations / tiled / 1e12, tiled_frequency,
           naive / tiled, complete * 1e3, complete_frequency,
           (complete - tiled) * 1e3,
           complete * 30.0 / rows * 1e3);
    bench_ze_check("zeMemFree router route weight",
                   zeMemFree(gpu->context, route_weight));
    bench_ze_check("zeMemFree router route expert",
                   zeMemFree(gpu->context, route_expert));
    bench_ze_check("zeMemFree router logits", zeMemFree(gpu->context, logits));
    bench_ze_check("zeMemFree router weight", zeMemFree(gpu->context, weight));
    bench_ze_check("zeMemFree router input", zeMemFree(gpu->context, input));
}

static double bench_input_launch(bench_gpu *gpu, int repetitions) {
    ze_group_count_t rms_groups = { BENCH_MOE_TOKENS, 1, 1 };
    ze_group_count_t groups = { BENCH_MOE_TOKENS * 88 / 8, 1, 1 };
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        bench_ze_check("zeCommandListAppendLaunchKernel FFN RMS",
                       zeCommandListAppendLaunchKernel(gpu->commands,
                                                       gpu->rms_scale,
                                                       &rms_groups,
                                                       NULL, 0, NULL));
        bench_ze_check("zeCommandListAppendLaunchKernel FFN input Q8 rowscale",
                       zeCommandListAppendLaunchKernel(gpu->commands,
                                                       gpu->ffn_input_q8_rowscale,
                                                       &groups,
                                                       NULL, 0, NULL));
    }
    bench_ze_check("zeCommandListHostSynchronize FFN input Q8",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static bench_sample bench_input_measure(bench_gpu *gpu, int repetitions) {
    bench_telemetry telemetry;
    bench_telemetry_start(&telemetry);
    double seconds = bench_input_launch(gpu, repetitions);
    bench_telemetry_stop(&telemetry);
    return (bench_sample) {
        .seconds = seconds / repetitions,
        .actual_mhz = (double)telemetry.actual_sum / telemetry.samples,
        .requested_mhz = (double)telemetry.requested_sum / telemetry.samples,
        .actual_min = telemetry.actual_min,
        .actual_max = telemetry.actual_max,
        .pl1 = telemetry.pl1,
        .pl2 = telemetry.pl2,
        .thermal = telemetry.thermal
    };
}

static void bench_input_run(bench_gpu *gpu, int rounds,
                            double target_seconds) {
    const int rows = BENCH_MOE_TOKENS;
    const int width = 2816;
    const int blocks = width / 32;
    size_t elements = (size_t)rows * width;
    size_t scale_elements = (size_t)rows * blocks;
    float *input = bench_gpu_alloc(gpu, elements * sizeof(float));
    float *dense_weight = bench_gpu_alloc(gpu, width * sizeof(float));
    float *moe_weight = bench_gpu_alloc(gpu, width * sizeof(float));
    float *router_weight = bench_gpu_alloc(gpu, width * sizeof(float));
    int8_t *dense_q = bench_gpu_alloc(gpu, elements);
    _Float16 *dense_d = bench_gpu_alloc(gpu,
                                        scale_elements * sizeof(_Float16));
    int16_t *dense_s = bench_gpu_alloc(gpu,
                                       scale_elements * sizeof(int16_t));
    int8_t *moe_q = bench_gpu_alloc(gpu, elements);
    _Float16 *moe_d = bench_gpu_alloc(gpu,
                                      scale_elements * sizeof(_Float16));
    int16_t *moe_s = bench_gpu_alloc(gpu,
                                     scale_elements * sizeof(int16_t));
    float *router_input = bench_gpu_alloc(gpu, elements * sizeof(float));
    float *row_scale = bench_gpu_alloc(gpu, rows * sizeof(float));
    uint32_t random = UINT32_C(0x5be0cd19);
    for (size_t index = 0; index < elements; index++)
        input[index] = ((int)(bench_random(&random) % 2001) - 1000) / 250.0f;
    for (int index = 0; index < width; index++) {
        dense_weight[index] =
            ((int)(bench_random(&random) % 2001) - 1000) / 1000.0f;
        moe_weight[index] =
            ((int)(bench_random(&random) % 2001) - 1000) / 1000.0f;
        router_weight[index] =
            ((int)(bench_random(&random) % 2001) - 1000) / 1000.0f;
    }
    float router_scale = 1.0f / sqrtf((float)width);
    bench_pointer_arg(gpu->rms_scale, 0, input);
    bench_pointer_arg(gpu->rms_scale, 1, row_scale);
    bench_int_arg(gpu->rms_scale, 2, rows);
    bench_int_arg(gpu->rms_scale, 3, width);
    bench_float_arg(gpu->rms_scale, 4, 1e-6f);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 0, input);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 1, dense_weight);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 2, moe_weight);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 3, router_weight);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 4, row_scale);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 5, dense_q);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 6, dense_d);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 7, dense_s);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 8, moe_q);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 9, moe_d);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 10, moe_s);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 11, router_input);
    bench_float_arg(gpu->ffn_input_q8_rowscale, 12, router_scale);
    bench_int_arg(gpu->ffn_input_q8_rowscale, 13, rows);
    bench_int_arg(gpu->ffn_input_q8_rowscale, 14, width);
    bench_input_launch(gpu, 1);

    float *expected_scale = bench_alloc(rows * sizeof(float));
    for (int row = 0; row < rows; row++) {
        double sum = 0.0;
        for (int column = 0; column < width; column++) {
            double value = input[(size_t)row * width + column];
            sum += value * value;
        }
        expected_scale[row] = 1.0f / sqrtf((float)(sum / width) + 1e-6f);
    }

    int q_mismatches = 0;
    int sigma_mismatches = 0;
    double reconstruction_error = 0.0;
    double reconstruction_reference = 0.0;
    double router_error = 0.0;
    double router_reference = 0.0;
    for (int probe = 0; probe < 256; probe++) {
        int row = (probe * 37 + 11) % rows;
        int block = (probe * 17 + 5) % blocks;
        float dense_values[32];
        float moe_values[32];
        float dense_max = 0.0f;
        float moe_max = 0.0f;
        for (int i = 0; i < 32; i++) {
            int column = block * 32 + i;
            float x = input[(size_t)row * width + column];
            dense_values[i] = x * expected_scale[row] * dense_weight[column];
            moe_values[i] = x * expected_scale[row] * moe_weight[column];
            dense_max = fmaxf(dense_max, fabsf(dense_values[i]));
            moe_max = fmaxf(moe_max, fabsf(moe_values[i]));
            double expected_router = x * expected_scale[row] * router_scale
                                     * router_weight[column];
            double difference = router_input[(size_t)row * width + column]
                                - expected_router;
            router_error += difference * difference;
            router_reference += expected_router * expected_router;
        }
        float dd = dense_max / 127.0f;
        float md = moe_max / 127.0f;
        float dense_inverse = dd ? 1.0f / dd : 0.0f;
        float moe_inverse = md ? 1.0f / md : 0.0f;
        int expected_dense_sigma = 0;
        int expected_moe_sigma = 0;
        size_t base = ((size_t)row * blocks + block) * 32;
        for (int i = 0; i < 32; i++) {
            int expected_dense = (int)roundf(dense_values[i] * dense_inverse);
            int expected_moe = (int)roundf(moe_values[i] * moe_inverse);
            int actual_dense = dense_q[base + i];
            int actual_moe = moe_q[base + i];
            expected_dense_sigma += expected_dense;
            expected_moe_sigma += expected_moe;
            if (actual_dense != expected_dense) q_mismatches++;
            if (actual_moe != expected_moe) q_mismatches++;
            double expected_dense_value = expected_dense * (float)(_Float16)dd;
            double actual_dense_value = actual_dense
                                        * (float)dense_d[(size_t)row * blocks
                                                         + block];
            double expected_moe_value = expected_moe * (float)(_Float16)md;
            double actual_moe_value = actual_moe
                                      * (float)moe_d[(size_t)row * blocks
                                                     + block];
            double dense_difference = actual_dense_value - expected_dense_value;
            double moe_difference = actual_moe_value - expected_moe_value;
            reconstruction_error += dense_difference * dense_difference
                                    + moe_difference * moe_difference;
            reconstruction_reference += expected_dense_value
                                         * expected_dense_value
                                         + expected_moe_value * expected_moe_value;
        }
        size_t scale_index = (size_t)row * blocks + block;
        if (dense_s[scale_index] != expected_dense_sigma) sigma_mismatches++;
        if (moe_s[scale_index] != expected_moe_sigma) sigma_mismatches++;
    }
    double rel_rms = sqrt(reconstruction_error
                          / (reconstruction_reference + 1e-30));
    double router_rel_rms = sqrt(router_error / (router_reference + 1e-30));
    double scale_error = 0.0;
    double scale_reference = 0.0;
    for (int row = 0; row < rows; row++) {
        double difference = row_scale[row] - expected_scale[row];
        scale_error += difference * difference;
        scale_reference += (double)expected_scale[row] * expected_scale[row];
    }
    double scale_rel_rms = sqrt(scale_error / scale_reference);
    printf("prefill-input: correctness rms rel-rms %.9g q8 rel-rms %.9g q-mismatch %d/16384 sigma-mismatch %d/512 router rel-rms %.9g\n",
           scale_rel_rms, rel_rms, q_mismatches, sigma_mismatches,
           router_rel_rms);
    if (!isfinite(rel_rms) || rel_rms > 5e-5 || !isfinite(router_rel_rms)
        || router_rel_rms > 1e-6 || scale_rel_rms > 1e-6)
        bench_fatal("FFN input correctness failed");
    free(expected_scale);

    double probe = bench_input_launch(gpu, 5) / 5.0;
    int repetitions = (int)ceil(target_seconds / probe);
    if (repetitions < 5) repetitions = 5;
    printf("prefill-input: shape rows %d width %d repetitions %d target %.3f s\n",
           rows, width, repetitions, target_seconds);
    bench_sample samples[BENCH_ROUNDS_MAX];
    double times[BENCH_ROUNDS_MAX];
    double frequencies[BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        samples[round] = bench_input_measure(gpu, repetitions);
        times[round] = samples[round].seconds;
        frequencies[round] = samples[round].actual_mhz;
        printf("prefill-input: round %d %.6f ms@%.0fMHz[%ld,%ld] throttle=%ld/%ld/%ld\n",
               round + 1, samples[round].seconds * 1e3,
               samples[round].actual_mhz, samples[round].actual_min,
               samples[round].actual_max, samples[round].pl1,
               samples[round].pl2, samples[round].thermal);
    }
    double median = bench_median(times, rounds);
    double frequency = bench_median(frequencies, rounds);
    printf("prefill-input: median RMS-plus-transform %.6f ms @%.0fMHz model-cost %.6f ms/token\n",
           median * 1e3, frequency,
           median * 30.0 / BENCH_MOE_TOKENS * 1e3);
    bench_ze_check("zeMemFree FFN router input",
                   zeMemFree(gpu->context, router_input));
    bench_ze_check("zeMemFree FFN row scale",
                   zeMemFree(gpu->context, row_scale));
    bench_ze_check("zeMemFree FFN moe sigma", zeMemFree(gpu->context, moe_s));
    bench_ze_check("zeMemFree FFN moe scale", zeMemFree(gpu->context, moe_d));
    bench_ze_check("zeMemFree FFN moe q", zeMemFree(gpu->context, moe_q));
    bench_ze_check("zeMemFree FFN dense sigma",
                   zeMemFree(gpu->context, dense_s));
    bench_ze_check("zeMemFree FFN dense scale",
                   zeMemFree(gpu->context, dense_d));
    bench_ze_check("zeMemFree FFN dense q", zeMemFree(gpu->context, dense_q));
    bench_ze_check("zeMemFree FFN router weight",
                   zeMemFree(gpu->context, router_weight));
    bench_ze_check("zeMemFree FFN moe weight",
                   zeMemFree(gpu->context, moe_weight));
    bench_ze_check("zeMemFree FFN dense weight",
                   zeMemFree(gpu->context, dense_weight));
    bench_ze_check("zeMemFree FFN input", zeMemFree(gpu->context, input));
}

static double bench_qkv_post_launch(bench_gpu *gpu, int rows, int kv_heads,
                                    int repetitions) {
    ze_group_count_t groups = { (uint32_t)(rows * (16 + 2 * kv_heads)), 1, 1 };
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        bench_ze_check("zeCommandListAppendLaunchKernel QKV post",
                       zeCommandListAppendLaunchKernel(gpu->commands,
                                                       gpu->qkv_post,
                                                       &groups,
                                                       NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize QKV post",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static void bench_qkv_post_run(bench_gpu *gpu, const char *name, int rounds,
                               double target_seconds) {
    const int rows = 512;
    int dimension;
    int kv_heads;
    int has_v;
    float rope_base;
    if (!name || !strcmp(name, "swa")) {
        dimension = 256;
        kv_heads = 8;
        has_v = 1;
        rope_base = 10000.0f;
    } else if (!strcmp(name, "global")) {
        dimension = 512;
        kv_heads = 2;
        has_v = 0;
        rope_base = 1000000.0f;
    } else {
        bench_fatal("unknown QKV post shape: %s", name);
    }
    size_t q_elements = (size_t)rows * 16 * dimension;
    size_t kv_elements = (size_t)rows * kv_heads * dimension;
    size_t rope_elements = (size_t)rows * dimension / 2;
    float *q_projection = bench_gpu_alloc(gpu, q_elements * sizeof(float));
    float *k_projection = bench_gpu_alloc(gpu, kv_elements * sizeof(float));
    float *v_projection = bench_gpu_alloc(gpu, kv_elements * sizeof(float));
    float *q_weight = bench_gpu_alloc(gpu, dimension * sizeof(float));
    float *k_weight = bench_gpu_alloc(gpu, dimension * sizeof(float));
    float *rope_cos = bench_gpu_alloc(gpu, rope_elements * sizeof(float));
    float *rope_sin = bench_gpu_alloc(gpu, rope_elements * sizeof(float));
    float *q = bench_gpu_alloc(gpu, q_elements * sizeof(float));
    _Float16 *k = bench_gpu_alloc(gpu, kv_elements * sizeof(_Float16));
    _Float16 *v = bench_gpu_alloc(gpu, kv_elements * sizeof(_Float16));
    uint32_t random = UINT32_C(0x243f6a88) ^ (uint32_t)dimension;
    for (size_t index = 0; index < q_elements; index++)
        q_projection[index] =
            ((int)(bench_random(&random) % 2001) - 1000) / 500.0f;
    for (size_t index = 0; index < kv_elements; index++) {
        k_projection[index] =
            ((int)(bench_random(&random) % 2001) - 1000) / 500.0f;
        v_projection[index] =
            ((int)(bench_random(&random) % 2001) - 1000) / 500.0f;
    }
    for (int d = 0; d < dimension; d++) {
        q_weight[d] = ((int)(bench_random(&random) % 1001) + 500) / 1000.0f;
        k_weight[d] = ((int)(bench_random(&random) % 1001) + 500) / 1000.0f;
    }
    for (int row = 0; row < rows; row++) {
        for (int d = 0; d < dimension / 2; d++) {
            float inverse = powf(rope_base,
                                 -2.0f * (float)d / (float)dimension);
            float theta = row * inverse;
            rope_cos[(size_t)row * (dimension / 2) + d] = cosf(theta);
            rope_sin[(size_t)row * (dimension / 2) + d] = sinf(theta);
        }
    }
    bench_pointer_arg(gpu->qkv_post, 0, q_projection);
    bench_pointer_arg(gpu->qkv_post, 1, k_projection);
    bench_pointer_arg(gpu->qkv_post, 2, v_projection);
    bench_pointer_arg(gpu->qkv_post, 3, q_weight);
    bench_pointer_arg(gpu->qkv_post, 4, k_weight);
    bench_pointer_arg(gpu->qkv_post, 5, rope_cos);
    bench_pointer_arg(gpu->qkv_post, 6, rope_sin);
    bench_pointer_arg(gpu->qkv_post, 7, q);
    bench_pointer_arg(gpu->qkv_post, 8, k);
    bench_pointer_arg(gpu->qkv_post, 9, v);
    bench_int_arg(gpu->qkv_post, 10, rows);
    bench_int_arg(gpu->qkv_post, 11, dimension);
    bench_int_arg(gpu->qkv_post, 12, kv_heads);
    bench_int_arg(gpu->qkv_post, 13, has_v);
    bench_float_arg(gpu->qkv_post, 14, 1e-6f);
    bench_qkv_post_launch(gpu, rows, kv_heads, 1);

    double error = 0.0;
    double reference = 0.0;
    double max_abs = 0.0;
    for (int probe = 0; probe < 256; probe++) {
        int kind = probe % 3;
        int head_count = kind == 0 ? 16 : kv_heads;
        int head = (probe * 17 + 5) % head_count;
        int row = (probe * 37 + 11) % rows;
        int d = (probe * 101 + 17) % dimension;
        const float *source = kind == 0 ? q_projection
                              : kind == 1 || !has_v ? k_projection
                              : v_projection;
        size_t source_base = ((size_t)row * head_count + head) * dimension;
        double sum = 0.0;
        for (int column = 0; column < dimension; column++) {
            double value = source[source_base + column];
            sum += value * value;
        }
        float scale = 1.0f / sqrtf((float)(sum / dimension) + 1e-6f);
        double expected;
        double actual;
        if (kind == 2) {
            expected = (float)(_Float16)(source[source_base + d] * scale);
            actual = (float)v[((size_t)head * rows + row) * dimension + d];
        } else {
            int half = dimension / 2;
            int pair = d % half;
            const float *weight = kind == 0 ? q_weight : k_weight;
            float lo = source[source_base + pair] * scale * weight[pair];
            float hi = source[source_base + pair + half] * scale
                       * weight[pair + half];
            float cosine = rope_cos[(size_t)row * half + pair];
            float sine = rope_sin[(size_t)row * half + pair];
            float rotated = d < half ? lo * cosine - hi * sine
                            : lo * sine + hi * cosine;
            if (kind == 0) {
                expected = rotated;
                actual = q[((size_t)head * rows + row) * dimension + d];
            } else {
                expected = (float)(_Float16)rotated;
                actual = (float)k[((size_t)head * rows + row) * dimension + d];
            }
        }
        double difference = actual - expected;
        error += difference * difference;
        reference += expected * expected;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    double rel_rms = sqrt(error / (reference + 1e-30));
    printf("prefill-qkv-post: correctness %s rel-rms %.9g max-abs %.9g probes 256\n",
           name ? name : "swa", rel_rms, max_abs);
    if (!isfinite(rel_rms) || rel_rms > 1e-6)
        bench_fatal("QKV post correctness failed");

    double probe = bench_qkv_post_launch(gpu, rows, kv_heads, 5) / 5.0;
    int repetitions = (int)ceil(target_seconds / probe);
    if (repetitions < 5) repetitions = 5;
    bench_sample samples[BENCH_ROUNDS_MAX];
    double times[BENCH_ROUNDS_MAX];
    double frequencies[BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        bench_telemetry telemetry;
        bench_telemetry_start(&telemetry);
        double seconds = bench_qkv_post_launch(
            gpu, rows, kv_heads, repetitions);
        bench_telemetry_stop(&telemetry);
        samples[round] = (bench_sample) {
            .seconds = seconds / repetitions,
            .actual_mhz = (double)telemetry.actual_sum / telemetry.samples
        };
        times[round] = samples[round].seconds;
        frequencies[round] = samples[round].actual_mhz;
        printf("prefill-qkv-post: round %d %.6f ms@%.0fMHz\n",
               round + 1, times[round] * 1e3, frequencies[round]);
    }
    double median = bench_median(times, rounds);
    double frequency = bench_median(frequencies, rounds);
    printf("prefill-qkv-post: median %s %.6f ms @%.0fMHz model-cost %.6f ms/token\n",
           name ? name : "swa", median * 1e3, frequency,
           median * 30.0 / rows * 1e3);

    bench_ze_check("zeMemFree QKV v", zeMemFree(gpu->context, v));
    bench_ze_check("zeMemFree QKV k", zeMemFree(gpu->context, k));
    bench_ze_check("zeMemFree QKV q", zeMemFree(gpu->context, q));
    bench_ze_check("zeMemFree QKV rope sin", zeMemFree(gpu->context, rope_sin));
    bench_ze_check("zeMemFree QKV rope cos", zeMemFree(gpu->context, rope_cos));
    bench_ze_check("zeMemFree QKV k weight", zeMemFree(gpu->context, k_weight));
    bench_ze_check("zeMemFree QKV q weight", zeMemFree(gpu->context, q_weight));
    bench_ze_check("zeMemFree QKV V projection",
                   zeMemFree(gpu->context, v_projection));
    bench_ze_check("zeMemFree QKV K projection",
                   zeMemFree(gpu->context, k_projection));
    bench_ze_check("zeMemFree QKV Q projection",
                   zeMemFree(gpu->context, q_projection));
}

static double bench_glue_launch(bench_gpu *gpu, int variant, int rows,
                                int attention_width, int repetitions) {
    ze_group_count_t groups = {
        variant == 0 ? (uint32_t)(rows * (attention_width / 32) / 8)
                     : (uint32_t)rows,
        1, 1
    };
    ze_kernel_handle_t kernel = variant == 0 ? gpu->heads_q8
                                : variant == 1 ? gpu->rms_residual
                                : gpu->ffn_finish;
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        bench_ze_check("zeCommandListAppendLaunchKernel glue",
                       zeCommandListAppendLaunchKernel(gpu->commands, kernel,
                                                       &groups,
                                                       NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize glue",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static void bench_glue_run(bench_gpu *gpu, const char *name, int rounds,
                           double target_seconds) {
    const int rows = 512;
    const int width = 2816;
    int dimension;
    if (!name || !strcmp(name, "swa")) dimension = 256;
    else if (!strcmp(name, "global")) dimension = 512;
    else bench_fatal("unknown glue shape: %s", name);
    const int head_count = 16;
    int attention_width = head_count * dimension;
    int attention_blocks = attention_width / 32;
    size_t head_elements = (size_t)rows * attention_width;
    size_t hidden_elements = (size_t)rows * width;
    float *heads = bench_gpu_alloc(gpu, head_elements * sizeof(float));
    int8_t *heads_q = bench_gpu_alloc(gpu, head_elements);
    _Float16 *heads_d = bench_gpu_alloc(
        gpu, (size_t)rows * attention_blocks * sizeof(_Float16));
    int16_t *heads_s = bench_gpu_alloc(
        gpu, (size_t)rows * attention_blocks * sizeof(int16_t));
    float *projection = bench_gpu_alloc(gpu, hidden_elements * sizeof(float));
    float *dense = bench_gpu_alloc(gpu, hidden_elements * sizeof(float));
    float *moe = bench_gpu_alloc(gpu, hidden_elements * sizeof(float));
    float *residual = bench_gpu_alloc(gpu, hidden_elements * sizeof(float));
    float *post_weight = bench_gpu_alloc(gpu, width * sizeof(float));
    float *dense_weight = bench_gpu_alloc(gpu, width * sizeof(float));
    float *moe_weight = bench_gpu_alloc(gpu, width * sizeof(float));
    float *combine_weight = bench_gpu_alloc(gpu, width * sizeof(float));
    float *post_output = bench_gpu_alloc(gpu, hidden_elements * sizeof(float));
    float *finish_output = bench_gpu_alloc(gpu, hidden_elements * sizeof(float));
    float *layer_scale = bench_gpu_alloc(gpu, sizeof(float));
    uint32_t random = UINT32_C(0x13198a2e) ^ (uint32_t)dimension;
    for (size_t index = 0; index < head_elements; index++)
        heads[index] = ((int)(bench_random(&random) % 2001) - 1000) / 500.0f;
    for (size_t index = 0; index < hidden_elements; index++) {
        projection[index] =
            ((int)(bench_random(&random) % 2001) - 1000) / 500.0f;
        dense[index] =
            ((int)(bench_random(&random) % 2001) - 1000) / 500.0f;
        moe[index] =
            ((int)(bench_random(&random) % 2001) - 1000) / 500.0f;
        residual[index] =
            ((int)(bench_random(&random) % 2001) - 1000) / 500.0f;
    }
    for (int column = 0; column < width; column++) {
        post_weight[column] =
            ((int)(bench_random(&random) % 1001) + 500) / 1000.0f;
        dense_weight[column] =
            ((int)(bench_random(&random) % 1001) + 500) / 1000.0f;
        moe_weight[column] =
            ((int)(bench_random(&random) % 1001) + 500) / 1000.0f;
        combine_weight[column] =
            ((int)(bench_random(&random) % 1001) + 500) / 1000.0f;
    }
    layer_scale[0] = 0.875f;

    bench_pointer_arg(gpu->heads_q8, 0, heads);
    bench_pointer_arg(gpu->heads_q8, 1, heads_q);
    bench_pointer_arg(gpu->heads_q8, 2, heads_d);
    bench_pointer_arg(gpu->heads_q8, 3, heads_s);
    bench_int_arg(gpu->heads_q8, 4, rows);
    bench_int_arg(gpu->heads_q8, 5, head_count);
    bench_int_arg(gpu->heads_q8, 6, dimension);

    bench_pointer_arg(gpu->rms_residual, 0, projection);
    bench_pointer_arg(gpu->rms_residual, 1, post_weight);
    bench_pointer_arg(gpu->rms_residual, 2, residual);
    bench_pointer_arg(gpu->rms_residual, 3, post_output);
    bench_int_arg(gpu->rms_residual, 4, rows);
    bench_int_arg(gpu->rms_residual, 5, width);
    bench_float_arg(gpu->rms_residual, 6, 1e-6f);

    bench_pointer_arg(gpu->ffn_finish, 0, dense);
    bench_pointer_arg(gpu->ffn_finish, 1, moe);
    bench_pointer_arg(gpu->ffn_finish, 2, dense_weight);
    bench_pointer_arg(gpu->ffn_finish, 3, moe_weight);
    bench_pointer_arg(gpu->ffn_finish, 4, combine_weight);
    bench_pointer_arg(gpu->ffn_finish, 5, residual);
    bench_pointer_arg(gpu->ffn_finish, 6, layer_scale);
    bench_pointer_arg(gpu->ffn_finish, 7, finish_output);
    bench_int_arg(gpu->ffn_finish, 8, rows);
    bench_int_arg(gpu->ffn_finish, 9, width);
    bench_float_arg(gpu->ffn_finish, 10, 1e-6f);

    for (int variant = 0; variant < 3; variant++)
        bench_glue_launch(gpu, variant, rows, attention_width, 1);

    int q_mismatches = 0;
    int sigma_mismatches = 0;
    int q_max_delta = 0;
    double q_error = 0.0;
    double q_reference = 0.0;
    for (int probe = 0; probe < 128; probe++) {
        int row = (probe * 37 + 11) % rows;
        int block = (probe * 17 + 5) % attention_blocks;
        float values[32];
        float maximum = 0.0f;
        for (int i = 0; i < 32; i++) {
            int column = block * 32 + i;
            int head = column / dimension;
            int d = column - head * dimension;
            values[i] = heads[((size_t)head * rows + row) * dimension + d];
            maximum = fmaxf(maximum, fabsf(values[i]));
        }
        float d = maximum / 127.0f;
        float inverse = d ? 1.0f / d : 0.0f;
        int sigma = 0;
        size_t base = ((size_t)row * attention_blocks + block) * 32;
        for (int i = 0; i < 32; i++) {
            int expected = (int)roundf(values[i] * inverse);
            sigma += expected;
            int actual = heads_q[base + i];
            int delta = abs(actual - expected);
            if (delta) {
                if (q_mismatches < 8)
                    printf("prefill-glue: q8 mismatch row %d block %d lane %d value %.9g max %.9g d %.9g inverse %.9g expected %d actual %d stored-d %.9g\n",
                           row, block, i, values[i], maximum, d, inverse,
                           expected, actual,
                           (float)heads_d[(size_t)row * attention_blocks
                                          + block]);
                q_mismatches++;
            }
            if (delta > q_max_delta) q_max_delta = delta;
            double expected_value = expected * (float)(_Float16)d;
            double actual_value = actual
                                  * (float)heads_d[(size_t)row
                                                   * attention_blocks + block];
            double difference = actual_value - expected_value;
            q_error += difference * difference;
            q_reference += expected_value * expected_value;
        }
        if (heads_s[(size_t)row * attention_blocks + block] != sigma)
            sigma_mismatches++;
    }

    double post_error = 0.0;
    double post_reference = 0.0;
    double finish_error = 0.0;
    double finish_reference = 0.0;
    for (int probe = 0; probe < 64; probe++) {
        int row = (probe * 37 + 11) % rows;
        int column = (probe * 101 + 17) % width;
        double projection_sum = 0.0;
        double dense_sum = 0.0;
        double moe_sum = 0.0;
        for (int d = 0; d < width; d++) {
            size_t index = (size_t)row * width + d;
            projection_sum += (double)projection[index] * projection[index];
            dense_sum += (double)dense[index] * dense[index];
            moe_sum += (double)moe[index] * moe[index];
        }
        float projection_scale =
            1.0f / sqrtf((float)(projection_sum / width) + 1e-6f);
        size_t index = (size_t)row * width + column;
        double expected_post = projection[index] * projection_scale
                               * post_weight[column] + residual[index];
        double post_difference = post_output[index] - expected_post;
        post_error += post_difference * post_difference;
        post_reference += expected_post * expected_post;
        float dense_scale = 1.0f / sqrtf((float)(dense_sum / width) + 1e-6f);
        float moe_scale = 1.0f / sqrtf((float)(moe_sum / width) + 1e-6f);
        double combine_sum = 0.0;
        for (int d = 0; d < width; d++) {
            size_t other = (size_t)row * width + d;
            double value = dense[other] * dense_scale * dense_weight[d]
                           + moe[other] * moe_scale * moe_weight[d];
            combine_sum += value * value;
        }
        float combine_scale =
            1.0f / sqrtf((float)(combine_sum / width) + 1e-6f);
        double combined = dense[index] * dense_scale * dense_weight[column]
                          + moe[index] * moe_scale * moe_weight[column];
        double expected_finish =
            (combined * combine_scale * combine_weight[column]
             + residual[index]) * layer_scale[0];
        double finish_difference = finish_output[index] - expected_finish;
        finish_error += finish_difference * finish_difference;
        finish_reference += expected_finish * expected_finish;
    }
    double post_rel_rms = sqrt(post_error / post_reference);
    double finish_rel_rms = sqrt(finish_error / finish_reference);
    double q_rel_rms = sqrt(q_error / (q_reference + 1e-30));
    printf("prefill-glue: correctness %s heads q-mismatch %d/4096 max-delta %d sigma-mismatch %d/128 q8 rel-rms %.9g post rel-rms %.9g finish rel-rms %.9g\n",
           name ? name : "swa", q_mismatches, q_max_delta,
           sigma_mismatches, q_rel_rms, post_rel_rms, finish_rel_rms);
    if (q_max_delta > 1 || q_rel_rms > 5e-4 || post_rel_rms > 1e-6
        || finish_rel_rms > 1e-6)
        bench_fatal("glue correctness failed");

    const char *variant_names[3] = { "heads-q8", "post-attn", "ffn-finish" };
    for (int variant = 0; variant < 3; variant++) {
        double probe = bench_glue_launch(
            gpu, variant, rows, attention_width, 5) / 5.0;
        int repetitions = (int)ceil(target_seconds / probe);
        if (repetitions < 5) repetitions = 5;
        if (repetitions > 2000) repetitions = 2000;
        double times[BENCH_ROUNDS_MAX];
        double frequencies[BENCH_ROUNDS_MAX];
        for (int round = 0; round < rounds; round++) {
            bench_telemetry telemetry;
            bench_telemetry_start(&telemetry);
            double seconds = bench_glue_launch(
                gpu, variant, rows, attention_width, repetitions);
            bench_telemetry_stop(&telemetry);
            times[round] = seconds / repetitions;
            frequencies[round] = (double)telemetry.actual_sum
                                 / telemetry.samples;
        }
        double median = bench_median(times, rounds);
        double frequency = bench_median(frequencies, rounds);
        printf("prefill-glue: median %s %.6f ms @%.0fMHz model-cost %.6f ms/token\n",
               variant_names[variant], median * 1e3, frequency,
               median * 30.0 / rows * 1e3);
    }

    bench_ze_check("zeMemFree glue layer scale",
                   zeMemFree(gpu->context, layer_scale));
    bench_ze_check("zeMemFree glue finish output",
                   zeMemFree(gpu->context, finish_output));
    bench_ze_check("zeMemFree glue post output",
                   zeMemFree(gpu->context, post_output));
    bench_ze_check("zeMemFree glue combine weight",
                   zeMemFree(gpu->context, combine_weight));
    bench_ze_check("zeMemFree glue moe weight",
                   zeMemFree(gpu->context, moe_weight));
    bench_ze_check("zeMemFree glue dense weight",
                   zeMemFree(gpu->context, dense_weight));
    bench_ze_check("zeMemFree glue post weight",
                   zeMemFree(gpu->context, post_weight));
    bench_ze_check("zeMemFree glue residual",
                   zeMemFree(gpu->context, residual));
    bench_ze_check("zeMemFree glue moe", zeMemFree(gpu->context, moe));
    bench_ze_check("zeMemFree glue dense", zeMemFree(gpu->context, dense));
    bench_ze_check("zeMemFree glue projection",
                   zeMemFree(gpu->context, projection));
    bench_ze_check("zeMemFree glue heads sigma",
                   zeMemFree(gpu->context, heads_s));
    bench_ze_check("zeMemFree glue heads scale",
                   zeMemFree(gpu->context, heads_d));
    bench_ze_check("zeMemFree glue heads q", zeMemFree(gpu->context, heads_q));
    bench_ze_check("zeMemFree glue heads", zeMemFree(gpu->context, heads));
}

static void bench_attention_set_args(bench_gpu *gpu,
                                     const bench_attention_data *data,
                                     const bench_attention_shape *shape) {
    bench_pointer_arg(gpu->attn_qk, 0, data->q);
    bench_pointer_arg(gpu->attn_qk, 1, data->k);
    bench_pointer_arg(gpu->attn_qk, 2, data->scores);
    bench_int_arg(gpu->attn_qk, 3, shape->m);
    bench_int_arg(gpu->attn_qk, 4, shape->n);
    bench_int_arg(gpu->attn_qk, 5, shape->dimension);
    bench_int_arg(gpu->attn_qk, 6, shape->heads);
    bench_int_arg(gpu->attn_qk, 7, shape->kv_heads);
    bench_int_arg(gpu->attn_qk, 8, shape->query_offset);
    bench_int_arg(gpu->attn_qk, 9, shape->window);

    bench_pointer_arg(gpu->attn_softmax, 0, data->scores);
    bench_int_arg(gpu->attn_softmax, 1, shape->heads * shape->m);
    bench_int_arg(gpu->attn_softmax, 2, shape->n);

    bench_pointer_arg(gpu->attn_pv, 0, data->scores);
    bench_pointer_arg(gpu->attn_pv, 1, data->v);
    bench_pointer_arg(gpu->attn_pv, 2, data->materialized);
    bench_int_arg(gpu->attn_pv, 3, shape->m);
    bench_int_arg(gpu->attn_pv, 4, shape->n);
    bench_int_arg(gpu->attn_pv, 5, shape->dimension);
    bench_int_arg(gpu->attn_pv, 6, shape->heads);
    bench_int_arg(gpu->attn_pv, 7, shape->kv_heads);

    bench_pointer_arg(gpu->attn_online, 0, data->q);
    bench_pointer_arg(gpu->attn_online, 1, data->k);
    bench_pointer_arg(gpu->attn_online, 2, data->v);
    bench_pointer_arg(gpu->attn_online, 3, data->online);
    bench_int_arg(gpu->attn_online, 4, shape->m);
    bench_int_arg(gpu->attn_online, 5, shape->n);
    bench_int_arg(gpu->attn_online, 6, shape->dimension);
    bench_int_arg(gpu->attn_online, 7, shape->heads);
    bench_int_arg(gpu->attn_online, 8, shape->kv_heads);
    bench_int_arg(gpu->attn_online, 9, shape->query_offset);
    bench_int_arg(gpu->attn_online, 10, shape->window);

    bench_pointer_arg(gpu->attn_online_b4, 0, data->q);
    bench_pointer_arg(gpu->attn_online_b4, 1, data->k);
    bench_pointer_arg(gpu->attn_online_b4, 2, data->v);
    bench_pointer_arg(gpu->attn_online_b4, 3, data->online_b4);
    bench_int_arg(gpu->attn_online_b4, 4, shape->m);
    bench_int_arg(gpu->attn_online_b4, 5, shape->n);
    bench_int_arg(gpu->attn_online_b4, 6, shape->dimension);
    bench_int_arg(gpu->attn_online_b4, 7, shape->heads);
    bench_int_arg(gpu->attn_online_b4, 8, shape->kv_heads);
    bench_int_arg(gpu->attn_online_b4, 9, shape->query_offset);
    bench_int_arg(gpu->attn_online_b4, 10, shape->window);

    bench_pointer_arg(gpu->attn_online_b8, 0, data->q);
    bench_pointer_arg(gpu->attn_online_b8, 1, data->k);
    bench_pointer_arg(gpu->attn_online_b8, 2, data->v);
    bench_pointer_arg(gpu->attn_online_b8, 3, data->online_b8);
    bench_int_arg(gpu->attn_online_b8, 4, shape->m);
    bench_int_arg(gpu->attn_online_b8, 5, shape->n);
    bench_int_arg(gpu->attn_online_b8, 6, shape->dimension);
    bench_int_arg(gpu->attn_online_b8, 7, shape->heads);
    bench_int_arg(gpu->attn_online_b8, 8, shape->kv_heads);
    bench_int_arg(gpu->attn_online_b8, 9, shape->query_offset);
    bench_int_arg(gpu->attn_online_b8, 10, shape->window);

    int stage_base = shape->window > 0
        ? shape->query_offset - shape->window + 1 : 0;
    if (stage_base < 0) stage_base = 0;
    int ring_capacity = shape->n - stage_base;
    bench_pointer_arg(gpu->attn_online_b8_ring, 0, data->q);
    bench_pointer_arg(gpu->attn_online_b8_ring, 1, data->ring_k);
    bench_pointer_arg(gpu->attn_online_b8_ring, 2, data->ring_v);
    bench_pointer_arg(gpu->attn_online_b8_ring, 3, data->batch_k);
    bench_pointer_arg(gpu->attn_online_b8_ring, 4, data->batch_v);
    bench_pointer_arg(gpu->attn_online_b8_ring, 5, data->online_b8_ring);
    bench_int_arg(gpu->attn_online_b8_ring, 6, shape->m);
    bench_int_arg(gpu->attn_online_b8_ring, 7, shape->n);
    bench_int_arg(gpu->attn_online_b8_ring, 8, shape->dimension);
    bench_int_arg(gpu->attn_online_b8_ring, 9, shape->heads);
    bench_int_arg(gpu->attn_online_b8_ring, 10, shape->kv_heads);
    bench_int_arg(gpu->attn_online_b8_ring, 11, shape->query_offset);
    bench_int_arg(gpu->attn_online_b8_ring, 12, shape->window);
    bench_int_arg(gpu->attn_online_b8_ring, 13, ring_capacity);

    bench_pointer_arg(gpu->attn_gqa8, 0, data->q);
    bench_pointer_arg(gpu->attn_gqa8, 1, data->k);
    bench_pointer_arg(gpu->attn_gqa8, 2, data->v);
    bench_pointer_arg(gpu->attn_gqa8, 3, data->gqa8);
    bench_int_arg(gpu->attn_gqa8, 4, shape->m);
    bench_int_arg(gpu->attn_gqa8, 5, shape->n);
    bench_int_arg(gpu->attn_gqa8, 6, shape->dimension);
    bench_int_arg(gpu->attn_gqa8, 7, shape->heads);
    bench_int_arg(gpu->attn_gqa8, 8, shape->kv_heads);
    bench_int_arg(gpu->attn_gqa8, 9, shape->query_offset);
    bench_int_arg(gpu->attn_gqa8, 10, shape->window);
}

static void bench_attention_data_init(bench_gpu *gpu,
                                      bench_attention_data *data,
                                      const bench_attention_shape *shape) {
    size_t q_elements = (size_t)shape->heads * shape->m * shape->dimension;
    size_t kv_elements = (size_t)shape->kv_heads * shape->n * shape->dimension;
    int stage_base = shape->window > 0
        ? shape->query_offset - shape->window + 1 : 0;
    if (stage_base < 0) stage_base = 0;
    int ring_capacity = shape->n - stage_base;
    int physical_capacity = shape->window > 0 ? shape->window : shape->n;
    size_t ring_elements =
        (size_t)shape->kv_heads * ring_capacity * shape->dimension;
    size_t batch_elements =
        (size_t)shape->kv_heads * shape->m * shape->dimension;
    size_t score_elements = (size_t)shape->heads * shape->m * shape->n;
    data->q = bench_gpu_alloc(gpu, q_elements * sizeof(float));
    data->k = bench_gpu_alloc(gpu, kv_elements * sizeof(_Float16));
    data->v = bench_gpu_alloc(gpu, kv_elements * sizeof(_Float16));
    data->ring_k = bench_gpu_alloc(gpu, ring_elements * sizeof(_Float16));
    data->ring_v = bench_gpu_alloc(gpu, ring_elements * sizeof(_Float16));
    data->physical_k = bench_gpu_alloc(
        gpu, (size_t)shape->kv_heads * physical_capacity * shape->dimension
             * sizeof(_Float16));
    data->physical_v = bench_gpu_alloc(
        gpu, (size_t)shape->kv_heads * physical_capacity * shape->dimension
             * sizeof(_Float16));
    data->batch_k = bench_gpu_alloc(gpu, batch_elements * sizeof(_Float16));
    data->batch_v = bench_gpu_alloc(gpu, batch_elements * sizeof(_Float16));
    data->scores = bench_gpu_alloc(gpu, score_elements * sizeof(float));
    data->materialized = bench_gpu_alloc(gpu, q_elements * sizeof(float));
    data->online = bench_gpu_alloc(gpu, q_elements * sizeof(float));
    data->online_b4 = bench_gpu_alloc(gpu, q_elements * sizeof(float));
    data->online_b8 = bench_gpu_alloc(gpu, q_elements * sizeof(float));
    data->online_b8_ring = bench_gpu_alloc(gpu, q_elements * sizeof(float));
    data->gqa8 = bench_gpu_alloc(gpu, q_elements * sizeof(float));
    uint32_t random = UINT32_C(0x5be0cd19) ^ (uint32_t)shape->dimension;
    for (size_t i = 0; i < q_elements; i++) {
        int value = (int)(bench_random(&random) % 2001) - 1000;
        data->q[i] = (float)value / 20000.0f;
    }
    for (size_t i = 0; i < kv_elements; i++) {
        int key = (int)(bench_random(&random) % 2001) - 1000;
        int value = (int)(bench_random(&random) % 2001) - 1000;
        data->k[i] = (_Float16)((float)key / 20000.0f);
        data->v[i] = shape->kv_heads == 2 ? data->k[i]
                                          : (_Float16)((float)value / 1000.0f);
    }
    memset(data->ring_k, 0, ring_elements * sizeof(*data->ring_k));
    memset(data->ring_v, 0, ring_elements * sizeof(*data->ring_v));
    memset(data->physical_k, 0,
           (size_t)shape->kv_heads * physical_capacity * shape->dimension
           * sizeof(*data->physical_k));
    memset(data->physical_v, 0,
           (size_t)shape->kv_heads * physical_capacity * shape->dimension
           * sizeof(*data->physical_v));
    for (int head = 0; head < shape->kv_heads; head++) {
        for (int key = 0; key < shape->query_offset; key++) {
            size_t source = ((size_t)head * shape->n + key) * shape->dimension;
            size_t target = ((size_t)head * physical_capacity
                             + key % physical_capacity) * shape->dimension;
            memcpy(data->physical_k + target, data->k + source,
                   (size_t)shape->dimension * sizeof(*data->physical_k));
            memcpy(data->physical_v + target, data->v + source,
                   (size_t)shape->dimension * sizeof(*data->physical_v));
        }
        for (int key = stage_base; key < shape->n; key++) {
            size_t source = ((size_t)head * shape->n + key) * shape->dimension;
            size_t target = ((size_t)head * ring_capacity
                             + key - stage_base) * shape->dimension;
            memcpy(data->ring_k + target, data->k + source,
                   (size_t)shape->dimension * sizeof(*data->ring_k));
            memcpy(data->ring_v + target, data->v + source,
                   (size_t)shape->dimension * sizeof(*data->ring_v));
        }
        for (int query = 0; query < shape->m; query++) {
            size_t source = ((size_t)head * shape->n
                             + shape->query_offset + query) * shape->dimension;
            size_t target = ((size_t)head * shape->m + query) * shape->dimension;
            memcpy(data->batch_k + target, data->k + source,
                   (size_t)shape->dimension * sizeof(*data->batch_k));
            memcpy(data->batch_v + target, data->v + source,
                   (size_t)shape->dimension * sizeof(*data->batch_v));
        }
    }
    bench_pointer_arg(gpu->swa_stage, 0, data->physical_k);
    bench_pointer_arg(gpu->swa_stage, 1, data->physical_v);
    bench_pointer_arg(gpu->swa_stage, 2, data->batch_k);
    bench_pointer_arg(gpu->swa_stage, 3, data->batch_v);
    bench_pointer_arg(gpu->swa_stage, 4, data->ring_k);
    bench_pointer_arg(gpu->swa_stage, 5, data->ring_v);
    bench_int_arg(gpu->swa_stage, 6, shape->m);
    bench_int_arg(gpu->swa_stage, 7, shape->dimension);
    bench_int_arg(gpu->swa_stage, 8, shape->kv_heads);
    bench_int_arg(gpu->swa_stage, 9, shape->query_offset);
    bench_int_arg(gpu->swa_stage, 10, stage_base);
    bench_int_arg(gpu->swa_stage, 11, ring_capacity);
    bench_int_arg(gpu->swa_stage, 12, physical_capacity);
    bench_pointer_arg(gpu->swa_commit, 0, data->physical_k);
    bench_pointer_arg(gpu->swa_commit, 1, data->physical_v);
    bench_pointer_arg(gpu->swa_commit, 2, data->batch_k);
    bench_pointer_arg(gpu->swa_commit, 3, data->batch_v);
    bench_int_arg(gpu->swa_commit, 4, shape->m);
    bench_int_arg(gpu->swa_commit, 5, shape->dimension);
    bench_int_arg(gpu->swa_commit, 6, shape->kv_heads);
    bench_int_arg(gpu->swa_commit, 7, shape->query_offset);
    bench_int_arg(gpu->swa_commit, 8, physical_capacity);
    bench_attention_set_args(gpu, data, shape);
}

static void bench_attention_data_destroy(bench_gpu *gpu,
                                         bench_attention_data *data) {
    bench_ze_check("zeMemFree attention gqa8",
                   zeMemFree(gpu->context, data->gqa8));
    bench_ze_check("zeMemFree attention online",
                   zeMemFree(gpu->context, data->online));
    bench_ze_check("zeMemFree attention online b4",
                   zeMemFree(gpu->context, data->online_b4));
    bench_ze_check("zeMemFree attention online b8",
                   zeMemFree(gpu->context, data->online_b8));
    bench_ze_check("zeMemFree attention online b8 ring",
                   zeMemFree(gpu->context, data->online_b8_ring));
    bench_ze_check("zeMemFree attention materialized",
                   zeMemFree(gpu->context, data->materialized));
    bench_ze_check("zeMemFree attention scores",
                   zeMemFree(gpu->context, data->scores));
    bench_ze_check("zeMemFree attention v", zeMemFree(gpu->context, data->v));
    bench_ze_check("zeMemFree attention k", zeMemFree(gpu->context, data->k));
    bench_ze_check("zeMemFree attention ring v",
                   zeMemFree(gpu->context, data->ring_v));
    bench_ze_check("zeMemFree attention ring k",
                   zeMemFree(gpu->context, data->ring_k));
    bench_ze_check("zeMemFree attention physical v",
                   zeMemFree(gpu->context, data->physical_v));
    bench_ze_check("zeMemFree attention physical k",
                   zeMemFree(gpu->context, data->physical_k));
    bench_ze_check("zeMemFree attention batch v",
                   zeMemFree(gpu->context, data->batch_v));
    bench_ze_check("zeMemFree attention batch k",
                   zeMemFree(gpu->context, data->batch_k));
    bench_ze_check("zeMemFree attention q", zeMemFree(gpu->context, data->q));
}

static double bench_attention_launch(bench_gpu *gpu, int variant,
                                     const bench_attention_shape *shape,
                                     int repetitions) {
    size_t score_subgroups = (size_t)shape->heads * shape->m * shape->n;
    size_t row_subgroups = (size_t)shape->heads * shape->m;
    ze_group_count_t qk_groups = { (uint32_t)((score_subgroups + 7) / 8), 1, 1 };
    ze_group_count_t row_groups = { (uint32_t)((row_subgroups + 7) / 8), 1, 1 };
    ze_group_count_t pv_groups = { (uint32_t)row_subgroups, 1, 1 };
    ze_group_count_t gqa_groups = { (uint32_t)(shape->kv_heads * shape->m),
                                    1, 1 };
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        if (variant == BENCH_ATTN_MATERIALIZED) {
            bench_ze_check("zeCommandListAppendLaunchKernel attention qk",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->attn_qk,
                                                           &qk_groups,
                                                           NULL, 0, NULL));
            bench_ze_check("zeCommandListAppendLaunchKernel attention softmax",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->attn_softmax,
                                                           &row_groups,
                                                           NULL, 0, NULL));
            bench_ze_check("zeCommandListAppendLaunchKernel attention pv",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->attn_pv,
                                                           &pv_groups,
                                                           NULL, 0, NULL));
        } else if (variant == BENCH_ATTN_ONLINE) {
            bench_ze_check("zeCommandListAppendLaunchKernel attention online",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->attn_online,
                                                           &row_groups,
                                                           NULL, 0, NULL));
        } else if (variant == BENCH_ATTN_ONLINE_B4) {
            bench_ze_check("zeCommandListAppendLaunchKernel attention online b4",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->attn_online_b4,
                                                           &row_groups,
                                                           NULL, 0, NULL));
        } else if (variant == BENCH_ATTN_ONLINE_B8) {
            bench_ze_check("zeCommandListAppendLaunchKernel attention online b8",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->attn_online_b8,
                                                           &row_groups,
                                                           NULL, 0, NULL));
        } else if (variant == BENCH_ATTN_ONLINE_B8_RING) {
            bench_ze_check("zeCommandListAppendLaunchKernel attention online b8 stage",
                           zeCommandListAppendLaunchKernel(
                               gpu->commands, gpu->attn_online_b8_ring,
                               &row_groups, NULL, 0, NULL));
        } else {
            bench_ze_check("zeCommandListAppendLaunchKernel attention gqa8",
                           zeCommandListAppendLaunchKernel(gpu->commands,
                                                           gpu->attn_gqa8,
                                                           &gqa_groups,
                                                           NULL, 0, NULL));
        }
    }
    bench_ze_check("zeCommandListHostSynchronize attention",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static double bench_swa_stage_launch(bench_gpu *gpu,
                                     const bench_attention_shape *shape,
                                     int repetitions) {
    int stage_base = shape->query_offset - shape->window + 1;
    if (stage_base < 0) stage_base = 0;
    size_t elements = (size_t)shape->kv_heads * (shape->n - stage_base)
                      * shape->dimension;
    ze_group_count_t groups = { (uint32_t)((elements + 255) / 256), 1, 1 };
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        bench_ze_check("zeCommandListAppendLaunchKernel SWA stage",
                       zeCommandListAppendLaunchKernel(gpu->commands,
                                                       gpu->swa_stage, &groups,
                                                       NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize SWA stage",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static double bench_swa_commit_launch(bench_gpu *gpu,
                                      const bench_attention_shape *shape,
                                      int repetitions) {
    size_t elements =
        (size_t)shape->kv_heads * shape->m * shape->dimension;
    ze_group_count_t groups = { (uint32_t)((elements + 255) / 256), 1, 1 };
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        bench_ze_check("zeCommandListAppendLaunchKernel SWA commit",
                       zeCommandListAppendLaunchKernel(gpu->commands,
                                                       gpu->swa_commit, &groups,
                                                       NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize SWA commit",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static bench_sample bench_attention_measure(bench_gpu *gpu, int variant,
                                            const bench_attention_shape *shape,
                                            int repetitions) {
    bench_telemetry telemetry;
    bench_telemetry_start(&telemetry);
    double seconds = bench_attention_launch(gpu, variant, shape, repetitions);
    bench_telemetry_stop(&telemetry);
    return (bench_sample) {
        .seconds = seconds / repetitions,
        .actual_mhz = (double)telemetry.actual_sum / telemetry.samples,
        .requested_mhz = (double)telemetry.requested_sum / telemetry.samples,
        .actual_min = telemetry.actual_min,
        .actual_max = telemetry.actual_max,
        .pl1 = telemetry.pl1,
        .pl2 = telemetry.pl2,
        .thermal = telemetry.thermal
    };
}

static void bench_attention_verify(const bench_attention_data *data,
                                   const bench_attention_shape *shape) {
    size_t elements = (size_t)shape->heads * shape->m * shape->dimension;
    const float *outputs[5] = {
        data->online, data->online_b4, data->online_b8,
        data->online_b8_ring, data->gqa8
    };
    const char *names[5] = {
        "online", "online-b4", "online-b8", "online-b8-stage", "gqa8"
    };
    for (int variant = 0; variant < 5; variant++) {
        if (variant == 3 && shape->window == 0) continue;
        if (variant == 4 && shape->kv_heads != 2) continue;
        double error = 0.0;
        double reference = 0.0;
        double max_abs = 0.0;
        for (size_t i = 0; i < elements; i++) {
            double difference = (double)outputs[variant][i]
                                - data->materialized[i];
            error += difference * difference;
            reference += (double)data->materialized[i] * data->materialized[i];
            if (fabs(difference) > max_abs) max_abs = fabs(difference);
        }
        double rel_rms = sqrt(error / (reference + 1e-30));
        printf("prefill-attention: correctness %s rel-rms %.9g max-abs %.9g elements %zu\n",
               names[variant], rel_rms, max_abs, elements);
        if (!isfinite(rel_rms) || rel_rms > 2e-5)
            bench_fatal("attention %s correctness failed", names[variant]);
    }
}

static void bench_attention_shape_run(bench_gpu *gpu,
                                      const bench_attention_shape *shape,
                                      int rounds, double target_seconds) {
    static const char *names[BENCH_ATTN_VARIANTS] = {
        "materialized", "online", "online-b4", "online-b8",
        "online-b8-stage", "gqa8"
    };
    int variants = shape->window > 0 ? 5 : 4;
    bench_attention_data data = {0};
    bench_attention_data_init(gpu, &data, shape);
    size_t score_bytes = (size_t)shape->heads * shape->m * shape->n * sizeof(float);
    printf("prefill-attention: shape %s M %d N %d D %d H %d KVH %d offset %d window %d scores %.3f MiB\n",
           shape->name, shape->m, shape->n, shape->dimension, shape->heads,
           shape->kv_heads, shape->query_offset, shape->window,
           (double)score_bytes / (1024.0 * 1024.0));
    bench_attention_launch(gpu, BENCH_ATTN_MATERIALIZED, shape, 1);
    bench_attention_launch(gpu, BENCH_ATTN_ONLINE, shape, 1);
    bench_attention_launch(gpu, BENCH_ATTN_ONLINE_B4, shape, 1);
    bench_attention_launch(gpu, BENCH_ATTN_ONLINE_B8, shape, 1);
    if (shape->window > 0) {
        bench_swa_stage_launch(gpu, shape, 1);
        bench_attention_launch(gpu, BENCH_ATTN_ONLINE_B8_RING, shape, 1);
    }
    if (shape->kv_heads == 2)
        bench_attention_launch(gpu, BENCH_ATTN_GQA8, shape, 1);
    bench_attention_verify(&data, shape);
    double probe = bench_attention_launch(gpu, BENCH_ATTN_MATERIALIZED,
                                          shape, 2) / 2.0;
    int repetitions = (int)ceil(target_seconds / probe);
    if (repetitions < 3) repetitions = 3;
    bench_sample samples[BENCH_ATTN_VARIANTS][BENCH_ROUNDS_MAX];
    double times[BENCH_ATTN_VARIANTS][BENCH_ROUNDS_MAX];
    double frequencies[BENCH_ATTN_VARIANTS][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        for (int position = 0; position < variants; position++) {
            int variant = (round + position) % variants;
            samples[variant][round] =
                bench_attention_measure(gpu, variant, shape, repetitions);
            times[variant][round] = samples[variant][round].seconds;
            frequencies[variant][round] = samples[variant][round].actual_mhz;
        }
        printf("prefill-attention: round %d", round + 1);
        for (int variant = 0; variant < variants; variant++) {
            bench_sample *sample = &samples[variant][round];
            printf(" %s %.6f ms@%.0fMHz[%ld,%ld] throttle=%ld/%ld/%ld",
                   names[variant], sample->seconds * 1e3, sample->actual_mhz,
                   sample->actual_min, sample->actual_max, sample->pl1,
                   sample->pl2, sample->thermal);
        }
        printf("\n");
    }
    double materialized = bench_median(times[BENCH_ATTN_MATERIALIZED], rounds);
    double online = bench_median(times[BENCH_ATTN_ONLINE], rounds);
    double online_b4 = bench_median(times[BENCH_ATTN_ONLINE_B4], rounds);
    double online_b8 = bench_median(times[BENCH_ATTN_ONLINE_B8], rounds);
    double online_b8_ring = shape->window > 0
        ? bench_median(times[BENCH_ATTN_ONLINE_B8_RING], rounds) : online_b8;
    double materialized_frequency =
        bench_median(frequencies[BENCH_ATTN_MATERIALIZED], rounds);
    double online_frequency = bench_median(frequencies[BENCH_ATTN_ONLINE], rounds);
    double online_b4_frequency =
        bench_median(frequencies[BENCH_ATTN_ONLINE_B4], rounds);
    double online_b8_frequency =
        bench_median(frequencies[BENCH_ATTN_ONLINE_B8], rounds);
    double online_b8_ring_frequency = shape->window > 0
        ? bench_median(frequencies[BENCH_ATTN_ONLINE_B8_RING], rounds)
        : online_b8_frequency;
    int64_t visible = 0;
    for (int query = 0; query < shape->m; query++) {
        int position = shape->query_offset + query;
        int first = shape->window > 0 ? position - shape->window + 1 : 0;
        if (first < 0) first = 0;
        visible += position - first + 1;
    }
    double operations = 4.0 * shape->heads * visible * shape->dimension;
    double frequency_max = fmax(materialized_frequency,
                           fmax(online_frequency,
                           fmax(online_b4_frequency, online_b8_frequency)));
    double frequency_min = fmin(materialized_frequency,
                           fmin(online_frequency,
                           fmin(online_b4_frequency, online_b8_frequency)));
    if (shape->window > 0) {
        frequency_max = fmax(frequency_max, online_b8_ring_frequency);
        frequency_min = fmin(frequency_min, online_b8_ring_frequency);
    }
    double frequency_span = frequency_max / frequency_min;
    const char *decision = materialized <= online
                           && materialized <= online_b4
                           && materialized <= online_b8
                           ? "materialized-go"
                           : online <= online_b4 && online <= online_b8
                           ? "online-go"
                           : online_b4 <= online_b8 ? "online-b4-go"
                           : "online-b8-go";
    printf("prefill-attention: median materialized %.6f ms %.6f TFLOP/s @%.0fMHz online %.6f ms %.6f TFLOP/s @%.0fMHz online-b4 %.6f ms %.6f TFLOP/s @%.0fMHz online-b8 %.6f ms %.6f TFLOP/s @%.0fMHz online-b8-stage %.6f ms @%.0fMHz stage-cost %.6fx frequency-span %.6fx decision %s\n",
           materialized * 1e3, operations / materialized / 1e12,
           materialized_frequency, online * 1e3, operations / online / 1e12,
           online_frequency, online_b4 * 1e3,
           operations / online_b4 / 1e12, online_b4_frequency,
           online_b8 * 1e3,
           operations / online_b8 / 1e12, online_b8_frequency,
           online_b8_ring * 1e3, online_b8_ring_frequency,
           online_b8_ring / online_b8, frequency_span,
           frequency_span <= 1.05 ? decision : "frequency-reject");
    if (shape->window > 0) {
        double stage_probe = bench_swa_stage_launch(gpu, shape, 2) / 2.0;
        int stage_repetitions = (int)ceil(target_seconds / stage_probe);
        if (stage_repetitions < 3) stage_repetitions = 3;
        double stage_times[BENCH_ROUNDS_MAX];
        for (int round = 0; round < rounds; round++)
            stage_times[round] = bench_swa_stage_launch(
                gpu, shape, stage_repetitions) / stage_repetitions;
        double stage = bench_median(stage_times, rounds);
        bench_swa_commit_launch(gpu, shape, 1);
        int capacity = shape->window;
        for (int head = 0; head < shape->kv_heads; head++)
            for (int query = 0; query < shape->m; query++) {
                size_t source = ((size_t)head * shape->m + query)
                                * shape->dimension;
                size_t target = ((size_t)head * capacity
                                 + ((shape->query_offset + query)
                                    & (capacity - 1))) * shape->dimension;
                size_t bytes_per_key =
                    (size_t)shape->dimension * sizeof(*data.batch_k);
                if (memcmp(data.physical_k + target, data.batch_k + source,
                           bytes_per_key)
                    || memcmp(data.physical_v + target, data.batch_v + source,
                              bytes_per_key))
                    bench_fatal("SWA commit correctness failed");
            }
        double commit_probe = bench_swa_commit_launch(gpu, shape, 2) / 2.0;
        int commit_repetitions = (int)ceil(target_seconds / commit_probe);
        if (commit_repetitions < 3) commit_repetitions = 3;
        double commit_times[BENCH_ROUNDS_MAX];
        for (int round = 0; round < rounds; round++)
            commit_times[round] = bench_swa_commit_launch(
                gpu, shape, commit_repetitions) / commit_repetitions;
        double commit = bench_median(commit_times, rounds);
        int stage_base = shape->query_offset - shape->window + 1;
        if (stage_base < 0) stage_base = 0;
        double bytes = 4.0 * shape->kv_heads * (shape->n - stage_base)
                       * shape->dimension;
        double commit_bytes = 4.0 * shape->kv_heads * shape->m
                              * shape->dimension;
        printf("prefill-attention: median swa-stage %.6f ms %.3f GB/s swa-commit %.6f ms %.3f GB/s total-with-b8 %.6f ms copy-overhead %.3f%% commit-correct exact\n",
               stage * 1e3, bytes / stage / 1e9,
               commit * 1e3, commit_bytes / commit / 1e9,
               (stage + online_b8_ring + commit) * 1e3,
               100.0 * (stage + commit) / online_b8);
    }
    bench_attention_data_destroy(gpu, &data);
}

static void bench_attention_run(bench_gpu *gpu, const char *selected_shape,
                                int rounds, double target_seconds) {
    const bench_attention_shape shapes[] = {
        { "swa-initial", 512, 512, 256, 16, 8, 0, 1024 },
        { "swa", 512, 1024, 256, 16, 8, 512, 1024 },
        { "swa-wrap", 256, 1280, 256, 16, 8, 1024, 1024 },
        { "global", 512, 512, 512, 16, 2, 0, 0 },
        { "global1024", 512, 1024, 512, 16, 2, 512, 0 },
        { "global1536", 512, 1536, 512, 16, 2, 1024, 0 },
        { "global2048", 512, 2048, 512, 16, 2, 1536, 0 },
        { "global4096", 512, 4096, 512, 16, 2, 3584, 0 },
        { "global8192", 128, 8192, 512, 16, 2, 8064, 0 }
    };
    int matched = 0;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
        if (selected_shape && strcmp(selected_shape, shapes[i].name)) continue;
        bench_attention_shape_run(gpu, &shapes[i], rounds, target_seconds);
        matched = 1;
    }
    if (!matched) bench_fatal("unknown attention shape: %s", selected_shape);
}

static void bench_attention_long_run(bench_gpu *gpu, int rows, int keys,
                                     int chunk_size) {
    const int dimension = 512;
    const int heads = 16;
    const int kv_heads = 2;
    size_t q_elements = (size_t)heads * rows * dimension;
    size_t kv_elements = (size_t)kv_heads * keys * dimension;
    size_t score_elements = (size_t)heads * rows * keys;
    float *q = bench_gpu_alloc(gpu, q_elements * sizeof(*q));
    _Float16 *kv = bench_gpu_alloc(gpu, kv_elements * sizeof(*kv));
    float *scores = bench_gpu_alloc(gpu, score_elements * sizeof(*scores));
    float *materialized = bench_gpu_alloc(
        gpu, q_elements * sizeof(*materialized));
    float *online = bench_gpu_alloc(gpu, q_elements * sizeof(*online));
    float *scalar = bench_gpu_alloc(gpu, q_elements * sizeof(*scalar));
    const int partial_stride = 528;
    int max_chunks = (keys + chunk_size - 1) / chunk_size;
    size_t partial_elements = (size_t)heads * rows * max_chunks
                              * partial_stride;
    float *partial = bench_gpu_alloc(gpu, partial_elements * sizeof(*partial));
    float *hierarchical = bench_gpu_alloc(
        gpu, q_elements * sizeof(*hierarchical));
    float *query_major = bench_gpu_alloc(
        gpu, q_elements * sizeof(*query_major));
    for (size_t index = 0; index < q_elements; index++)
        q[index] = (float)((int)(index * 17 % 257) - 128) / 4096.0f;

    bench_pointer_arg(gpu->attn_long_init, 0, kv);
    uint64_t init_elements = kv_elements;
    bench_ze_check("zeKernelSetArgumentValue attention long init elements",
                   zeKernelSetArgumentValue(gpu->attn_long_init, 1,
                                            sizeof(init_elements),
                                            &init_elements));
    ze_group_count_t init_groups = {
        (uint32_t)((kv_elements + 255) / 256), 1, 1
    };
    bench_ze_check("zeCommandListAppendLaunchKernel attention long init",
                   zeCommandListAppendLaunchKernel(gpu->commands,
                                                   gpu->attn_long_init,
                                                   &init_groups,
                                                   NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize attention long init",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));

    bench_pointer_arg(gpu->attn_qk, 0, q);
    bench_pointer_arg(gpu->attn_qk, 1, kv);
    bench_pointer_arg(gpu->attn_qk, 2, scores);
    bench_int_arg(gpu->attn_qk, 3, rows);
    bench_int_arg(gpu->attn_qk, 4, keys);
    bench_int_arg(gpu->attn_qk, 5, dimension);
    bench_int_arg(gpu->attn_qk, 6, heads);
    bench_int_arg(gpu->attn_qk, 7, kv_heads);
    bench_int_arg(gpu->attn_qk, 8, keys - rows);
    bench_int_arg(gpu->attn_qk, 9, 0);
    bench_pointer_arg(gpu->attn_softmax, 0, scores);
    bench_int_arg(gpu->attn_softmax, 1, heads * rows);
    bench_int_arg(gpu->attn_softmax, 2, keys);
    bench_pointer_arg(gpu->attn_pv4, 0, scores);
    bench_pointer_arg(gpu->attn_pv4, 1, kv);
    bench_pointer_arg(gpu->attn_pv4, 2, materialized);
    bench_int_arg(gpu->attn_pv4, 3, rows);
    bench_int_arg(gpu->attn_pv4, 4, keys);
    bench_int_arg(gpu->attn_pv4, 5, dimension);
    bench_int_arg(gpu->attn_pv4, 6, heads);
    bench_int_arg(gpu->attn_pv4, 7, kv_heads);
    bench_pointer_arg(gpu->attn_online_b8, 0, q);
    bench_pointer_arg(gpu->attn_online_b8, 1, kv);
    bench_pointer_arg(gpu->attn_online_b8, 2, kv);
    bench_pointer_arg(gpu->attn_online_b8, 3, online);
    bench_int_arg(gpu->attn_online_b8, 4, rows);
    bench_int_arg(gpu->attn_online_b8, 5, keys);
    bench_int_arg(gpu->attn_online_b8, 6, dimension);
    bench_int_arg(gpu->attn_online_b8, 7, heads);
    bench_int_arg(gpu->attn_online_b8, 8, kv_heads);
    bench_int_arg(gpu->attn_online_b8, 9, keys - rows);
    bench_int_arg(gpu->attn_online_b8, 10, 0);
    bench_pointer_arg(gpu->attn_online, 0, q);
    bench_pointer_arg(gpu->attn_online, 1, kv);
    bench_pointer_arg(gpu->attn_online, 2, kv);
    bench_pointer_arg(gpu->attn_online, 3, scalar);
    bench_int_arg(gpu->attn_online, 4, rows);
    bench_int_arg(gpu->attn_online, 5, keys);
    bench_int_arg(gpu->attn_online, 6, dimension);
    bench_int_arg(gpu->attn_online, 7, heads);
    bench_int_arg(gpu->attn_online, 8, kv_heads);
    bench_int_arg(gpu->attn_online, 9, keys - rows);
    bench_int_arg(gpu->attn_online, 10, 0);
    bench_pointer_arg(gpu->attn_partial_b8, 0, q);
    bench_pointer_arg(gpu->attn_partial_b8, 1, kv);
    bench_pointer_arg(gpu->attn_partial_b8, 2, kv);
    bench_pointer_arg(gpu->attn_partial_b8, 3, partial);
    bench_int_arg(gpu->attn_partial_b8, 4, rows);
    bench_int_arg(gpu->attn_partial_b8, 5, keys);
    bench_int_arg(gpu->attn_partial_b8, 6, dimension);
    bench_int_arg(gpu->attn_partial_b8, 7, heads);
    bench_int_arg(gpu->attn_partial_b8, 8, kv_heads);
    bench_int_arg(gpu->attn_partial_b8, 9, keys - rows);
    bench_int_arg(gpu->attn_partial_b8, 10, chunk_size);
    bench_int_arg(gpu->attn_partial_b8, 11, max_chunks);
    bench_int_arg(gpu->attn_partial_b8, 12, partial_stride);
    bench_int_arg(gpu->attn_partial_b8, 13, 0);
    bench_pointer_arg(gpu->attn_partial_merge, 0, partial);
    bench_pointer_arg(gpu->attn_partial_merge, 1, hierarchical);
    bench_int_arg(gpu->attn_partial_merge, 2, rows);
    bench_int_arg(gpu->attn_partial_merge, 3, dimension);
    bench_int_arg(gpu->attn_partial_merge, 4, heads);
    bench_int_arg(gpu->attn_partial_merge, 5, keys - rows);
    bench_int_arg(gpu->attn_partial_merge, 6, chunk_size);
    bench_int_arg(gpu->attn_partial_merge, 7, max_chunks);
    bench_int_arg(gpu->attn_partial_merge, 8, partial_stride);

    size_t score_subgroups = (size_t)heads * rows * keys;
    size_t row_subgroups = (size_t)heads * rows;
    ze_group_count_t qk_groups = {
        (uint32_t)((score_subgroups + 7) / 8), 1, 1
    };
    ze_group_count_t row_groups = {
        (uint32_t)((row_subgroups + 7) / 8), 1, 1
    };
    ze_group_count_t pv_groups = { (uint32_t)row_subgroups, 1, 1 };
    size_t partial_subgroups = row_subgroups * max_chunks;
    ze_group_count_t partial_groups = {
        (uint32_t)((partial_subgroups + 7) / 8), 1, 1
    };
    double times[3];
    bench_telemetry telemetry[3];
    printf("prefill-attention-long: M %d N %d chunk %d scores %.3f MiB partial %.3f MiB begin\n",
           rows, keys, chunk_size, (double)(score_elements * sizeof(*scores))
                       / (1024.0 * 1024.0),
           (double)(partial_elements * sizeof(*partial)) / (1024.0 * 1024.0));
    bench_telemetry_start(&telemetry[0]);
    double start = bench_now();
    bench_ze_check("zeCommandListAppendLaunchKernel long qk",
                   zeCommandListAppendLaunchKernel(gpu->commands, gpu->attn_qk,
                                                   &qk_groups, NULL, 0, NULL));
    bench_ze_check("zeCommandListAppendLaunchKernel long softmax",
                   zeCommandListAppendLaunchKernel(gpu->commands,
                                                   gpu->attn_softmax,
                                                   &row_groups, NULL, 0, NULL));
    bench_ze_check("zeCommandListAppendLaunchKernel long pv",
                   zeCommandListAppendLaunchKernel(gpu->commands, gpu->attn_pv4,
                                                   &pv_groups, NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize attention long materialized",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    times[0] = bench_now() - start;
    bench_telemetry_stop(&telemetry[0]);
    printf("prefill-attention-long: materialized %.6f s\n", times[0]);
    bench_telemetry_start(&telemetry[1]);
    start = bench_now();
    bench_ze_check("zeCommandListAppendLaunchKernel long b8",
                   zeCommandListAppendLaunchKernel(gpu->commands,
                                                   gpu->attn_online_b8,
                                                   &row_groups, NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize attention long b8",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    times[1] = bench_now() - start;
    bench_telemetry_stop(&telemetry[1]);
    bench_pointer_arg(gpu->attn_partial_merge, 1, query_major);
    start = bench_now();
    bench_ze_check("zeCommandListAppendLaunchKernel long partial query major",
                   zeCommandListAppendLaunchKernel(gpu->commands,
                                                   gpu->attn_partial_b8,
                                                   &partial_groups,
                                                   NULL, 0, NULL));
    bench_ze_check("zeCommandListAppendLaunchKernel long query major merge",
                   zeCommandListAppendLaunchKernel(gpu->commands,
                                                   gpu->attn_partial_merge,
                                                   &row_groups, NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize attention long query major",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    double query_major_time = bench_now() - start;
    bench_int_arg(gpu->attn_partial_b8, 13, 1);
    bench_pointer_arg(gpu->attn_partial_merge, 1, hierarchical);
    bench_telemetry_start(&telemetry[2]);
    start = bench_now();
    bench_ze_check("zeCommandListAppendLaunchKernel long partial",
                   zeCommandListAppendLaunchKernel(gpu->commands,
                                                   gpu->attn_partial_b8,
                                                   &partial_groups,
                                                   NULL, 0, NULL));
    bench_ze_check("zeCommandListAppendLaunchKernel long partial merge",
                   zeCommandListAppendLaunchKernel(gpu->commands,
                                                   gpu->attn_partial_merge,
                                                   &row_groups, NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize attention long hierarchical",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    times[2] = bench_now() - start;
    bench_telemetry_stop(&telemetry[2]);
    start = bench_now();
    bench_ze_check("zeCommandListAppendLaunchKernel long scalar",
                   zeCommandListAppendLaunchKernel(gpu->commands,
                                                   gpu->attn_online,
                                                   &row_groups, NULL, 0, NULL));
    bench_ze_check("zeCommandListHostSynchronize attention long scalar",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    double scalar_time = bench_now() - start;

    double oracle_start = bench_now();
    double *oracle_scores = malloc((size_t)keys * sizeof(*oracle_scores));
    if (!oracle_scores) bench_fatal("long attention oracle allocation");
    int oracle_probes = rows >= 4 ? 4 : 1;
    double oracle_reference = 0.0;
    double oracle_materialized_error = 0.0;
    double oracle_b8_error = 0.0;
    double oracle_hierarchical_error = 0.0;
    double oracle_scalar_error = 0.0;
    double oracle_query_major_error = 0.0;
    for (int probe = 0; probe < oracle_probes; probe++) {
        int query = oracle_probes == 1 ? 0 : probe * (rows - 1) / 3;
        int head = oracle_probes == 1 ? 0 : probe * 5;
        int kv_head = head * kv_heads / heads;
        size_t query_head = (size_t)head * rows + query;
        int oracle_last = keys - rows + query;
        double oracle_maximum = -INFINITY;
        for (int key = 0; key <= oracle_last; key++) {
            double value = 0.0;
            for (int d = 0; d < dimension; d++)
                value += (double)q[query_head * dimension + d]
                         * (double)kv[((size_t)kv_head * keys + key)
                                      * dimension + d];
            oracle_scores[key] = value;
            if (value > oracle_maximum) oracle_maximum = value;
        }
        double oracle_denominator = 0.0;
        for (int key = 0; key <= oracle_last; key++) {
            oracle_scores[key] = exp(oracle_scores[key] - oracle_maximum);
            oracle_denominator += oracle_scores[key];
        }
        for (int d = 0; d < dimension; d++) {
            double value = 0.0;
            for (int key = 0; key <= oracle_last; key++)
                value += oracle_scores[key]
                         * (double)kv[((size_t)kv_head * keys + key)
                                      * dimension + d];
            value /= oracle_denominator;
            size_t index = query_head * dimension + d;
            oracle_reference += value * value;
            double difference = materialized[index] - value;
            oracle_materialized_error += difference * difference;
            difference = online[index] - value;
            oracle_b8_error += difference * difference;
            difference = hierarchical[index] - value;
            oracle_hierarchical_error += difference * difference;
            difference = scalar[index] - value;
            oracle_scalar_error += difference * difference;
            difference = query_major[index] - value;
            oracle_query_major_error += difference * difference;
        }
    }
    double oracle_time = bench_now() - oracle_start;

    double error = 0.0;
    double reference = 0.0;
    double max_abs = 0.0;
    double scalar_error = 0.0;
    double scalar_b8_error = 0.0;
    double hierarchical_error = 0.0;
    for (size_t index = 0; index < q_elements; index++) {
        double difference = online[index] - materialized[index];
        error += difference * difference;
        reference += (double)materialized[index] * materialized[index];
        double scalar_difference = scalar[index] - materialized[index];
        double scalar_b8_difference = scalar[index] - online[index];
        scalar_error += scalar_difference * scalar_difference;
        scalar_b8_error += scalar_b8_difference * scalar_b8_difference;
        double hierarchical_difference = hierarchical[index]
                                         - materialized[index];
        hierarchical_error += hierarchical_difference
                              * hierarchical_difference;
        if (fabs(difference) > max_abs) max_abs = fabs(difference);
    }
    double rel_rms = sqrt(error / (reference + 1e-30));
    double scalar_rel_rms = sqrt(scalar_error / (reference + 1e-30));
    double scalar_b8_rel_rms = sqrt(scalar_b8_error / (reference + 1e-30));
    double hierarchical_rel_rms = sqrt(hierarchical_error
                                       / (reference + 1e-30));
    double operations = 4.0 * heads * rows * keys * dimension;
    printf("prefill-attention-long: b8 %.6f s query-major %.6f s hierarchical %.6f s scalar %.6f s materialized %.6f TFLOP/s b8 %.6f TFLOP/s speedup %.6fx hierarchical %.6fx locality %.6fx rel-rms b8-mat %.9g hierarchical-mat %.9g scalar-mat %.9g scalar-b8 %.9g max-abs %.9g clocks %.0f/%.0f/%.0fMHz throttle %ld/%ld/%ld %ld/%ld/%ld %ld/%ld/%ld numerical %s\n",
           times[1], query_major_time, times[2], scalar_time,
           operations / times[0] / 1e12,
           operations / times[1] / 1e12, times[0] / times[1],
           times[0] / times[2], query_major_time / times[2], rel_rms,
           hierarchical_rel_rms, scalar_rel_rms,
           scalar_b8_rel_rms, max_abs,
           (double)telemetry[0].actual_sum / telemetry[0].samples,
           (double)telemetry[1].actual_sum / telemetry[1].samples,
           (double)telemetry[2].actual_sum / telemetry[2].samples,
           telemetry[0].pl1, telemetry[0].pl2, telemetry[0].thermal,
           telemetry[1].pl1, telemetry[1].pl2, telemetry[1].thermal,
           telemetry[2].pl1, telemetry[2].pl2, telemetry[2].thermal,
           hierarchical_rel_rms <= 2e-5 ? "pass" : "investigate");
    printf("prefill-attention-long: double-oracle probes %d %.6f s rel-rms materialized %.9g b8 %.9g query-major %.9g hierarchical %.9g scalar %.9g\n",
           oracle_probes, oracle_time,
           sqrt(oracle_materialized_error / (oracle_reference + 1e-30)),
           sqrt(oracle_b8_error / (oracle_reference + 1e-30)),
           sqrt(oracle_query_major_error / (oracle_reference + 1e-30)),
           sqrt(oracle_hierarchical_error / (oracle_reference + 1e-30)),
           sqrt(oracle_scalar_error / (oracle_reference + 1e-30)));
    if (!isfinite(rel_rms) || !isfinite(scalar_rel_rms)
        || !isfinite(scalar_b8_rel_rms) || !isfinite(hierarchical_rel_rms))
        bench_fatal("long attention produced non-finite error");

    free(oracle_scores);
    bench_ze_check("zeMemFree attention long query major",
                   zeMemFree(gpu->context, query_major));
    bench_ze_check("zeMemFree attention long hierarchical",
                   zeMemFree(gpu->context, hierarchical));
    bench_ze_check("zeMemFree attention long partial",
                   zeMemFree(gpu->context, partial));
    bench_ze_check("zeMemFree attention long scalar",
                   zeMemFree(gpu->context, scalar));
    bench_ze_check("zeMemFree attention long online",
                   zeMemFree(gpu->context, online));
    bench_ze_check("zeMemFree attention long materialized",
                   zeMemFree(gpu->context, materialized));
    bench_ze_check("zeMemFree attention long scores",
                   zeMemFree(gpu->context, scores));
    bench_ze_check("zeMemFree attention long kv", zeMemFree(gpu->context, kv));
    bench_ze_check("zeMemFree attention long q", zeMemFree(gpu->context, q));
}

static void *bench_layer_alloc(bench_gpu *gpu, bench_layer_data *data,
                               size_t bytes) {
    if (data->allocation_count >= 96) bench_fatal("too many layer allocations");
    void *pointer = bench_gpu_alloc(gpu, bytes);
    data->allocations[data->allocation_count++] = pointer;
    return pointer;
}

static void bench_layer_matrix_init(bench_gpu *gpu, bench_layer_data *data,
                                    bench_layer_matrix *matrix, int rows,
                                    int n, int blocks, int experts,
                                    uint8_t pattern) {
    matrix->n = n;
    matrix->blocks = blocks;
    matrix->experts = experts;
    size_t weight_bytes = (size_t)experts * n * blocks * 16;
    size_t scale_elements = (size_t)experts * n * blocks;
    matrix->weight = bench_layer_alloc(gpu, data, weight_bytes);
    matrix->weight_scale = bench_layer_alloc(
        gpu, data, scale_elements * sizeof(_Float16));
    matrix->output = bench_layer_alloc(
        gpu, data, (size_t)rows * n * sizeof(float));
    memset(matrix->weight, pattern, weight_bytes);
    for (size_t index = 0; index < scale_elements; index++)
        matrix->weight_scale[index] =
            (_Float16)(0.0005f * (float)(1 + ((index + pattern) % 11)));
}

static void bench_layer_set_dense(ze_kernel_handle_t kernel,
                                  const bench_layer_matrix *matrix,
                                  const int8_t *q, const _Float16 *d,
                                  const int16_t *sigma, int rows) {
    bench_pointer_arg(kernel, 0, matrix->weight);
    bench_pointer_arg(kernel, 1, matrix->weight_scale);
    bench_pointer_arg(kernel, 2, q);
    bench_pointer_arg(kernel, 3, d);
    bench_pointer_arg(kernel, 4, sigma);
    bench_pointer_arg(kernel, 5, matrix->output);
    bench_int_arg(kernel, 6, rows);
    bench_int_arg(kernel, 7, matrix->n);
    bench_int_arg(kernel, 8, matrix->blocks);
}

static void bench_layer_set_grouped(ze_kernel_handle_t kernel,
                                    const bench_layer_matrix *matrix,
                                    const int8_t *q, const _Float16 *d,
                                    const int16_t *sigma,
                                    const bench_layer_data *data) {
    bench_pointer_arg(kernel, 0, matrix->weight);
    bench_pointer_arg(kernel, 1, matrix->weight_scale);
    bench_pointer_arg(kernel, 2, q);
    bench_pointer_arg(kernel, 3, d);
    bench_pointer_arg(kernel, 4, sigma);
    bench_pointer_arg(kernel, 5, matrix->output);
    bench_pointer_arg(kernel, 6, data->expert_count);
    bench_pointer_arg(kernel, 7, data->token_offset);
    bench_pointer_arg(kernel, 8, data->tile_expert);
    bench_pointer_arg(kernel, 9, data->tile_m0);
    bench_int_arg(kernel, 10, matrix->n);
    bench_int_arg(kernel, 11, matrix->blocks);
}

static void bench_layer_data_init(bench_gpu *gpu, bench_layer_data *data,
                                  int global) {
    memset(data, 0, sizeof *data);
    data->rows = 512;
    data->dimension = global ? 512 : 256;
    data->kv_heads = global ? 2 : 8;
    data->has_v = !global;
    int rows = data->rows;
    int width = 2816;
    int blocks = width / 32;
    int attention_width = 16 * data->dimension;
    int attention_blocks = attention_width / 32;
    int routes = rows * 8;
    int expert_blocks = 704 / 32;
    size_t hidden_elements = (size_t)rows * width;
    size_t q_elements = (size_t)rows * attention_width;
    size_t kv_elements = (size_t)rows * data->kv_heads * data->dimension;
    size_t route_elements = (size_t)routes * width;

    data->hidden = bench_layer_alloc(gpu, data,
                                     hidden_elements * sizeof(float));
    data->ones = bench_layer_alloc(gpu, data, width * sizeof(float));
    data->row_scale = bench_layer_alloc(gpu, data, rows * sizeof(float));
    data->attn_q8 = bench_layer_alloc(gpu, data, hidden_elements);
    data->attn_d = bench_layer_alloc(
        gpu, data, (size_t)rows * blocks * sizeof(_Float16));
    data->attn_s = bench_layer_alloc(
        gpu, data, (size_t)rows * blocks * sizeof(int16_t));

    bench_layer_matrix_init(gpu, data, &data->q, rows, attention_width,
                            blocks, 1, 0x31);
    bench_layer_matrix_init(gpu, data, &data->k, rows,
                            data->kv_heads * data->dimension, blocks, 1, 0x57);
    bench_layer_matrix_init(gpu, data, &data->v_projection, rows,
                            data->kv_heads * data->dimension, blocks, 1, 0x79);
    bench_layer_matrix_init(gpu, data, &data->o, rows, width,
                            attention_blocks, 1, 0x93);
    bench_layer_matrix_init(gpu, data, &data->dense_gate, rows, 2112,
                            blocks, 1, 0xa5);
    bench_layer_matrix_init(gpu, data, &data->dense_up, rows, 2112,
                            blocks, 1, 0xc7);
    bench_layer_matrix_init(gpu, data, &data->dense_down, rows, width,
                            2112 / 32, 1, 0xe9);
    bench_layer_matrix_init(gpu, data, &data->expert_gate_up, routes, 1408,
                            blocks, 128, 0x4b);
    bench_layer_matrix_init(gpu, data, &data->expert_down, routes, width,
                            expert_blocks, 128, 0x6d);

    data->q_weight = bench_layer_alloc(
        gpu, data, data->dimension * sizeof(float));
    data->k_weight = bench_layer_alloc(
        gpu, data, data->dimension * sizeof(float));
    data->rope_cos = bench_layer_alloc(
        gpu, data, (size_t)rows * data->dimension / 2 * sizeof(float));
    data->rope_sin = bench_layer_alloc(
        gpu, data, (size_t)rows * data->dimension / 2 * sizeof(float));
    data->q_heads = bench_layer_alloc(gpu, data, q_elements * sizeof(float));
    data->k_heads = bench_layer_alloc(
        gpu, data, kv_elements * sizeof(_Float16));
    data->v_heads = bench_layer_alloc(
        gpu, data, kv_elements * sizeof(_Float16));
    data->attn_heads = bench_layer_alloc(gpu, data,
                                         q_elements * sizeof(float));
    data->heads_q8 = bench_layer_alloc(gpu, data, q_elements);
    data->heads_d = bench_layer_alloc(
        gpu, data, (size_t)rows * attention_blocks * sizeof(_Float16));
    data->heads_s = bench_layer_alloc(
        gpu, data, (size_t)rows * attention_blocks * sizeof(int16_t));
    data->attn_out = bench_layer_alloc(gpu, data,
                                       hidden_elements * sizeof(float));
    data->dense_q8 = bench_layer_alloc(gpu, data, hidden_elements);
    data->dense_d = bench_layer_alloc(
        gpu, data, (size_t)rows * blocks * sizeof(_Float16));
    data->dense_s = bench_layer_alloc(
        gpu, data, (size_t)rows * blocks * sizeof(int16_t));
    data->moe_q8 = bench_layer_alloc(gpu, data, hidden_elements);
    data->moe_d = bench_layer_alloc(
        gpu, data, (size_t)rows * blocks * sizeof(_Float16));
    data->moe_s = bench_layer_alloc(
        gpu, data, (size_t)rows * blocks * sizeof(int16_t));
    data->router_input = bench_layer_alloc(gpu, data,
                                           hidden_elements * sizeof(float));
    data->router_weight = bench_layer_alloc(
        gpu, data, (size_t)128 * width * sizeof(float));
    data->router_logits = bench_layer_alloc(
        gpu, data, (size_t)rows * 128 * sizeof(float));
    data->route_expert = bench_layer_alloc(gpu, data, routes * sizeof(int));
    data->route_weight = bench_layer_alloc(gpu, data,
                                           routes * sizeof(float));
    data->route_token = bench_layer_alloc(gpu, data, routes * sizeof(int));
    data->expert_count = bench_layer_alloc(gpu, data, 128 * sizeof(int));
    data->token_offset = bench_layer_alloc(gpu, data, 129 * sizeof(int));
    data->cursor = bench_layer_alloc(gpu, data, 128 * sizeof(int));
    data->tile_expert = bench_layer_alloc(gpu, data, 256 * sizeof(int));
    data->tile_m0 = bench_layer_alloc(gpu, data, 256 * sizeof(int));
    data->packed_route = bench_layer_alloc(gpu, data, routes * sizeof(int));
    data->route_packed = bench_layer_alloc(gpu, data, routes * sizeof(int));
    data->packed_q8 = bench_layer_alloc(gpu, data, route_elements);
    data->packed_d = bench_layer_alloc(
        gpu, data, (size_t)routes * blocks * sizeof(_Float16));
    data->packed_s = bench_layer_alloc(
        gpu, data, (size_t)routes * blocks * sizeof(int16_t));
    data->dense_act_q8 = bench_layer_alloc(gpu, data,
                                           (size_t)rows * 2112);
    data->dense_act_d = bench_layer_alloc(
        gpu, data, (size_t)rows * (2112 / 32) * sizeof(_Float16));
    data->dense_act_s = bench_layer_alloc(
        gpu, data, (size_t)rows * (2112 / 32) * sizeof(int16_t));
    data->expert_act_q8 = bench_layer_alloc(gpu, data,
                                            (size_t)routes * 704);
    data->expert_act_d = bench_layer_alloc(
        gpu, data, (size_t)routes * expert_blocks * sizeof(_Float16));
    data->expert_act_s = bench_layer_alloc(
        gpu, data, (size_t)routes * expert_blocks * sizeof(int16_t));
    data->moe_reduced = bench_layer_alloc(gpu, data,
                                          hidden_elements * sizeof(float));
    data->expert_scale = bench_layer_alloc(gpu, data, 128 * sizeof(float));
    data->finish_output = bench_layer_alloc(gpu, data,
                                            hidden_elements * sizeof(float));
    data->layer_scale = bench_layer_alloc(gpu, data, sizeof(float));

    for (size_t index = 0; index < hidden_elements; index++)
        data->hidden[index] =
            ((int)((index * 37 + 11) % 2001) - 1000) / 1000.0f;
    for (int column = 0; column < width; column++) data->ones[column] = 1.0f;
    for (int d = 0; d < data->dimension; d++) {
        data->q_weight[d] = 0.75f + (float)(d % 17) / 32.0f;
        data->k_weight[d] = 0.625f + (float)(d % 13) / 32.0f;
    }
    float rope_base = global ? 1000000.0f : 10000.0f;
    for (int row = 0; row < rows; row++) {
        for (int d = 0; d < data->dimension / 2; d++) {
            float theta = row * powf(rope_base,
                                     -2.0f * d / data->dimension);
            size_t index = (size_t)row * data->dimension / 2 + d;
            data->rope_cos[index] = cosf(theta);
            data->rope_sin[index] = sinf(theta);
        }
    }
    for (size_t index = 0; index < (size_t)128 * width; index++)
        data->router_weight[index] =
            ((int)((index * 17 + index / width * 101) % 2001) - 1000)
            / 50000.0f;
    for (int route = 0; route < routes; route++)
        data->route_token[route] = route / 8;
    for (int expert = 0; expert < 128; expert++)
        data->expert_scale[expert] = 0.75f + (float)(expert % 11) / 32.0f;
    data->layer_scale[0] = 0.875f;

    const char *names[9] = {
        "prefill_q4q8_x8_tn64",
        global ? "prefill_q4q8_x8_tn64" : "prefill_q4q8_x8_fused",
        "prefill_q4q8_x8_fused",
        "prefill_q4q8_x8_tn64",
        "prefill_q4q8_x8_tn64",
        "prefill_q4q8_x8_tn64",
        "prefill_q4q8_x8_tn64",
        "prefill_q4q8_grouped_tn64",
        "prefill_q4q8_grouped_tn64"
    };
    for (int index = 0; index < 9; index++)
        data->kernels[index] = bench_kernel_create(gpu, names[index]);
    data->ffn_rms = bench_kernel_create(gpu, "prefill_rms_scale");

    bench_pointer_arg(gpu->rms_scale, 0, data->hidden);
    bench_pointer_arg(gpu->rms_scale, 1, data->row_scale);
    bench_int_arg(gpu->rms_scale, 2, rows);
    bench_int_arg(gpu->rms_scale, 3, width);
    bench_float_arg(gpu->rms_scale, 4, 1e-6f);
    bench_pointer_arg(gpu->norm_q8, 0, data->hidden);
    bench_pointer_arg(gpu->norm_q8, 1, data->ones);
    bench_pointer_arg(gpu->norm_q8, 2, data->row_scale);
    bench_pointer_arg(gpu->norm_q8, 3, data->attn_q8);
    bench_pointer_arg(gpu->norm_q8, 4, data->attn_d);
    bench_pointer_arg(gpu->norm_q8, 5, data->attn_s);
    bench_int_arg(gpu->norm_q8, 6, rows);
    bench_int_arg(gpu->norm_q8, 7, width);
    bench_layer_set_dense(data->kernels[0], &data->q, data->attn_q8,
                          data->attn_d, data->attn_s, rows);
    bench_layer_set_dense(data->kernels[1], &data->k, data->attn_q8,
                          data->attn_d, data->attn_s, rows);
    bench_layer_set_dense(data->kernels[2], &data->v_projection,
                          data->attn_q8, data->attn_d, data->attn_s, rows);

    bench_pointer_arg(gpu->qkv_post, 0, data->q.output);
    bench_pointer_arg(gpu->qkv_post, 1, data->k.output);
    bench_pointer_arg(gpu->qkv_post, 2, data->v_projection.output);
    bench_pointer_arg(gpu->qkv_post, 3, data->q_weight);
    bench_pointer_arg(gpu->qkv_post, 4, data->k_weight);
    bench_pointer_arg(gpu->qkv_post, 5, data->rope_cos);
    bench_pointer_arg(gpu->qkv_post, 6, data->rope_sin);
    bench_pointer_arg(gpu->qkv_post, 7, data->q_heads);
    bench_pointer_arg(gpu->qkv_post, 8, data->k_heads);
    bench_pointer_arg(gpu->qkv_post, 9, data->v_heads);
    bench_int_arg(gpu->qkv_post, 10, rows);
    bench_int_arg(gpu->qkv_post, 11, data->dimension);
    bench_int_arg(gpu->qkv_post, 12, data->kv_heads);
    bench_int_arg(gpu->qkv_post, 13, data->has_v);
    bench_float_arg(gpu->qkv_post, 14, 1e-6f);

    bench_pointer_arg(gpu->attn_online, 0, data->q_heads);
    bench_pointer_arg(gpu->attn_online, 1, data->k_heads);
    bench_pointer_arg(gpu->attn_online, 2, data->v_heads);
    bench_pointer_arg(gpu->attn_online, 3, data->attn_heads);
    bench_int_arg(gpu->attn_online, 4, rows);
    bench_int_arg(gpu->attn_online, 5, rows);
    bench_int_arg(gpu->attn_online, 6, data->dimension);
    bench_int_arg(gpu->attn_online, 7, 16);
    bench_int_arg(gpu->attn_online, 8, data->kv_heads);
    bench_int_arg(gpu->attn_online, 9, 0);
    bench_int_arg(gpu->attn_online, 10, global ? 0 : 1024);
    bench_pointer_arg(gpu->attn_online_b8, 0, data->q_heads);
    bench_pointer_arg(gpu->attn_online_b8, 1, data->k_heads);
    bench_pointer_arg(gpu->attn_online_b8, 2, data->v_heads);
    bench_pointer_arg(gpu->attn_online_b8, 3, data->attn_heads);
    bench_int_arg(gpu->attn_online_b8, 4, rows);
    bench_int_arg(gpu->attn_online_b8, 5, rows);
    bench_int_arg(gpu->attn_online_b8, 6, data->dimension);
    bench_int_arg(gpu->attn_online_b8, 7, 16);
    bench_int_arg(gpu->attn_online_b8, 8, data->kv_heads);
    bench_int_arg(gpu->attn_online_b8, 9, 0);
    bench_int_arg(gpu->attn_online_b8, 10, global ? 0 : 1024);

    bench_pointer_arg(gpu->heads_q8, 0, data->attn_heads);
    bench_pointer_arg(gpu->heads_q8, 1, data->heads_q8);
    bench_pointer_arg(gpu->heads_q8, 2, data->heads_d);
    bench_pointer_arg(gpu->heads_q8, 3, data->heads_s);
    bench_int_arg(gpu->heads_q8, 4, rows);
    bench_int_arg(gpu->heads_q8, 5, 16);
    bench_int_arg(gpu->heads_q8, 6, data->dimension);
    bench_layer_set_dense(data->kernels[3], &data->o, data->heads_q8,
                          data->heads_d, data->heads_s, rows);
    bench_pointer_arg(gpu->rms_residual, 0, data->o.output);
    bench_pointer_arg(gpu->rms_residual, 1, data->ones);
    bench_pointer_arg(gpu->rms_residual, 2, data->hidden);
    bench_pointer_arg(gpu->rms_residual, 3, data->attn_out);
    bench_int_arg(gpu->rms_residual, 4, rows);
    bench_int_arg(gpu->rms_residual, 5, width);
    bench_float_arg(gpu->rms_residual, 6, 1e-6f);

    bench_pointer_arg(data->ffn_rms, 0, data->attn_out);
    bench_pointer_arg(data->ffn_rms, 1, data->row_scale);
    bench_int_arg(data->ffn_rms, 2, rows);
    bench_int_arg(data->ffn_rms, 3, width);
    bench_float_arg(data->ffn_rms, 4, 1e-6f);

    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 0, data->attn_out);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 1, data->ones);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 2, data->ones);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 3, data->ones);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 4, data->row_scale);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 5, data->dense_q8);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 6, data->dense_d);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 7, data->dense_s);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 8, data->moe_q8);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 9, data->moe_d);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 10, data->moe_s);
    bench_pointer_arg(gpu->ffn_input_q8_rowscale, 11, data->router_input);
    bench_float_arg(gpu->ffn_input_q8_rowscale, 12,
                    1.0f / sqrtf((float)width));
    bench_int_arg(gpu->ffn_input_q8_rowscale, 13, rows);
    bench_int_arg(gpu->ffn_input_q8_rowscale, 14, width);
    bench_layer_set_dense(data->kernels[4], &data->dense_gate,
                          data->dense_q8, data->dense_d, data->dense_s, rows);
    bench_layer_set_dense(data->kernels[5], &data->dense_up,
                          data->dense_q8, data->dense_d, data->dense_s, rows);
    bench_pointer_arg(gpu->geglu_q8_pair, 0, data->dense_gate.output);
    bench_pointer_arg(gpu->geglu_q8_pair, 1, data->dense_up.output);
    bench_pointer_arg(gpu->geglu_q8_pair, 2, data->dense_act_q8);
    bench_pointer_arg(gpu->geglu_q8_pair, 3, data->dense_act_d);
    bench_pointer_arg(gpu->geglu_q8_pair, 4, data->dense_act_s);
    bench_int_arg(gpu->geglu_q8_pair, 5, rows);
    bench_int_arg(gpu->geglu_q8_pair, 6, 2112);
    bench_layer_set_dense(data->kernels[6], &data->dense_down,
                          data->dense_act_q8, data->dense_act_d,
                          data->dense_act_s, rows);

    bench_pointer_arg(gpu->router_gemm_tiled, 0, data->router_input);
    bench_pointer_arg(gpu->router_gemm_tiled, 1, data->router_weight);
    bench_pointer_arg(gpu->router_gemm_tiled, 2, data->router_logits);
    bench_int_arg(gpu->router_gemm_tiled, 3, rows);
    bench_int_arg(gpu->router_gemm_tiled, 4, width);
    bench_int_arg(gpu->router_gemm_tiled, 5, 128);
    bench_pointer_arg(gpu->router_top8, 0, data->router_logits);
    bench_pointer_arg(gpu->router_top8, 1, data->route_expert);
    bench_pointer_arg(gpu->router_top8, 2, data->route_weight);
    bench_int_arg(gpu->router_top8, 3, rows);
    bench_int_arg(gpu->router_top8, 4, 128);
    bench_pointer_arg(gpu->route_reset, 0, data->expert_count);
    bench_pointer_arg(gpu->route_reset, 1, data->cursor);
    bench_pointer_arg(gpu->route_reset, 2, data->tile_expert);
    bench_pointer_arg(gpu->route_reset, 3, data->tile_m0);
    bench_pointer_arg(gpu->route_count, 0, data->route_expert);
    bench_pointer_arg(gpu->route_count, 1, data->expert_count);
    bench_int_arg(gpu->route_count, 2, routes);
    bench_pointer_arg(gpu->route_prefix, 0, data->expert_count);
    bench_pointer_arg(gpu->route_prefix, 1, data->token_offset);
    bench_pointer_arg(gpu->route_prefix, 2, data->cursor);
    bench_pointer_arg(gpu->route_prefix, 3, data->tile_expert);
    bench_pointer_arg(gpu->route_prefix, 4, data->tile_m0);
    bench_pointer_arg(gpu->route_scatter, 0, data->route_expert);
    bench_pointer_arg(gpu->route_scatter, 1, data->cursor);
    bench_pointer_arg(gpu->route_scatter, 2, data->packed_route);
    bench_pointer_arg(gpu->route_scatter, 3, data->route_packed);
    bench_int_arg(gpu->route_scatter, 4, routes);
    bench_pointer_arg(gpu->route_pack, 0, data->moe_q8);
    bench_pointer_arg(gpu->route_pack, 1, data->moe_d);
    bench_pointer_arg(gpu->route_pack, 2, data->moe_s);
    bench_pointer_arg(gpu->route_pack, 3, data->packed_q8);
    bench_pointer_arg(gpu->route_pack, 4, data->packed_d);
    bench_pointer_arg(gpu->route_pack, 5, data->packed_s);
    bench_pointer_arg(gpu->route_pack, 6, data->route_token);
    bench_pointer_arg(gpu->route_pack, 7, data->packed_route);
    bench_int_arg(gpu->route_pack, 8, blocks);
    bench_int_arg(gpu->route_pack, 9, routes);
    bench_layer_set_grouped(data->kernels[7], &data->expert_gate_up,
                            data->packed_q8, data->packed_d, data->packed_s,
                            data);
    bench_pointer_arg(gpu->geglu_q8, 0, data->expert_gate_up.output);
    bench_pointer_arg(gpu->geglu_q8, 1, data->expert_act_q8);
    bench_pointer_arg(gpu->geglu_q8, 2, data->expert_act_d);
    bench_pointer_arg(gpu->geglu_q8, 3, data->expert_act_s);
    bench_int_arg(gpu->geglu_q8, 4, routes);
    bench_int_arg(gpu->geglu_q8, 5, 704);
    bench_layer_set_grouped(data->kernels[8], &data->expert_down,
                            data->expert_act_q8, data->expert_act_d,
                            data->expert_act_s, data);
    bench_pointer_arg(gpu->route_reduce_scaled, 0, data->expert_down.output);
    bench_pointer_arg(gpu->route_reduce_scaled, 1, data->route_weight);
    bench_pointer_arg(gpu->route_reduce_scaled, 2, data->route_expert);
    bench_pointer_arg(gpu->route_reduce_scaled, 3, data->route_packed);
    bench_pointer_arg(gpu->route_reduce_scaled, 4, data->expert_scale);
    bench_pointer_arg(gpu->route_reduce_scaled, 5, data->moe_reduced);
    bench_int_arg(gpu->route_reduce_scaled, 6, width);
    bench_pointer_arg(gpu->ffn_finish, 0, data->dense_down.output);
    bench_pointer_arg(gpu->ffn_finish, 1, data->moe_reduced);
    bench_pointer_arg(gpu->ffn_finish, 2, data->ones);
    bench_pointer_arg(gpu->ffn_finish, 3, data->ones);
    bench_pointer_arg(gpu->ffn_finish, 4, data->ones);
    bench_pointer_arg(gpu->ffn_finish, 5, data->attn_out);
    bench_pointer_arg(gpu->ffn_finish, 6, data->layer_scale);
    bench_pointer_arg(gpu->ffn_finish, 7, data->finish_output);
    bench_int_arg(gpu->ffn_finish, 8, rows);
    bench_int_arg(gpu->ffn_finish, 9, width);
    bench_float_arg(gpu->ffn_finish, 10, 1e-6f);
}

static void bench_layer_append(bench_gpu *gpu, const bench_layer_data *data,
                               int use_b8, ze_event_handle_t *events) {
    int rows = data->rows;
    int width = 2816;
    int attention_width = 16 * data->dimension;
    ze_group_count_t rows_groups = { (uint32_t)rows, 1, 1 };
    ze_group_count_t norm_groups = { (uint32_t)(rows * (width / 32) / 8), 1, 1 };
    ze_group_count_t q_groups = { (uint32_t)(rows / 32),
                                  (uint32_t)(attention_width / 64), 1 };
    ze_group_count_t k_groups = { (uint32_t)(rows / 32),
                                  (uint32_t)(data->k.n
                                             / (data->has_v ? 32 : 64)), 1 };
    ze_group_count_t post_groups = {
        (uint32_t)(rows * (16 + 2 * data->kv_heads)), 1, 1
    };
    ze_group_count_t attention_groups = { (uint32_t)(16 * rows / 8), 1, 1 };
    ze_group_count_t heads_groups = {
        (uint32_t)(rows * (attention_width / 32) / 8), 1, 1
    };
    ze_group_count_t o_groups = { (uint32_t)(rows / 32), width / 64, 1 };
    ze_group_count_t dense_gu_groups = { (uint32_t)(rows / 32), 2112 / 64, 1 };
    ze_group_count_t dense_act_groups = {
        (uint32_t)(rows * (2112 / 32) / 8), 1, 1
    };
    ze_group_count_t dense_down_groups = { (uint32_t)(rows / 32), width / 64, 1 };
    ze_group_count_t router_groups = { (uint32_t)(rows / 16), 128 / 16, 1 };
    ze_group_count_t top_groups = { (uint32_t)(rows / 8), 1, 1 };
    ze_group_count_t reset_groups = { 2, 1, 1 };
    ze_group_count_t route_groups = { (uint32_t)(rows * 8 / 128), 1, 1 };
    ze_group_count_t one_group = { 1, 1, 1 };
    ze_group_count_t pack_groups = { (uint32_t)(rows * 8), 1, 1 };
    ze_group_count_t expert_gate_groups = { 256, 1408 / 64, 1 };
    ze_group_count_t expert_act_groups = {
        (uint32_t)(rows * 8 * (704 / 32) / 8), 1, 1
    };
    ze_group_count_t expert_down_groups = { 256, width / 64, 1 };
    int event_index = 0;

#define BENCH_LAYER_LAUNCH(label, kernel, groups) \
    bench_ze_check(label, zeCommandListAppendLaunchKernel( \
        gpu->commands, kernel, groups, \
        events ? events[event_index++] : NULL, 0, NULL))
    BENCH_LAYER_LAUNCH("layer RMS input", gpu->rms_scale, &rows_groups);
    BENCH_LAYER_LAUNCH("layer norm Q8", gpu->norm_q8, &norm_groups);
    BENCH_LAYER_LAUNCH("layer Q", data->kernels[0], &q_groups);
    BENCH_LAYER_LAUNCH("layer K", data->kernels[1], &k_groups);
    if (data->has_v)
        BENCH_LAYER_LAUNCH("layer V", data->kernels[2], &k_groups);
    BENCH_LAYER_LAUNCH("layer QKV post", gpu->qkv_post, &post_groups);
    BENCH_LAYER_LAUNCH("layer attention",
                       use_b8 ? gpu->attn_online_b8 : gpu->attn_online,
                       &attention_groups);
    BENCH_LAYER_LAUNCH("layer heads Q8", gpu->heads_q8, &heads_groups);
    BENCH_LAYER_LAUNCH("layer O", data->kernels[3], &o_groups);
    BENCH_LAYER_LAUNCH("layer attention residual", gpu->rms_residual,
                       &rows_groups);
    BENCH_LAYER_LAUNCH("layer FFN RMS", data->ffn_rms, &rows_groups);
    BENCH_LAYER_LAUNCH("layer FFN input", gpu->ffn_input_q8_rowscale,
                       &norm_groups);
    BENCH_LAYER_LAUNCH("layer dense gate", data->kernels[4], &dense_gu_groups);
    BENCH_LAYER_LAUNCH("layer dense up", data->kernels[5], &dense_gu_groups);
    BENCH_LAYER_LAUNCH("layer dense activation", gpu->geglu_q8_pair,
                       &dense_act_groups);
    BENCH_LAYER_LAUNCH("layer dense down", data->kernels[6],
                       &dense_down_groups);
    BENCH_LAYER_LAUNCH("layer router", gpu->router_gemm_tiled, &router_groups);
    BENCH_LAYER_LAUNCH("layer top8", gpu->router_top8, &top_groups);
    BENCH_LAYER_LAUNCH("layer route reset", gpu->route_reset, &reset_groups);
    BENCH_LAYER_LAUNCH("layer route count", gpu->route_count, &route_groups);
    BENCH_LAYER_LAUNCH("layer route prefix", gpu->route_prefix, &one_group);
    BENCH_LAYER_LAUNCH("layer route scatter", gpu->route_scatter, &route_groups);
    BENCH_LAYER_LAUNCH("layer route pack", gpu->route_pack, &pack_groups);
    BENCH_LAYER_LAUNCH("layer expert gate up", data->kernels[7],
                       &expert_gate_groups);
    BENCH_LAYER_LAUNCH("layer expert activation", gpu->geglu_q8,
                       &expert_act_groups);
    BENCH_LAYER_LAUNCH("layer expert down", data->kernels[8],
                       &expert_down_groups);
    BENCH_LAYER_LAUNCH("layer expert reduce", gpu->route_reduce_scaled,
                       &rows_groups);
    BENCH_LAYER_LAUNCH("layer FFN finish", gpu->ffn_finish, &rows_groups);
#undef BENCH_LAYER_LAUNCH
}

static double bench_layer_launch(bench_gpu *gpu, const bench_layer_data *data,
                                 int use_b8, int repetitions) {
    double start = bench_now();
    for (int repetition = 0; repetition < repetitions; repetition++)
        bench_layer_append(gpu, data, use_b8, NULL);
    bench_ze_check("zeCommandListHostSynchronize layer",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    return bench_now() - start;
}

static void bench_layer_profile(bench_gpu *gpu,
                                const bench_layer_data *data) {
    const char *labels[30] = {
        "rms-input", "norm-q8", "q", "k", "v", "qkv-post",
        "attention", "heads-q8", "o", "attention-residual", "ffn-rms",
        "ffn-input", "dense-gate", "dense-up", "dense-geglu",
        "dense-down", "router-gemm", "router-top8", "route-reset",
        "route-count", "route-prefix", "route-scatter", "route-pack",
        "expert-gate-up", "expert-geglu", "expert-down", "expert-reduce",
        "ffn-finish"
    };
    int count = data->has_v ? 28 : 27;
    if (!data->has_v)
        for (int index = 4; index < count; index++) labels[index] = labels[index + 1];
    ze_event_pool_desc_t pool_desc = {
        .stype = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        .flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE
                 | ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP,
        .count = (uint32_t)count
    };
    ze_event_pool_handle_t pool = NULL;
    bench_ze_check("zeEventPoolCreate layer",
                   zeEventPoolCreate(gpu->context, &pool_desc, 1,
                                     &gpu->device, &pool));
    ze_event_handle_t events[30] = {0};
    for (int index = 0; index < count; index++) {
        ze_event_desc_t desc = {
            .stype = ZE_STRUCTURE_TYPE_EVENT_DESC,
            .index = (uint32_t)index,
            .signal = ZE_EVENT_SCOPE_FLAG_DEVICE,
            .wait = ZE_EVENT_SCOPE_FLAG_HOST
        };
        bench_ze_check("zeEventCreate layer",
                       zeEventCreate(pool, &desc, &events[index]));
    }
    bench_layer_append(gpu, data, 1, events);
    bench_ze_check("zeCommandListHostSynchronize layer profile",
                   zeCommandListHostSynchronize(gpu->commands, UINT64_MAX));
    uint64_t mask = gpu->timestamp_bits == 64
                    ? UINT64_MAX : (UINT64_C(1) << gpu->timestamp_bits) - 1;
    double total = 0.0;
    for (int index = 0; index < count; index++) {
        ze_kernel_timestamp_result_t timestamp;
        bench_ze_check("zeEventQueryKernelTimestamp layer",
                       zeEventQueryKernelTimestamp(events[index], &timestamp));
        uint64_t ticks = (timestamp.context.kernelEnd
                          - timestamp.context.kernelStart) & mask;
        double milliseconds = (double)ticks / gpu->timer_resolution * 1e3;
        total += milliseconds;
        printf("prefill-layer: profile %s %.6f ms\n", labels[index],
               milliseconds);
    }
    printf("prefill-layer: profile kernel-total %.6f ms\n", total);
    for (int index = count - 1; index >= 0; index--)
        bench_ze_check("zeEventDestroy layer", zeEventDestroy(events[index]));
    bench_ze_check("zeEventPoolDestroy layer", zeEventPoolDestroy(pool));
}

static void bench_layer_run(bench_gpu *gpu, const char *name, int rounds,
                            double target_seconds) {
    int global = name && !strcmp(name, "global");
    if (name && strcmp(name, "swa") && strcmp(name, "global"))
        bench_fatal("unknown layer shape: %s", name);
    bench_layer_data data;
    bench_layer_data_init(gpu, &data, global);
    bench_layer_launch(gpu, &data, 1, 1);
    size_t elements = (size_t)data.rows * 2816;
    float *reference = bench_alloc(elements * sizeof(float));
    memcpy(reference, data.finish_output, elements * sizeof(float));
    bench_layer_launch(gpu, &data, 1, 1);
    double error = 0.0;
    double norm = 0.0;
    int finite = 1;
    for (size_t index = 0; index < elements; index++) {
        double actual = data.finish_output[index];
        double difference = actual - reference[index];
        error += difference * difference;
        norm += (double)reference[index] * reference[index];
        if (!isfinite(actual)) finite = 0;
    }
    int route_sum = 0;
    int route_valid = 1;
    int active_experts = 0;
    int active_tiles = 0;
    int max_count = 0;
    for (int expert = 0; expert < 128; expert++) {
        route_sum += data.expert_count[expert];
        if (data.expert_count[expert]) active_experts++;
        active_tiles += (data.expert_count[expert] + 31) / 32;
        if (data.expert_count[expert] > max_count)
            max_count = data.expert_count[expert];
    }
    for (int route = 0; route < data.rows * 8; route++)
        if (data.route_expert[route] < 0 || data.route_expert[route] >= 128)
            route_valid = 0;
    double rel_rms = sqrt(error / (norm + 1e-30));
    printf("prefill-layer: correctness %s repeat rel-rms %.9g finite %s routes %d/%d valid %s\n",
           global ? "global" : "swa", rel_rms, finite ? "yes" : "no",
           route_sum, data.rows * 8, route_valid ? "yes" : "no");
    printf("prefill-layer: routing active-experts %d tiles %d padding %.6fx max-count %d\n",
           active_experts, active_tiles,
           (double)(active_tiles * 32) / route_sum, max_count);
    if (!finite || !route_valid || route_sum != data.rows * 8
        || rel_rms > 1e-6)
        bench_fatal("layer correctness failed");
    free(reference);

    bench_layer_profile(gpu, &data);

    double probe = bench_layer_launch(gpu, &data, 1, 1);
    int repetitions = (int)ceil(target_seconds / probe);
    if (repetitions < 2) repetitions = 2;
    double times[2][BENCH_ROUNDS_MAX];
    double frequencies[2][BENCH_ROUNDS_MAX];
    for (int round = 0; round < rounds; round++) {
        printf("prefill-layer: round %d %s", round + 1,
               global ? "global" : "swa");
        for (int position = 0; position < 2; position++) {
            int variant = (round + position) & 1;
            bench_telemetry telemetry;
            bench_telemetry_start(&telemetry);
            double seconds = bench_layer_launch(gpu, &data, variant,
                                                repetitions);
            bench_telemetry_stop(&telemetry);
            times[variant][round] = seconds / repetitions;
            frequencies[variant][round] =
                (double)telemetry.actual_sum / telemetry.samples;
            printf(" %s %.6f ms@%.0fMHz[%ld,%ld] throttle=%ld/%ld/%ld",
                   variant ? "b8" : "scalar", times[variant][round] * 1e3,
                   frequencies[variant][round], telemetry.actual_min,
                   telemetry.actual_max, telemetry.pl1, telemetry.pl2,
                   telemetry.thermal);
        }
        printf("\n");
    }
    double scalar = bench_median(times[0], rounds);
    double b8 = bench_median(times[1], rounds);
    double scalar_frequency = bench_median(frequencies[0], rounds);
    double b8_frequency = bench_median(frequencies[1], rounds);
    printf("prefill-layer: median %s scalar %.6f ms @%.0fMHz b8 %.6f ms @%.0fMHz speedup %.6fx batch-throughput %.6f tok/s model-cost %.6f ms/token\n",
           global ? "global" : "swa", scalar * 1e3, scalar_frequency,
           b8 * 1e3, b8_frequency, scalar / b8, data.rows / b8,
           b8 / data.rows * 1e3);

    bench_ze_check("zeKernelDestroy layer FFN RMS",
                   zeKernelDestroy(data.ffn_rms));
    for (int index = 8; index >= 0; index--)
        bench_ze_check("zeKernelDestroy layer kernel",
                       zeKernelDestroy(data.kernels[index]));
    for (int index = data.allocation_count - 1; index >= 0; index--)
        bench_ze_check("zeMemFree layer", zeMemFree(gpu->context,
                                                    data.allocations[index]));
}

static double bench_parse_double(const char *name, const char *text) {
    char *end;
    errno = 0;
    double value = strtod(text, &end);
    if (errno || *end || !isfinite(value) || value <= 0.0)
        bench_fatal("invalid %s: %s", name, text);
    return value;
}

static int bench_parse_int(const char *name, const char *text) {
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || *end || value < 1 || value > INT_MAX)
        bench_fatal("invalid %s: %s", name, text);
    return (int)value;
}

static void bench_gpu_destroy(bench_gpu *gpu) {
    bench_ze_check("zeKernelDestroy FFN finish",
                   zeKernelDestroy(gpu->ffn_finish));
    bench_ze_check("zeKernelDestroy RMS residual",
                   zeKernelDestroy(gpu->rms_residual));
    bench_ze_check("zeKernelDestroy heads Q8",
                   zeKernelDestroy(gpu->heads_q8));
    bench_ze_check("zeKernelDestroy QKV post", zeKernelDestroy(gpu->qkv_post));
    bench_ze_check("zeKernelDestroy FFN input Q8 rowscale",
                   zeKernelDestroy(gpu->ffn_input_q8_rowscale));
    bench_ze_check("zeKernelDestroy norm Q8", zeKernelDestroy(gpu->norm_q8));
    bench_ze_check("zeKernelDestroy RMS scale",
                   zeKernelDestroy(gpu->rms_scale));
    bench_ze_check("zeKernelDestroy FFN input Q8",
                   zeKernelDestroy(gpu->ffn_input_q8));
    bench_ze_check("zeKernelDestroy router top8",
                   zeKernelDestroy(gpu->router_top8));
    bench_ze_check("zeKernelDestroy router GEMM",
                   zeKernelDestroy(gpu->router_gemm));
    bench_ze_check("zeKernelDestroy router tiled GEMM",
                   zeKernelDestroy(gpu->router_gemm_tiled));
    bench_ze_check("zeKernelDestroy GeGLU Q8", zeKernelDestroy(gpu->geglu_q8));
    bench_ze_check("zeKernelDestroy GeGLU Q8 pair",
                   zeKernelDestroy(gpu->geglu_q8_pair));
    bench_ze_check("zeKernelDestroy attn gqa8", zeKernelDestroy(gpu->attn_gqa8));
    bench_ze_check("zeKernelDestroy attn online b8",
                   zeKernelDestroy(gpu->attn_online_b8));
    bench_ze_check("zeKernelDestroy attn online b8 ring",
                   zeKernelDestroy(gpu->attn_online_b8_ring));
    bench_ze_check("zeKernelDestroy SWA stage", zeKernelDestroy(gpu->swa_stage));
    bench_ze_check("zeKernelDestroy SWA commit", zeKernelDestroy(gpu->swa_commit));
    bench_ze_check("zeKernelDestroy attention long init",
                   zeKernelDestroy(gpu->attn_long_init));
    bench_ze_check("zeKernelDestroy attention partial b8",
                   zeKernelDestroy(gpu->attn_partial_b8));
    bench_ze_check("zeKernelDestroy attention partial merge",
                   zeKernelDestroy(gpu->attn_partial_merge));
    bench_ze_check("zeKernelDestroy attn online b4",
                   zeKernelDestroy(gpu->attn_online_b4));
    bench_ze_check("zeKernelDestroy attn online", zeKernelDestroy(gpu->attn_online));
    bench_ze_check("zeKernelDestroy attn pv", zeKernelDestroy(gpu->attn_pv));
    bench_ze_check("zeKernelDestroy attn pv4", zeKernelDestroy(gpu->attn_pv4));
    bench_ze_check("zeKernelDestroy attn softmax",
                   zeKernelDestroy(gpu->attn_softmax));
    bench_ze_check("zeKernelDestroy attn qk", zeKernelDestroy(gpu->attn_qk));
    bench_ze_check("zeKernelDestroy tail", zeKernelDestroy(gpu->tail));
    bench_ze_check("zeKernelDestroy route pack", zeKernelDestroy(gpu->route_pack));
    bench_ze_check("zeKernelDestroy route reduce",
                   zeKernelDestroy(gpu->route_reduce));
    bench_ze_check("zeKernelDestroy route reduce scaled",
                   zeKernelDestroy(gpu->route_reduce_scaled));
    bench_ze_check("zeKernelDestroy route scatter",
                   zeKernelDestroy(gpu->route_scatter));
    bench_ze_check("zeKernelDestroy route prefix",
                   zeKernelDestroy(gpu->route_prefix));
    bench_ze_check("zeKernelDestroy route count",
                   zeKernelDestroy(gpu->route_count));
    bench_ze_check("zeKernelDestroy route reset",
                   zeKernelDestroy(gpu->route_reset));
    bench_ze_check("zeKernelDestroy grouped gather",
                   zeKernelDestroy(gpu->grouped_gather));
    bench_ze_check("zeKernelDestroy grouped expert",
                   zeKernelDestroy(gpu->grouped_expert));
    bench_ze_check("zeKernelDestroy grouped64", zeKernelDestroy(gpu->grouped64));
    bench_ze_check("zeKernelDestroy grouped tn48",
                   zeKernelDestroy(gpu->grouped_tn48));
    bench_ze_check("zeKernelDestroy grouped tn64",
                   zeKernelDestroy(gpu->grouped_tn64));
    bench_ze_check("zeKernelDestroy grouped tn64 direct",
                   zeKernelDestroy(gpu->grouped_tn64_direct));
    bench_ze_check("zeKernelDestroy grouped tn64 coalesced",
                   zeKernelDestroy(gpu->grouped_tn64_coalesced));
    bench_ze_check("zeKernelDestroy grouped tn64 slmacc",
                   zeKernelDestroy(gpu->grouped_tn64_slmacc));
    bench_ze_check("zeKernelDestroy grouped tn64 wg256",
                   zeKernelDestroy(gpu->grouped_tn64_wg256));
    bench_ze_check("zeKernelDestroy grouped tm64 tn64 wg256",
                   zeKernelDestroy(gpu->grouped_tm64_tn64_wg256));
    bench_ze_check("zeKernelDestroy grouped tm24 tn64",
                   zeKernelDestroy(gpu->grouped_tm24_tn64));
    bench_ze_check("zeKernelDestroy grouped tn64 signed",
                   zeKernelDestroy(gpu->grouped_tn64_signed));
    bench_ze_check("zeKernelDestroy grouped tn64 kb128",
                   zeKernelDestroy(gpu->grouped_tn64_kb128));
    bench_ze_check("zeKernelDestroy grouped tm16 tn64",
                   zeKernelDestroy(gpu->grouped_tm16_tn64));
    bench_ze_check("zeKernelDestroy grouped tn128 wg256",
                   zeKernelDestroy(gpu->grouped_tn128_wg256));
    bench_ze_check("zeKernelDestroy grouped", zeKernelDestroy(gpu->grouped));
    bench_ze_check("zeKernelDestroy correction", zeKernelDestroy(gpu->correction));
    bench_ze_check("zeKernelDestroy raw", zeKernelDestroy(gpu->raw));
    bench_ze_check("zeKernelDestroy row", zeKernelDestroy(gpu->row));
    bench_ze_check("zeKernelDestroy fused", zeKernelDestroy(gpu->fused));
    bench_ze_check("zeKernelDestroy tn48", zeKernelDestroy(gpu->tn48));
    bench_ze_check("zeKernelDestroy tn64", zeKernelDestroy(gpu->tn64));
    bench_ze_check("zeKernelDestroy tm64 tn64",
                   zeKernelDestroy(gpu->tm64_tn64));
    bench_ze_check("zeKernelDestroy tn64 alt1",
                   zeKernelDestroy(gpu->tn64_alt1));
    bench_ze_check("zeKernelDestroy tn64 alt2",
                   zeKernelDestroy(gpu->tn64_alt2));
    bench_ze_check("zeKernelDestroy tn64 alt3",
                   zeKernelDestroy(gpu->tn64_alt3));
    bench_ze_check("zeKernelDestroy tn64 wg256",
                   zeKernelDestroy(gpu->tn64_wg256));
    bench_ze_check("zeKernelDestroy tn128 wg256",
                   zeKernelDestroy(gpu->tn128_wg256));
    bench_ze_check("zeModuleDestroy", zeModuleDestroy(gpu->module));
    bench_ze_check("zeCommandListDestroy", zeCommandListDestroy(gpu->commands));
    bench_ze_check("zeContextDestroy", zeContextDestroy(gpu->context));
}

int main(int argc, char **argv) {
    const bench_shape shapes[] = {
        { "dense", 512, 4096, 88 },
        { "expert32", 32, 1408, 88 },
        { "expert512", 512, 1408, 88 },
        { "down", 512, 2816, 22 },
        { "swa-kv", 512, 2048, 88 },
        { "swa-o", 512, 2816, 128 },
        { "global-q", 512, 8192, 88 },
        { "global-k", 512, 1024, 88 },
        { "global-o", 512, 2816, 256 },
        { "dense-gu", 512, 2112, 88 },
        { "dense-gu-pair", 512, 4224, 88 },
        { "dense-down", 512, 2816, 66 },
        { "swa-qkv", 512, 8192, 88 },
        { "global-qk", 512, 9216, 88 }
    };
    const char *spv_path = "tests/bench_prefill_gemm.spv";
    const char *selected_shape = NULL;
    const char *selected_distribution = NULL;
    const char *selected_attention = NULL;
    const char *selected_activation = NULL;
    const char *selected_qkv = NULL;
    const char *selected_glue = NULL;
    const char *selected_layer = NULL;
    double seconds = 0.15;
    int rounds = 5;
    int run_moe = 0;
    int run_moe_signed = 0;
    int run_moe_kb128 = 0;
    int run_moe_slmacc = 0;
    int run_moe_direct = 0;
    int run_moe_coalesced = 0;
    int run_b5 = 0;
    int run_attention = 0;
    int run_attention_long = 0;
    int attention_long_rows = 0;
    int attention_long_keys = 0;
    int attention_long_chunk = 4096;
    int run_routing = 0;
    int run_reduce = 0;
    int run_activate = 0;
    int run_router = 0;
    int run_input = 0;
    int run_moe_tile = 0;
    int run_tile_only = 0;
    int run_fusion = 0;
    int run_qkv_post = 0;
    int run_glue = 0;
    int run_layer = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--spv") && i + 1 < argc)
            spv_path = argv[++i];
        else if (!strcmp(argv[i], "--shape") && i + 1 < argc)
            selected_shape = argv[++i];
        else if (!strcmp(argv[i], "--moe"))
            run_moe = 1;
        else if (!strcmp(argv[i], "--moe-signed"))
            run_moe_signed = 1;
        else if (!strcmp(argv[i], "--moe-kb128"))
            run_moe_kb128 = 1;
        else if (!strcmp(argv[i], "--moe-slmacc"))
            run_moe_slmacc = 1;
        else if (!strcmp(argv[i], "--moe-direct"))
            run_moe_direct = 1;
        else if (!strcmp(argv[i], "--moe-coalesced"))
            run_moe_coalesced = 1;
        else if (!strcmp(argv[i], "--b5-reconstruct"))
            run_b5 = 1;
        else if (!strcmp(argv[i], "--moe-tile"))
            run_moe_tile = 1;
        else if (!strcmp(argv[i], "--tile-only"))
            run_tile_only = 1;
        else if (!strcmp(argv[i], "--fusion"))
            run_fusion = 1;
        else if (!strcmp(argv[i], "--qkv-post") && i + 1 < argc) {
            run_qkv_post = 1;
            selected_qkv = argv[++i];
        }
        else if (!strcmp(argv[i], "--glue") && i + 1 < argc) {
            run_glue = 1;
            selected_glue = argv[++i];
        }
        else if (!strcmp(argv[i], "--layer") && i + 1 < argc) {
            run_layer = 1;
            selected_layer = argv[++i];
        }
        else if (!strcmp(argv[i], "--attention"))
            run_attention = 1;
        else if (!strcmp(argv[i], "--attention-long") && i + 2 < argc) {
            run_attention_long = 1;
            attention_long_rows = bench_parse_int("attention rows", argv[++i]);
            attention_long_keys = bench_parse_int("attention keys", argv[++i]);
        }
        else if (!strcmp(argv[i], "--attention-chunk") && i + 1 < argc)
            attention_long_chunk = bench_parse_int("attention chunk", argv[++i]);
        else if (!strcmp(argv[i], "--routing"))
            run_routing = 1;
        else if (!strcmp(argv[i], "--reduce"))
            run_reduce = 1;
        else if (!strcmp(argv[i], "--activate"))
            run_activate = 1;
        else if (!strcmp(argv[i], "--router"))
            run_router = 1;
        else if (!strcmp(argv[i], "--input"))
            run_input = 1;
        else if (!strcmp(argv[i], "--attention-shape") && i + 1 < argc)
            selected_attention = argv[++i];
        else if (!strcmp(argv[i], "--activate-shape") && i + 1 < argc)
            selected_activation = argv[++i];
        else if (!strcmp(argv[i], "--distribution") && i + 1 < argc)
            selected_distribution = argv[++i];
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc)
            seconds = bench_parse_double("seconds", argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc)
            rounds = bench_parse_int("rounds", argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--spv PATH] [--shape NAME | --moe [--distribution NAME] | --routing [--distribution NAME] | --reduce [--distribution NAME] | --activate [--distribution NAME] | --router | --input | --attention [--attention-shape NAME] | --layer swa|global] [--seconds N] [--rounds N]\n",
                    argv[0]);
            return 1;
        }
    }
    if (rounds < 3 || rounds > BENCH_ROUNDS_MAX || !(rounds & 1))
        bench_fatal("rounds must be odd and between 3 and %d", BENCH_ROUNDS_MAX);
    if (run_moe + run_moe_signed + run_moe_kb128 + run_moe_slmacc + run_moe_direct + run_moe_coalesced + run_b5 + run_attention + run_attention_long + run_routing + run_reduce + run_activate
        + run_router + run_input + run_moe_tile + run_tile_only + run_fusion
        + run_qkv_post + run_glue + run_layer > 1
        || ((run_moe || run_attention || run_routing || run_reduce
             || run_activate || run_router || run_input || run_moe_tile)
            && selected_shape))
        bench_fatal("benchmark modes are mutually exclusive");
    if (!run_moe && !run_moe_tile && !run_routing && !run_reduce && !run_activate
        && selected_distribution)
        bench_fatal("--distribution requires --moe, --routing, --reduce or --activate");
    if (!run_attention && selected_attention)
        bench_fatal("--attention-shape requires --attention");
    if (!run_activate && selected_activation)
        bench_fatal("--activate-shape requires --activate");
    if (run_tile_only && !selected_shape)
        bench_fatal("--tile-only requires --shape");
    if (run_fusion && !selected_shape)
        bench_fatal("--fusion requires --shape dense, swa or global");

    bench_gpu gpu;
    bench_gpu_init(&gpu, spv_path);
    if (run_moe) {
        bench_moe_run(&gpu, selected_distribution, rounds, seconds);
    } else if (run_moe_signed) {
        bench_moe_pair_run(&gpu, BENCH_MOE_TN64_SIGNED, "signed", rounds,
                           seconds);
    } else if (run_moe_kb128) {
        bench_moe_pair_run(&gpu, BENCH_MOE_KB128, "kb128", rounds, seconds);
    } else if (run_moe_slmacc) {
        bench_moe_pair_run(&gpu, BENCH_MOE_TN64_SLMACC, "slmacc", rounds,
                           seconds);
    } else if (run_moe_direct) {
        bench_moe_pair_run(&gpu, BENCH_MOE_TN64_DIRECT, "direct", rounds,
                           seconds);
    } else if (run_moe_coalesced) {
        bench_moe_pair_run(&gpu, BENCH_MOE_TN64_COALESCED, "coalesced", rounds,
                           seconds);
    } else if (run_b5) {
        bench_b5_run(&gpu, rounds, seconds);
    } else if (run_moe_tile) {
        bench_moe_tile_run(&gpu, selected_distribution, rounds, seconds);
    } else if (run_routing) {
        bench_route_run(&gpu, selected_distribution, rounds, seconds);
    } else if (run_reduce) {
        bench_reduce_run(&gpu, selected_distribution, rounds, seconds);
    } else if (run_activate) {
        bench_activate_run(&gpu, selected_distribution, selected_activation,
                           rounds, seconds);
    } else if (run_router) {
        bench_router_run(&gpu, rounds, seconds);
    } else if (run_input) {
        bench_input_run(&gpu, rounds, seconds);
    } else if (run_tile_only) {
        int matched = 0;
        for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
            if (strcmp(selected_shape, shapes[i].name)) continue;
            bench_shape_tile_run(&gpu, &shapes[i], rounds, seconds);
            matched = 1;
        }
        if (!matched) bench_fatal("unknown shape: %s", selected_shape);
    } else if (run_fusion) {
        bench_fusion_run(&gpu, selected_shape, rounds, seconds);
    } else if (run_qkv_post) {
        bench_qkv_post_run(&gpu, selected_qkv, rounds, seconds);
    } else if (run_glue) {
        bench_glue_run(&gpu, selected_glue, rounds, seconds);
    } else if (run_layer) {
        bench_layer_run(&gpu, selected_layer, rounds, seconds);
    } else if (run_attention) {
        bench_attention_run(&gpu, selected_attention, rounds, seconds);
    } else if (run_attention_long) {
        uint64_t score_bytes = (uint64_t)attention_long_rows
                               * attention_long_keys * 16 * sizeof(float);
        if (attention_long_rows > 512 || attention_long_keys > 262144
            || attention_long_rows > attention_long_keys
            || score_bytes > 512ULL * 1024 * 1024
            || attention_long_chunk < 256 || attention_long_chunk > 16384
            || (attention_long_chunk & (attention_long_chunk - 1)))
            bench_fatal("long attention requires M<=512, M<=N<=262144 and scores<=512MiB");
        bench_attention_long_run(&gpu, attention_long_rows,
                                 attention_long_keys, attention_long_chunk);
    } else {
        int matched = 0;
        for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++) {
            if (selected_shape && strcmp(selected_shape, shapes[i].name)) continue;
            bench_shape_run(&gpu, &shapes[i], rounds, seconds);
            matched = 1;
        }
        if (!matched) bench_fatal("unknown shape: %s", selected_shape);
    }
    bench_gpu_destroy(&gpu);
    return 0;
}
