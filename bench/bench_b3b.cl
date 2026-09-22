#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void b3b_q4_q8(__global const uchar *weights,
                        __global const half *weight_scales,
                        __global const char *activation,
                        __global const half *activation_scales,
                        __global const short *activation_sigma,
                        __global float *output) {
    int row = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int part = lane & 3;
    float sum = 0.0f;

    #pragma unroll
    for (int first = 0; first < 88; first += 4) {
        int block = first + (lane >> 2);
        int at = part * 4;
        uchar4 packed = vload4(0, weights + ((size_t)row * 88 + block) * 16 + at);
        char4 lo = vload4(0, activation + block * 32 + at);
        char4 hi = vload4(0, activation + block * 32 + 16 + at);
        int integer = dot(packed & (uchar4)(15), lo) + dot(packed >> (uchar4)(4), hi);
        integer += sub_group_shuffle_xor(integer, 1);
        integer += sub_group_shuffle_xor(integer, 2);
        if (part == 0) {
            integer -= 8 * (int)activation_sigma[block];
            sum += (float)integer * (float)weight_scales[(size_t)row * 88 + block]
                   * (float)activation_scales[block];
        }
    }

    sum = sub_group_reduce_add(sum);
    if (lane == 0) output[row] = sum;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void b14_read(__global const uint4 *weights,
                       ulong weight_vectors,
                       __global const uint4 *scales,
                       ulong scale_vectors,
                       __global uint *output) {
    size_t id = get_global_id(0);
    size_t count = get_global_size(0);
    uint4 sum = (uint4)(0);
    for (size_t i = id; i < weight_vectors; i += count) sum += weights[i];
    for (size_t i = id; i < scale_vectors; i += count) sum += scales[i];
    uint total = sum.x + sum.y + sum.z + sum.w;
    total = sub_group_reduce_add(total);
    if (get_sub_group_local_id() == 0)
        output[get_group_id(0) * 16 + get_sub_group_id()] = total;
}
