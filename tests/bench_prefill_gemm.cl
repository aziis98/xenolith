#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define TM 32
#define TN 32
#define KB 64

#define GEMM_ARGS __global const uchar *wq, __global const half *wd, \
                  __global const char *aq, __global const half *ad, \
                  __global const short *as, __global float *out, \
                  int m_count, int n_count, int blocks

#define LOAD_W_X8() do { \
    _Pragma("unroll") \
    for (int seg = 0; seg < 8; seg++) { \
        int lng = seg >> 1; \
        int lb = seg & 1; \
        int chunk = lid >> 5; \
        int rem = lid & 31; \
        int row8 = rem >> 2; \
        int j = rem & 3; \
        int ln = lng * 8 + row8; \
        lw[ln * 32 + lb * 16 + chunk * 4 + j] = \
            wq[((size_t)((n0 >> 3) + lng) * blocks + kb / 32 + lb) * 128 + lid]; \
    } \
} while (0)

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_x8_fused(GEMM_ARGS) {
    __local char la[TM * KB];
    __local uchar lw[TN * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[TN * 2];
    int lid = get_local_id(0);
    int wm = lid >> 4;
    int wn = lid & 15;
    int m0 = get_group_id(0) * TM;
    int n0 = get_group_id(1) * TN;
    float acc[4][2] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = m0 + lm;
            la[x] = gm < m_count ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        LOAD_W_X8();
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = gm < m_count ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < TN * 2) {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[((size_t)((n0 >> 3) + lng) * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 2; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int gm = m0 + wm + im * 8;
        if (gm >= m_count) continue;
        #pragma unroll
        for (int in = 0; in < 2; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_x8_tn64(GEMM_ARGS) {
    __local char la[TM * KB];
    __local uchar lw[64 * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int wm = lid >> 4;
    int wn = lid & 15;
    int m0 = get_group_id(0) * TM;
    int n0 = get_group_id(1) * 64;
    float acc[4][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = m0 + lm;
            la[x] = gm < m_count ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 16; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[((size_t)((n0 >> 3) + lng) * blocks + kb / 32 + lb)
                   * 128 + lid];
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = gm < m_count
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[((size_t)((n0 >> 3) + lng) * blocks + kb / 32 + lb)
                   * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int gm = m0 + wm + im * 8;
        if (gm >= m_count) continue;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_x8_tm64_tn64(GEMM_ARGS) {
    __local char la[64 * KB];
    __local uchar lw[64 * (KB / 2)];
    __local half lad[64 * 2];
    __local short las[64 * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int wm = lid >> 4;
    int wn = lid & 15;
    int m0 = get_group_id(0) * 64;
    int n0 = get_group_id(1) * 64;
    float acc[8][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < 64 * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = m0 + lm;
            la[x] = gm < m_count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 16; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[((size_t)((n0 >> 3) + lng) * blocks + kb / 32 + lb)
                   * 128 + lid];
        }
        {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = gm < m_count
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[((size_t)((n0 >> 3) + lng) * blocks + kb / 32 + lb)
                   * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 8; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 8; im++) {
        int gm = m0 + wm + im * 8;
        if (gm >= m_count) continue;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_x8_tn48(GEMM_ARGS) {
    __local char la[TM * KB];
    __local uchar lw[48 * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[48 * 2];
    int lid = get_local_id(0);
    int wm = lid >> 4;
    int wn = lid & 15;
    int m0 = get_group_id(0) * TM;
    int n0 = get_group_id(1) * 48;
    float acc[4][3] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = m0 + lm;
            la[x] = gm < m_count ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 48 * 32; x += 128) {
            int ln = x >> 5;
            int p = x & 31;
            int lb = p >> 4;
            int byte = p & 15;
            int chunk = byte >> 2;
            int j = byte & 3;
            int gn = n0 + ln;
            lw[x] = gn < n_count
                    ? wq[((size_t)(gn >> 3) * blocks + kb / 32 + lb) * 128
                         + chunk * 32 + (gn & 7) * 4 + j] : 0;
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = gm < m_count
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < 48 * 2) {
            int ln = lid >> 1;
            int lb = lid & 1;
            int gn = n0 + ln;
            lwd[lid] = gn < n_count
                       ? wd[((size_t)(gn >> 3) * blocks + kb / 32 + lb) * 8
                            + (gn & 7)] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 3; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int gm = m0 + wm + im * 8;
        if (gm >= m_count) continue;
        #pragma unroll
        for (int in = 0; in < 3; in++) {
            int gn = n0 + wn + in * 16;
            if (gn < n_count)
                out[(size_t)gm * n_count + gn] = acc[im][in];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_x8_tn64_wg256(GEMM_ARGS) {
    __local char la[TM * KB];
    __local uchar lw[64 * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int wm = lid >> 4;
    int wn = lid & 15;
    int m0 = get_group_id(0) * TM;
    int n0 = get_group_id(1) * 64;
    float acc[2][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 256) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = m0 + lm;
            la[x] = gm < m_count ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 64 * 32; x += 256) {
            int ln = x >> 5;
            int p = x & 31;
            int lb = p >> 4;
            int byte = p & 15;
            int chunk = byte >> 2;
            int j = byte & 3;
            lw[x] = wq[((size_t)((n0 + ln) >> 3) * blocks + kb / 32 + lb)
                       * 128 + chunk * 32 + (ln & 7) * 4 + j];
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = gm < m_count
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < 64 * 2) {
            int ln = lid >> 1;
            int lb = lid & 1;
            lwd[lid] = wd[((size_t)((n0 + ln) >> 3) * blocks
                           + kb / 32 + lb) * 8 + (ln & 7)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 2; im++) {
                int lm = wm + im * 16;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 2; im++) {
        int gm = m0 + wm + im * 16;
        if (gm >= m_count) continue;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_x8_tn128_wg256(GEMM_ARGS) {
    __local char la[TM * KB];
    __local uchar lw[128 * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[128 * 2];
    int lid = get_local_id(0);
    int wm = lid >> 4;
    int wn = lid & 15;
    int m0 = get_group_id(0) * TM;
    int n0 = get_group_id(1) * 128;
    float acc[2][8] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 256) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = m0 + lm;
            la[x] = gm < m_count ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 128 * 32; x += 256) {
            int ln = x >> 5;
            int p = x & 31;
            int lb = p >> 4;
            int byte = p & 15;
            int chunk = byte >> 2;
            int j = byte & 3;
            int gn = n0 + ln;
            lw[x] = gn < n_count
                    ? wq[((size_t)(gn >> 3) * blocks + kb / 32 + lb) * 128
                         + chunk * 32 + (gn & 7) * 4 + j] : 0;
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = gm < m_count
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        {
            int ln = lid >> 1;
            int lb = lid & 1;
            int gn = n0 + ln;
            lwd[lid] = gn < n_count
                       ? wd[((size_t)(gn >> 3) * blocks + kb / 32 + lb) * 8
                            + (gn & 7)] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 2; im++) {
                int lm = wm + im * 16;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 8; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 2; im++) {
        int gm = m0 + wm + im * 16;
        if (gm >= m_count) continue;
        #pragma unroll
        for (int in = 0; in < 8; in++) {
            int gn = n0 + wn + in * 16;
            if (gn < n_count)
                out[(size_t)gm * n_count + gn] = acc[im][in];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_row_fused(GEMM_ARGS) {
    __local char la[TM * KB];
    __local uchar lw[TN * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[TN * 2];
    int lid = get_local_id(0);
    int wm = lid >> 4;
    int wn = lid & 15;
    int m0 = get_group_id(0) * TM;
    int n0 = get_group_id(1) * TN;
    float acc[4][2] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = m0 + lm;
            la[x] = gm < m_count ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < TN * 32; x += 128) {
            int ln = x >> 5;
            int p = x & 31;
            int lb = p >> 4;
            int byte = p & 15;
            lw[x] = wq[((size_t)(n0 + ln) * blocks + kb / 32 + lb) * 16 + byte];
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = gm < m_count ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < TN * 2) {
            int ln = lid >> 1;
            int lb = lid & 1;
            lwd[lid] = wd[(size_t)(n0 + ln) * blocks + kb / 32 + lb];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 2; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int gm = m0 + wm + im * 8;
        if (gm >= m_count) continue;
        #pragma unroll
        for (int in = 0; in < 2; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_x8_raw(GEMM_ARGS) {
    __local char la[TM * KB];
    __local uchar lw[TN * (KB / 2)];
    __local half lad[TM * 2];
    __local half lwd[TN * 2];
    int lid = get_local_id(0);
    int wm = lid >> 4;
    int wn = lid & 15;
    int m0 = get_group_id(0) * TM;
    int n0 = get_group_id(1) * TN;
    float acc[4][2] = {{0.0f}};
    (void)as;

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = m0 + lm;
            la[x] = gm < m_count ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        LOAD_W_X8();
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = m0 + lm;
            lad[lid] = gm < m_count ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
        }
        if (lid < TN * 2) {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[((size_t)((n0 >> 3) + lng) * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 2; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)integer * da * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int gm = m0 + wm + im * 8;
        if (gm >= m_count) continue;
        #pragma unroll
        for (int in = 0; in < 2; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_correction(__global const half *wd,
                                      __global const half *ad,
                                      __global const short *as,
                                      __global float *out,
                                      int m_count,
                                      int n_count,
                                      int blocks) {
    __local float la[TM * 16];
    __local half lw[TN * 16];
    int lid = get_local_id(0);
    int wm = lid >> 4;
    int wn = lid & 15;
    int m0 = get_group_id(0) * TM;
    int n0 = get_group_id(1) * TN;
    float acc[4][2] = {{0.0f}};

    for (int b0 = 0; b0 < blocks; b0 += 16) {
        for (int x = lid; x < TM * 16; x += 128) {
            int lm = x >> 4;
            int lb = x & 15;
            int gm = m0 + lm;
            int gb = b0 + lb;
            la[x] = gm < m_count && gb < blocks
                    ? 8.0f * (float)as[(size_t)gm * blocks + gb]
                      * (float)ad[(size_t)gm * blocks + gb]
                    : 0.0f;
        }
        for (int x = lid; x < TN * 16; x += 128) {
            int ln = x >> 4;
            int lb = x & 15;
            int gn = n0 + ln;
            int gb = b0 + lb;
            lw[x] = gb < blocks
                    ? wd[((size_t)(gn >> 3) * blocks + gb) * 8 + (gn & 7)]
                    : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int im = 0; im < 4; im++) {
            int lm = wm + im * 8;
            #pragma unroll
            for (int in = 0; in < 2; in++) {
                int ln = wn + in * 16;
                #pragma unroll
                for (int lb = 0; lb < 16; lb++)
                    acc[im][in] += la[lm * 16 + lb] * (float)lw[ln * 16 + lb];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int gm = m0 + wm + im * 8;
        if (gm >= m_count) continue;
        #pragma unroll
        for (int in = 0; in < 2; in++) {
            int gn = n0 + wn + in * 16;
            out[(size_t)gm * n_count + gn] -= acc[im][in];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped32(__global const uchar *wq,
                                     __global const half *wd,
                                     __global const char *aq,
                                     __global const half *ad,
                                     __global const short *as,
                                     __global float *out,
                                     __global const int *expert_count,
                                     __global const int *token_offset,
                                     __global const int *tile_expert,
                                     __global const int *tile_m0,
                                     int n_count,
                                     int blocks) {
    __local char la[TM * KB];
    __local char lw[TN * KB];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[TN * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * TN;
    float acc[4][2] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 8; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            uchar packed =
                wq[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                     * blocks + kb / 32 + lb) * 128 + lid];
            int base = ln * KB + lb * 32 + chunk * 4 + j;
            lw[base] = (char)(packed & 15) - 8;
            lw[base + 16] = (char)(packed >> 4) - 8;
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < TN * 2) {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                    * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 2; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 8; c++)
                        integer += dot(
                            vload4(0, lw + ln * KB + lb * 32 + c * 4),
                            vload4(0, la + lm * KB + lb * 32 + c * 4));
                    acc[im][in] += (float)integer * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 2; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped64(__global const uchar *wq,
                                     __global const half *wd,
                                     __global const char *aq,
                                     __global const half *ad,
                                     __global const short *as,
                                     __global float *out,
                                     __global const int *expert_count,
                                     __global const int *token_offset,
                                     __global const int *tile_expert,
                                     __global const int *tile_m0,
                                     int n_count,
                                     int blocks) {
    __local char la[64 * KB];
    __local uchar lw[TN * (KB / 2)];
    __local half lad[64 * 2];
    __local short las[64 * 2];
    __local half lwd[TN * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * TN;
    float acc[8][2] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < 64 * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 8; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                     * blocks + kb / 32 + lb) * 128 + lid];
        }
        for (int x = lid; x < 64 * 2; x += 128) {
            int lm = x >> 1;
            int lb = x & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[x] = valid ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[x] = valid ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < TN * 2) {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                    * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 8; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 2; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 8; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 2; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tn64(__global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[TM * KB];
    __local uchar lw[64 * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 64;
    float acc[4][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 16; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                     * blocks + kb / 32 + lb) * 128 + lid];
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                    * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tn64_direct(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 64;
    float acc[4][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            int block = kb / 32 + lb;
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                if (local_m0 + lm >= count) continue;
                int gm = packed_m0 + lm;
                int correction = 8 * (int)as[(size_t)gm * blocks + block];
                float da = (float)ad[(size_t)gm * blocks + block];
                __global const char *activation =
                    aq + (size_t)gm * blocks * 32 + kb + lb * 32;
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int gn = n0 + wn + in * 16;
                    int group8 = gn >> 3;
                    int row8 = gn & 7;
                    __global const uchar *weight =
                        wq + (((size_t)expert * (n_count >> 3) + group8)
                              * blocks + block) * 128 + row8 * 4;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(0, weight + c * 32);
                        char4 lo = vload4(0, activation + c * 4);
                        char4 hi = vload4(0, activation + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    float dw = (float)wd[(((size_t)expert * (n_count >> 3)
                                            + group8) * blocks + block) * 8
                                          + row8];
                    acc[im][in] += (float)(integer - correction) * da * dw;
                }
            }
        }
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tn64_coalesced(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[TM * KB];
    __local uint lw[64 * 8];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 64;
    float acc[4][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 64 * 8; x += 128) {
            int ln = x >> 3;
            int q = x & 7;
            int lb = q >> 2;
            int c = q & 3;
            int gn = n0 + ln;
            __global const uint *source = (__global const uint *)(
                wq + (((size_t)expert * (n_count >> 3) + (gn >> 3))
                      * blocks + kb / 32 + lb) * 128
                + c * 32 + (gn & 7) * 4);
            lw[(lb * 4 + c) * 64 + ln] = *source;
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        {
            int ln = lid >> 1;
            int lb = lid & 1;
            int gn = n0 + ln;
            lwd[lb * 64 + ln] =
                wd[(((size_t)expert * (n_count >> 3) + (gn >> 3))
                    * blocks + kb / 32 + lb) * 8 + (gn & 7)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = as_uchar4(lw[(lb * 4 + c) * 64 + ln]);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[lb * 64 + ln];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tn64_slmacc(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[TM * KB];
    __local uchar lw[64 * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[64 * 2];
    __local float lacc[128 * 16];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 64;
    #pragma unroll
    for (int i = 0; i < 16; i++) lacc[lid * 16 + i] = 0.0f;

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 16; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                     * blocks + kb / 32 + lb) * 128 + lid];
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                    * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    int ai = lid * 16 + im * 4 + in;
                    lacc[ai] += (float)(integer - correction) * da
                                * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] =
                lacc[lid * 16 + im * 4 + in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tm16_tn64(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[16 * KB];
    __local uchar lw[64 * (KB / 2)];
    __local half lad[16 * 2];
    __local short las[16 * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 64;
    float acc[2][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < 16 * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 16; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                     * blocks + kb / 32 + lb) * 128 + lid];
        }
        if (lid < 16 * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                    * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 2; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 2; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tn48(__global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[TM * KB];
    __local uchar lw[48 * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[48 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 48;
    float acc[4][3] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 48 * 32; x += 128) {
            int ln = x >> 5;
            int p = x & 31;
            int lb = p >> 4;
            int byte = p & 15;
            int chunk = byte >> 2;
            int j = byte & 3;
            int gn = n0 + ln;
            lw[x] = gn < n_count
                    ? wq[(((size_t)expert * (n_count >> 3) + (gn >> 3))
                           * blocks + kb / 32 + lb) * 128 + chunk * 32
                         + (gn & 7) * 4 + j] : 0;
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < 48 * 2) {
            int ln = lid >> 1;
            int lb = lid & 1;
            int gn = n0 + ln;
            lwd[lid] = gn < n_count
                       ? wd[(((size_t)expert * (n_count >> 3) + (gn >> 3))
                              * blocks + kb / 32 + lb) * 8 + (gn & 7)]
                       : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 3; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 3; in++) {
            int gn = n0 + wn + in * 16;
            if (gn < n_count)
                out[(size_t)gm * n_count + gn] = acc[im][in];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tn64_wg256(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[TM * KB];
    __local uchar lw[64 * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 64;
    float acc[2][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 256) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 64 * 32; x += 256) {
            int ln = x >> 5;
            int p = x & 31;
            int lb = p >> 4;
            int byte = p & 15;
            int chunk = byte >> 2;
            int j = byte & 3;
            lw[x] = wq[(((size_t)expert * (n_count >> 3)
                         + ((n0 + ln) >> 3)) * blocks + kb / 32 + lb)
                       * 128 + chunk * 32 + (ln & 7) * 4 + j];
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < 64 * 2) {
            int ln = lid >> 1;
            int lb = lid & 1;
            lwd[lid] = wd[(((size_t)expert * (n_count >> 3)
                            + ((n0 + ln) >> 3)) * blocks + kb / 32 + lb)
                          * 8 + (ln & 7)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 2; im++) {
                int lm = wm + im * 16;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 2; im++) {
        int lm = wm + im * 16;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tm64_tn64_wg256(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[64 * KB];
    __local uchar lw[64 * (KB / 2)];
    __local half lad[64 * 2];
    __local short las[64 * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 64;
    float acc[4][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < 64 * KB; x += 256) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 64 * 32; x += 256) {
            int ln = x >> 5;
            int p = x & 31;
            int lb = p >> 4;
            int byte = p & 15;
            int chunk = byte >> 2;
            int j = byte & 3;
            lw[x] = wq[(((size_t)expert * (n_count >> 3)
                         + ((n0 + ln) >> 3)) * blocks + kb / 32 + lb)
                       * 128 + chunk * 32 + (ln & 7) * 4 + j];
        }
        if (lid < 64 * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < 64 * 2) {
            int ln = lid >> 1;
            int lb = lid & 1;
            lwd[lid] = wd[(((size_t)expert * (n_count >> 3)
                            + ((n0 + ln) >> 3)) * blocks + kb / 32 + lb)
                          * 8 + (ln & 7)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 16;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 16;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tm24_tn64(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[24 * KB];
    __local uchar lw[64 * (KB / 2)];
    __local half lad[24 * 2];
    __local short las[24 * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 64;
    float acc[3][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < 24 * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 16; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                     * blocks + kb / 32 + lb) * 128 + lid];
        }
        if (lid < 24 * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                    * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 3; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 3; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tn64_signed(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[TM * KB];
    __local char lw[64 * KB];
    __local half lad[TM * 2];
    __local half lwd[64 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 64;
    float acc[4][4] = {{0.0f}};
    (void)as;

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 16; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            uchar packed =
                wq[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                     * blocks + kb / 32 + lb) * 128 + lid];
            int base = ln * KB + lb * 32 + chunk * 4 + j;
            lw[base] = (char)(packed & 15) - 8;
            lw[base + 16] = (char)(packed >> 4) - 8;
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            lad[lid] = local_m0 + lm < count
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
        }
        {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                    * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 8; c++)
                        integer += dot(
                            vload4(0, lw + ln * KB + lb * 32 + c * 4),
                            vload4(0, la + lm * KB + lb * 32 + c * 4));
                    acc[im][in] += (float)integer * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tn64_kb128(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[TM * 128];
    __local uchar lw[64 * 64];
    __local half lad[TM * 4];
    __local short las[TM * 4];
    __local half lwd[64 * 4];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 64;
    float acc[4][4] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += 128) {
        for (int x = lid; x < TM * 128; x += 128) {
            int lm = x / 128;
            int lk = x - lm * 128;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count && kb + lk < blocks * 32
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 64 * 64; x += 128) {
            int ln = x >> 6;
            int p = x & 63;
            int lb = p >> 4;
            int byte = p & 15;
            int chunk = byte >> 2;
            int j = byte & 3;
            int gb = kb / 32 + lb;
            lw[x] = gb < blocks
                    ? wq[(((size_t)expert * (n_count >> 3)
                           + ((n0 + ln) >> 3)) * blocks + gb)
                         * 128 + chunk * 32 + (ln & 7) * 4 + j] : 0;
        }
        for (int x = lid; x < TM * 4; x += 128) {
            int lm = x >> 2;
            int lb = x & 3;
            int gm = packed_m0 + lm;
            int gb = kb / 32 + lb;
            int valid = local_m0 + lm < count && gb < blocks;
            lad[x] = valid
                     ? ad[(size_t)gm * blocks + gb] : (half)0;
            las[x] = valid
                     ? as[(size_t)gm * blocks + gb] : (short)0;
        }
        for (int x = lid; x < 64 * 4; x += 128) {
            int ln = x >> 2;
            int lb = x & 3;
            int gb = kb / 32 + lb;
            lwd[x] = gb < blocks
                     ? wd[(((size_t)expert * (n_count >> 3)
                            + ((n0 + ln) >> 3)) * blocks + gb)
                          * 8 + (ln & 7)] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 4; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 4 + lb];
                float da = (float)lad[lm * 4 + lb];
                #pragma unroll
                for (int in = 0; in < 4; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 64 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * 128 + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * 128 + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 4 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 4; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_tn128_wg256(
                                        __global const uchar *wq,
                                        __global const half *wd,
                                        __global const char *aq,
                                        __global const half *ad,
                                        __global const short *as,
                                        __global float *out,
                                        __global const int *expert_count,
                                        __global const int *token_offset,
                                        __global const int *tile_expert,
                                        __global const int *tile_m0,
                                        int n_count,
                                        int blocks) {
    __local char la[TM * KB];
    __local uchar lw[128 * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[128 * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * 128;
    float acc[2][8] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 256) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        for (int x = lid; x < 128 * 32; x += 256) {
            int ln = x >> 5;
            int p = x & 31;
            int lb = p >> 4;
            int byte = p & 15;
            int chunk = byte >> 2;
            int j = byte & 3;
            int gn = n0 + ln;
            lw[x] = gn < n_count
                    ? wq[(((size_t)expert * (n_count >> 3) + (gn >> 3))
                           * blocks + kb / 32 + lb) * 128 + chunk * 32
                         + (gn & 7) * 4 + j] : 0;
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid
                       ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        {
            int ln = lid >> 1;
            int lb = lid & 1;
            int gn = n0 + ln;
            lwd[lid] = gn < n_count
                       ? wd[(((size_t)expert * (n_count >> 3) + (gn >> 3))
                              * blocks + kb / 32 + lb) * 8 + (gn & 7)]
                       : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 2; im++) {
                int lm = wm + im * 16;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 8; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(
                            0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(
                            0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(
                            0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 2; im++) {
        int lm = wm + im * 16;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 8; in++) {
            int gn = n0 + wn + in * 16;
            if (gn < n_count)
                out[(size_t)gm * n_count + gn] = acc[im][in];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_expert(__global const uchar *wq,
                                          __global const half *wd,
                                          __global const char *aq,
                                          __global const half *ad,
                                          __global const short *as,
                                          __global float *out,
                                          __global const int *expert_count,
                                          __global const int *token_offset,
                                          int n_count,
                                          int blocks) {
    __local char la[TM * KB];
    __local uchar lw[TN * 88 * 16];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[TN * 88];
    int lid = get_local_id(0);
    int expert = get_group_id(0);
    int count = expert_count[expert];
    int packed_base = token_offset[expert];
    int n0 = get_group_id(1) * TN;
    int wm = lid >> 4;
    int wn = lid & 15;
    size_t weight_base = ((size_t)expert * (n_count >> 3) + (n0 >> 3))
                         * blocks * 128;
    size_t scale_base = ((size_t)expert * (n_count >> 3) + (n0 >> 3))
                        * blocks * 8;

    for (int x = lid; x < (TN >> 3) * blocks * 128; x += 128)
        lw[x] = wq[weight_base + x];
    for (int x = lid; x < TN * blocks; x += 128)
        lwd[x] = wd[scale_base + x];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int local_m0 = 0; local_m0 < count; local_m0 += TM) {
        float acc[4][2] = {{0.0f}};
        for (int kb = 0; kb < blocks * 32; kb += KB) {
            for (int x = lid; x < TM * KB; x += 128) {
                int lm = x / KB;
                int lk = x - lm * KB;
                int gm = packed_base + local_m0 + lm;
                la[x] = local_m0 + lm < count
                        ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
            }
            if (lid < TM * 2) {
                int lm = lid >> 1;
                int lb = lid & 1;
                int gm = packed_base + local_m0 + lm;
                int valid = local_m0 + lm < count;
                lad[lid] = valid
                           ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
                las[lid] = valid
                           ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (int lb = 0; lb < 2; lb++) {
                int block = kb / 32 + lb;
                #pragma unroll
                for (int im = 0; im < 4; im++) {
                    int lm = wm + im * 8;
                    int correction = 8 * (int)las[lm * 2 + lb];
                    float da = (float)lad[lm * 2 + lb];
                    #pragma unroll
                    for (int in = 0; in < 2; in++) {
                        int ln = wn + in * 16;
                        int group = ln >> 3;
                        int row8 = ln & 7;
                        int integer = 0;
                        #pragma unroll
                        for (int c = 0; c < 4; c++) {
                            uchar4 packed = vload4(
                                0, lw + (group * blocks + block) * 128
                                   + c * 32 + row8 * 4);
                            char4 lo = vload4(
                                0, la + lm * KB + lb * 32 + c * 4);
                            char4 hi = vload4(
                                0, la + lm * KB + lb * 32 + 16 + c * 4);
                            integer += dot(packed & (uchar4)(15), lo)
                                       + dot(packed >> (uchar4)(4), hi);
                        }
                        half dw = lwd[(group * blocks + block) * 8 + row8];
                        acc[im][in] += (float)(integer - correction) * da
                                       * (float)dw;
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        #pragma unroll
        for (int im = 0; im < 4; im++) {
            int lm = wm + im * 8;
            if (local_m0 + lm >= count) continue;
            int gm = packed_base + local_m0 + lm;
            #pragma unroll
            for (int in = 0; in < 2; in++)
                out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_tail32(__global const uchar *wq,
                                  __global const half *wd,
                                  __global const char *aq,
                                  __global const half *ad,
                                  __global const short *as,
                                  __global float *out,
                                  __global const int *expert_count,
                                  __global const int *token_offset,
                                  __global const int *tile_expert,
                                  __global const int *tile_m0,
                                  int n_count,
                                  int blocks) {
    __local char la[TM * KB];
    __local uchar lw[TN * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[TN * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * TN;
    float acc[4][2] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int gm = packed_m0 + lm;
            la[x] = local_m0 + lm < count
                    ? aq[(size_t)gm * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 8; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                     * blocks + kb / 32 + lb) * 128 + lid];
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int gm = packed_m0 + lm;
            int valid = local_m0 + lm < count;
            lad[lid] = valid ? ad[(size_t)gm * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid ? as[(size_t)gm * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < TN * 2) {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                    * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                if (local_m0 + lm >= count) continue;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 2; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 2; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__kernel void prefill_route_reset(__global int *expert_count,
                                  __global int *cursor,
                                  __global int *tile_expert,
                                  __global int *tile_m0) {
    int index = get_global_id(0);
    if (index < 128) {
        expert_count[index] = 0;
        cursor[index] = 0;
    }
    if (index < 256) {
        tile_expert[index] = -1;
        tile_m0[index] = 0;
    }
}

__kernel void prefill_route_count(__global const int *route_expert,
                                  __global int *expert_count,
                                  int routes) {
    int route = get_global_id(0);
    if (route < routes)
        atomic_inc((volatile __global unsigned int *)&expert_count[route_expert[route]]);
}

__kernel void prefill_route_prefix(__global const int *expert_count,
                                   __global int *token_offset,
                                   __global int *cursor,
                                   __global int *tile_expert,
                                   __global int *tile_m0) {
    if (get_global_id(0) != 0) return;
    int offset = 0;
    int tile = 0;
    token_offset[0] = 0;
    for (int expert = 0; expert < 128; expert++) {
        int count = expert_count[expert];
        cursor[expert] = offset;
        offset += count;
        token_offset[expert + 1] = offset;
        for (int m0 = 0; m0 < count; m0 += 32) {
            tile_expert[tile] = expert;
            tile_m0[tile] = m0;
            tile++;
        }
    }
}

__kernel void prefill_route_scatter(__global const int *route_expert,
                                    __global int *cursor,
                                    __global int *packed_route,
                                    __global int *route_packed,
                                    int routes) {
    int route = get_global_id(0);
    if (route >= routes) return;
    int expert = route_expert[route];
    unsigned int packed = atomic_inc(
        (volatile __global unsigned int *)&cursor[expert]);
    packed_route[packed] = route;
    route_packed[route] = packed;
}

__kernel void prefill_route_pack(__global const char *source_q,
                                 __global const half *source_d,
                                 __global const short *source_s,
                                 __global char *packed_q,
                                 __global half *packed_d,
                                 __global short *packed_s,
                                 __global const int *route_token,
                                 __global const int *packed_route,
                                 int blocks,
                                 int routes) {
    int packed = get_group_id(0);
    int lid = get_local_id(0);
    if (packed >= routes) return;
    int token = route_token[packed_route[packed]];
    for (int k = lid; k < blocks * 32; k += 128)
        packed_q[(size_t)packed * blocks * 32 + k] =
            source_q[(size_t)token * blocks * 32 + k];
    for (int block = lid; block < blocks; block += 128) {
        packed_d[(size_t)packed * blocks + block] =
            source_d[(size_t)token * blocks + block];
        packed_s[(size_t)packed * blocks + block] =
            source_s[(size_t)token * blocks + block];
    }
}

__kernel void prefill_route_reduce(__global const float *routed,
                                   __global const float *route_weight,
                                   __global const int *route_packed,
                                   __global float *reduced,
                                   int dimension) {
    int token = get_group_id(0);
    int lid = get_local_id(0);
    for (int d = lid; d < dimension; d += 128) {
        float sum = 0.0f;
        for (int slot = 0; slot < 8; slot++) {
            int route = token * 8 + slot;
            int packed = route_packed[route];
            sum += route_weight[route]
                   * routed[(size_t)packed * dimension + d];
        }
        reduced[(size_t)token * dimension + d] = sum;
    }
}

__kernel void prefill_route_reduce_scaled(
                                   __global const float *routed,
                                   __global const float *route_weight,
                                   __global const int *route_expert,
                                   __global const int *route_packed,
                                   __global const float *expert_scale,
                                   __global float *reduced,
                                   int dimension) {
    int token = get_group_id(0);
    int lid = get_local_id(0);
    for (int d = lid; d < dimension; d += 128) {
        float sum = 0.0f;
        for (int slot = 0; slot < 8; slot++) {
            int route = token * 8 + slot;
            int packed = route_packed[route];
            sum += route_weight[route] * expert_scale[route_expert[route]]
                   * routed[(size_t)packed * dimension + d];
        }
        reduced[(size_t)token * dimension + d] = sum;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_geglu_q8(__global const float *gate_up,
                               __global char *quantized,
                               __global half *scale,
                               __global short *sigma,
                               int rows,
                               int width) {
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    int column = block * 32 + lane;
    float values[2];
    #pragma unroll
    for (int half_index = 0; half_index < 2; half_index++) {
        int j = column + half_index * 16;
        float gate = gate_up[(size_t)row * width * 2 + j];
        float up = gate_up[(size_t)row * width * 2 + width + j];
        float activated;
        if (gate <= -10.0f) {
            activated = 0.0f;
        } else if (gate >= 10.0f) {
            activated = gate;
        } else {
            float x = (float)convert_half(gate);
            float inner = 0.7978845608028654f
                          * (x + 0.044715f * x * x * x);
            activated = (float)convert_half(0.5f * x * (1.0f + tanh(inner)));
        }
        values[half_index] = activated * up;
    }
    float maximum = fmax(fabs(values[0]), fabs(values[1]));
    maximum = sub_group_reduce_max(maximum);
    float d = maximum / 127.0f;
    float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    int q0 = convert_int(round(values[0] * inverse));
    int q1 = convert_int(round(values[1] * inverse));
    size_t qbase = ((size_t)row * block_count + block) * 32;
    quantized[qbase + lane] = (char)q0;
    quantized[qbase + lane + 16] = (char)q1;
    int sum = sub_group_reduce_add(q0 + q1);
    if (lane == 0) {
        scale[(size_t)row * block_count + block] = (half)d;
        sigma[(size_t)row * block_count + block] = (short)sum;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_geglu_q8_pair(__global const float *gate,
                                    __global const float *up,
                                    __global char *quantized,
                                    __global half *scale,
                                    __global short *sigma,
                                    int rows,
                                    int width) {
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    int column = block * 32 + lane;
    float values[2];
    #pragma unroll
    for (int half_index = 0; half_index < 2; half_index++) {
        int j = column + half_index * 16;
        float x = gate[(size_t)row * width + j];
        float activated;
        if (x <= -10.0f) activated = 0.0f;
        else if (x >= 10.0f) activated = x;
        else {
            float h = (float)convert_half(x);
            float inner = 0.7978845608028654f
                          * (h + 0.044715f * h * h * h);
            activated = (float)convert_half(
                0.5f * h * (1.0f + tanh(inner)));
        }
        values[half_index] = activated * up[(size_t)row * width + j];
    }
    float maximum = sub_group_reduce_max(
        fmax(fabs(values[0]), fabs(values[1])));
    float d = maximum / 127.0f;
    float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    int q0 = convert_int(round(values[0] * inverse));
    int q1 = convert_int(round(values[1] * inverse));
    size_t base = ((size_t)row * block_count + block) * 32;
    quantized[base + lane] = (char)q0;
    quantized[base + lane + 16] = (char)q1;
    int sum = sub_group_reduce_add(q0 + q1);
    if (lane == 0) {
        size_t index = (size_t)row * block_count + block;
        scale[index] = (half)d;
        sigma[index] = (short)sum;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_router_gemm(__global const float *input,
                                  __global const float *weight,
                                  __global float *logits,
                                  int rows,
                                  int width,
                                  int experts) {
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / experts;
    int expert = task - row * experts;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int k = lane; k < width; k += 16)
        sum += input[(size_t)row * width + k]
               * weight[(size_t)expert * width + k];
    sum = sub_group_reduce_add(sum);
    if (lane == 0) logits[(size_t)row * experts + expert] = sum;
}

__kernel void prefill_router_gemm_tiled(__global const float *input,
                                        __global const float *weight,
                                        __global float *logits,
                                        int rows,
                                        int width,
                                        int experts) {
    __local float local_input[16 * 32];
    __local float local_weight[16 * 32];
    int lid = get_local_id(0);
    int local_row = lid >> 3;
    int local_column = (lid & 7) * 2;
    int row = get_group_id(0) * 16 + local_row;
    int column0 = get_group_id(1) * 16 + local_column;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int k0 = 0; k0 < width; k0 += 32) {
        for (int x = lid; x < 16 * 32; x += 128) {
            int tile_row = x >> 5;
            int k = x & 31;
            local_input[x] = input[(size_t)(get_group_id(0) * 16 + tile_row)
                                   * width + k0 + k];
            local_weight[x] = weight[(size_t)(get_group_id(1) * 16 + tile_row)
                                     * width + k0 + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            float value = local_input[local_row * 32 + k];
            sum0 += value * local_weight[local_column * 32 + k];
            sum1 += value * local_weight[(local_column + 1) * 32 + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < rows) {
        logits[(size_t)row * experts + column0] = sum0;
        logits[(size_t)row * experts + column0 + 1] = sum1;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_router_top8(__global const float *logits,
                                  __global int *route_expert,
                                  __global float *route_weight,
                                  int rows,
                                  int experts) {
    int row = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    if (row >= rows) return;
    float values[8];
    uchar used[8] = {0};
    #pragma unroll
    for (int i = 0; i < 8; i++)
        values[i] = logits[(size_t)row * experts + lane * 8 + i];
    float selected_value[8];
    int selected_index[8];
    #pragma unroll
    for (int slot = 0; slot < 8; slot++) {
        float best = -INFINITY;
        int index = 0x7fffffff;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int expert = lane * 8 + i;
            if (!used[i] && (values[i] > best
                             || (values[i] == best && expert < index))) {
                best = values[i];
                index = expert;
            }
        }
        float global_best = sub_group_reduce_max(best);
        int candidate = best == global_best ? index : 0x7fffffff;
        int global_index = sub_group_reduce_min(candidate);
        if (global_index / 8 == lane) used[global_index & 7] = 1;
        selected_value[slot] = global_best;
        selected_index[slot] = global_index;
    }
    if (lane == 0) {
        float sum = 0.0f;
        float weights[8];
        #pragma unroll
        for (int slot = 0; slot < 8; slot++) {
            weights[slot] = exp(selected_value[slot] - selected_value[0]);
            sum += weights[slot];
        }
        #pragma unroll
        for (int slot = 0; slot < 8; slot++) {
            route_expert[row * 8 + slot] = selected_index[slot];
            route_weight[row * 8 + slot] = weights[slot] / sum;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_ffn_input_q8(__global const float *input,
                                   __global const float *dense_weight,
                                   __global const float *moe_weight,
                                   __global const float *router_weight,
                                   __global char *dense_q,
                                   __global half *dense_d,
                                   __global short *dense_s,
                                   __global char *moe_q,
                                   __global half *moe_d,
                                   __global short *moe_s,
                                   __global float *router_input,
                                   float input_scale,
                                   float router_scale,
                                   int rows,
                                   int width) {
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    float dense[2];
    float moe[2];
    #pragma unroll
    for (int half_index = 0; half_index < 2; half_index++) {
        int column = block * 32 + lane + half_index * 16;
        float x = input[(size_t)row * width + column];
        dense[half_index] = x * input_scale * dense_weight[column];
        moe[half_index] = x * input_scale * moe_weight[column];
        router_input[(size_t)row * width + column] =
            x * router_scale * router_weight[column];
    }
    float dense_max = sub_group_reduce_max(
        fmax(fabs(dense[0]), fabs(dense[1])));
    float moe_max = sub_group_reduce_max(fmax(fabs(moe[0]), fabs(moe[1])));
    float dd = dense_max / 127.0f;
    float md = moe_max / 127.0f;
    float dense_inverse = dd == 0.0f ? 0.0f : 1.0f / dd;
    float moe_inverse = md == 0.0f ? 0.0f : 1.0f / md;
    int dense_q0 = convert_int(round(dense[0] * dense_inverse));
    int dense_q1 = convert_int(round(dense[1] * dense_inverse));
    int moe_q0 = convert_int(round(moe[0] * moe_inverse));
    int moe_q1 = convert_int(round(moe[1] * moe_inverse));
    size_t base = ((size_t)row * block_count + block) * 32;
    dense_q[base + lane] = (char)dense_q0;
    dense_q[base + lane + 16] = (char)dense_q1;
    moe_q[base + lane] = (char)moe_q0;
    moe_q[base + lane + 16] = (char)moe_q1;
    int dense_sum = sub_group_reduce_add(dense_q0 + dense_q1);
    int moe_sum = sub_group_reduce_add(moe_q0 + moe_q1);
    if (lane == 0) {
        size_t scale_index = (size_t)row * block_count + block;
        dense_d[scale_index] = (half)dd;
        dense_s[scale_index] = (short)dense_sum;
        moe_d[scale_index] = (half)md;
        moe_s[scale_index] = (short)moe_sum;
    }
}

__kernel void prefill_rms_scale(__global const float *input,
                                __global float *scale,
                                int rows,
                                int width,
                                float epsilon) {
    __local float partial[128];
    int row = get_group_id(0);
    int lid = get_local_id(0);
    if (row >= rows) return;
    float sum = 0.0f;
    for (int column = lid; column < width; column += 128) {
        float value = input[(size_t)row * width + column];
        sum += value * value;
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0)
        scale[row] = rsqrt(partial[0] / (float)width + epsilon);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_norm_q8(__global const float *input,
                              __global const float *weight,
                              __global const float *row_scale,
                              __global char *quantized,
                              __global half *scale,
                              __global short *sigma,
                              int rows,
                              int width) {
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    float values[2];
    #pragma unroll
    for (int half_index = 0; half_index < 2; half_index++) {
        int column = block * 32 + lane + half_index * 16;
        values[half_index] = input[(size_t)row * width + column]
                             * row_scale[row] * weight[column];
    }
    float maximum = sub_group_reduce_max(
        fmax(fabs(values[0]), fabs(values[1])));
    float d = maximum / 127.0f;
    float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    int q0 = convert_int(round(values[0] * inverse));
    int q1 = convert_int(round(values[1] * inverse));
    size_t base = ((size_t)row * block_count + block) * 32;
    quantized[base + lane] = (char)q0;
    quantized[base + lane + 16] = (char)q1;
    int sum = sub_group_reduce_add(q0 + q1);
    if (lane == 0) {
        size_t index = (size_t)row * block_count + block;
        scale[index] = (half)d;
        sigma[index] = (short)sum;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_ffn_input_q8_rowscale(
                                   __global const float *input,
                                   __global const float *dense_weight,
                                   __global const float *moe_weight,
                                   __global const float *router_weight,
                                   __global const float *row_scale,
                                   __global char *dense_q,
                                   __global half *dense_d,
                                   __global short *dense_s,
                                   __global char *moe_q,
                                   __global half *moe_d,
                                   __global short *moe_s,
                                   __global float *router_input,
                                   float router_scale,
                                   int rows,
                                   int width) {
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    float dense[2];
    float moe[2];
    float rms = row_scale[row];
    #pragma unroll
    for (int half_index = 0; half_index < 2; half_index++) {
        int column = block * 32 + lane + half_index * 16;
        float x = input[(size_t)row * width + column];
        dense[half_index] = x * rms * dense_weight[column];
        moe[half_index] = x * rms * moe_weight[column];
        router_input[(size_t)row * width + column] =
            x * rms * router_scale * router_weight[column];
    }
    float dense_max = sub_group_reduce_max(
        fmax(fabs(dense[0]), fabs(dense[1])));
    float moe_max = sub_group_reduce_max(fmax(fabs(moe[0]), fabs(moe[1])));
    float dd = dense_max / 127.0f;
    float md = moe_max / 127.0f;
    float dense_inverse = dd == 0.0f ? 0.0f : 1.0f / dd;
    float moe_inverse = md == 0.0f ? 0.0f : 1.0f / md;
    int dense_q0 = convert_int(round(dense[0] * dense_inverse));
    int dense_q1 = convert_int(round(dense[1] * dense_inverse));
    int moe_q0 = convert_int(round(moe[0] * moe_inverse));
    int moe_q1 = convert_int(round(moe[1] * moe_inverse));
    size_t base = ((size_t)row * block_count + block) * 32;
    dense_q[base + lane] = (char)dense_q0;
    dense_q[base + lane + 16] = (char)dense_q1;
    moe_q[base + lane] = (char)moe_q0;
    moe_q[base + lane + 16] = (char)moe_q1;
    int dense_sum = sub_group_reduce_add(dense_q0 + dense_q1);
    int moe_sum = sub_group_reduce_add(moe_q0 + moe_q1);
    if (lane == 0) {
        size_t index = (size_t)row * block_count + block;
        dense_d[index] = (half)dd;
        dense_s[index] = (short)dense_sum;
        moe_d[index] = (half)md;
        moe_s[index] = (short)moe_sum;
    }
}

__kernel void prefill_qkv_post(__global const float *q_projection,
                               __global const float *k_projection,
                               __global const float *v_projection,
                               __global const float *q_weight,
                               __global const float *k_weight,
                               __global const float *rope_cos,
                               __global const float *rope_sin,
                               __global float *q,
                               __global half *k,
                               __global half *v,
                               int rows,
                               int dimension,
                               int kv_heads,
                               int has_v,
                               float epsilon) {
    __local float partial[128];
    int task = get_group_id(0);
    int lid = get_local_id(0);
    int per_row = 16 + 2 * kv_heads;
    int row = task / per_row;
    int local_task = task - row * per_row;
    if (row >= rows) return;
    int kind = local_task < 16 ? 0
               : local_task < 16 + kv_heads ? 1 : 2;
    int head = kind == 0 ? local_task
               : kind == 1 ? local_task - 16
               : local_task - 16 - kv_heads;
    __global const float *source = kind == 0 ? q_projection
                                   : kind == 1 || !has_v ? k_projection
                                   : v_projection;
    int source_heads = kind == 0 ? 16 : kv_heads;
    size_t source_base = ((size_t)row * source_heads + head) * dimension;
    float sum = 0.0f;
    for (int d = lid; d < dimension; d += 128) {
        float value = source[source_base + d];
        sum += value * value;
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = rsqrt(partial[0] / (float)dimension + epsilon);
    if (kind == 2) {
        for (int d = lid; d < dimension; d += 128)
            v[((size_t)head * rows + row) * dimension + d] =
                (half)(source[source_base + d] * scale);
        return;
    }
    __global const float *weight = kind == 0 ? q_weight : k_weight;
    int half_dimension = dimension / 2;
    for (int d = lid; d < half_dimension; d += 128) {
        float lo = source[source_base + d] * scale * weight[d];
        float hi = source[source_base + d + half_dimension] * scale
                   * weight[d + half_dimension];
        float cosine = rope_cos[(size_t)row * half_dimension + d];
        float sine = rope_sin[(size_t)row * half_dimension + d];
        float rotated_lo = lo * cosine - hi * sine;
        float rotated_hi = lo * sine + hi * cosine;
        if (kind == 0) {
            size_t base = ((size_t)head * rows + row) * dimension;
            q[base + d] = rotated_lo;
            q[base + d + half_dimension] = rotated_hi;
        } else {
            size_t base = ((size_t)head * rows + row) * dimension;
            k[base + d] = (half)rotated_lo;
            k[base + d + half_dimension] = (half)rotated_hi;
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_heads_q8(__global const float *heads,
                               __global char *quantized,
                               __global half *scale,
                               __global short *sigma,
                               int rows,
                               int head_count,
                               int dimension) {
    int width = head_count * dimension;
    int block_count = width / 32;
    int task = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int row = task / block_count;
    int block = task - row * block_count;
    if (row >= rows) return;
    int column0 = block * 32 + lane;
    int column1 = column0 + 16;
    int head0 = column0 / dimension;
    int d0 = column0 - head0 * dimension;
    int head1 = column1 / dimension;
    int d1 = column1 - head1 * dimension;
    float value0 = heads[((size_t)head0 * rows + row) * dimension + d0];
    float value1 = heads[((size_t)head1 * rows + row) * dimension + d1];
    float maximum = sub_group_reduce_max(fmax(fabs(value0), fabs(value1)));
    volatile float d = maximum / 127.0f;
    volatile float inverse = d == 0.0f ? 0.0f : 1.0f / d;
    volatile float scaled0 = value0 * inverse;
    volatile float scaled1 = value1 * inverse;
    int q0 = convert_int(round(scaled0));
    int q1 = convert_int(round(scaled1));
    size_t base = ((size_t)row * block_count + block) * 32;
    quantized[base + lane] = (char)q0;
    quantized[base + lane + 16] = (char)q1;
    int sum = sub_group_reduce_add(q0 + q1);
    if (lane == 0) {
        size_t index = (size_t)row * block_count + block;
        scale[index] = (half)d;
        sigma[index] = (short)sum;
    }
}

__kernel void prefill_rms_residual(__global const float *input,
                                   __global const float *weight,
                                   __global const float *residual,
                                   __global float *output,
                                   int rows,
                                   int width,
                                   float epsilon) {
    __local float partial[128];
    int row = get_group_id(0);
    int lid = get_local_id(0);
    if (row >= rows) return;
    float sum = 0.0f;
    for (int column = lid; column < width; column += 128) {
        float value = input[(size_t)row * width + column];
        sum += value * value;
    }
    partial[lid] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lid < stride) partial[lid] += partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float scale = rsqrt(partial[0] / (float)width + epsilon);
    for (int column = lid; column < width; column += 128) {
        size_t index = (size_t)row * width + column;
        output[index] = input[index] * scale * weight[column] + residual[index];
    }
}

__kernel void prefill_ffn_finish(__global const float *dense,
                                 __global const float *moe,
                                 __global const float *dense_weight,
                                 __global const float *moe_weight,
                                 __global const float *combine_weight,
                                 __global const float *residual,
                                 __global const float *layer_scale,
                                 __global float *output,
                                 int rows,
                                 int width,
                                 float epsilon) {
    __local float dense_partial[128];
    __local float moe_partial[128];
    int row = get_group_id(0);
    int lid = get_local_id(0);
    if (row >= rows) return;
    float dense_sum = 0.0f;
    float moe_sum = 0.0f;
    for (int column = lid; column < width; column += 128) {
        size_t index = (size_t)row * width + column;
        float dense_value = dense[index];
        float moe_value = moe[index];
        dense_sum += dense_value * dense_value;
        moe_sum += moe_value * moe_value;
    }
    dense_partial[lid] = dense_sum;
    moe_partial[lid] = moe_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lid < stride) {
            dense_partial[lid] += dense_partial[lid + stride];
            moe_partial[lid] += moe_partial[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float dense_scale = rsqrt(dense_partial[0] / (float)width + epsilon);
    float moe_scale = rsqrt(moe_partial[0] / (float)width + epsilon);
    float combine_sum = 0.0f;
    for (int column = lid; column < width; column += 128) {
        size_t index = (size_t)row * width + column;
        float value = dense[index] * dense_scale * dense_weight[column]
                      + moe[index] * moe_scale * moe_weight[column];
        output[index] = value;
        combine_sum += value * value;
    }
    dense_partial[lid] = combine_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (lid < stride) dense_partial[lid] += dense_partial[lid + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float combine_scale = rsqrt(dense_partial[0] / (float)width + epsilon);
    for (int column = lid; column < width; column += 128) {
        size_t index = (size_t)row * width + column;
        output[index] = (output[index] * combine_scale * combine_weight[column]
                         + residual[index]) * layer_scale[0];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_q4q8_grouped_gather(__global const uchar *wq,
                                          __global const half *wd,
                                          __global const char *aq,
                                          __global const half *ad,
                                          __global const short *as,
                                          __global float *out,
                                          __global const int *expert_count,
                                          __global const int *token_offset,
                                          __global const int *tile_expert,
                                          __global const int *tile_m0,
                                          __global const int *route_token,
                                          __global const int *packed_route,
                                          int n_count,
                                          int blocks) {
    __local char la[TM * KB];
    __local uchar lw[TN * (KB / 2)];
    __local half lad[TM * 2];
    __local short las[TM * 2];
    __local half lwd[TN * 2];
    int lid = get_local_id(0);
    int tile = get_group_id(0);
    int expert = tile_expert[tile];
    if (expert < 0) return;
    int local_m0 = tile_m0[tile];
    int packed_m0 = token_offset[expert] + local_m0;
    int count = expert_count[expert];
    int wm = lid >> 4;
    int wn = lid & 15;
    int n0 = get_group_id(1) * TN;
    float acc[4][2] = {{0.0f}};

    for (int kb = 0; kb < blocks * 32; kb += KB) {
        for (int x = lid; x < TM * KB; x += 128) {
            int lm = x / KB;
            int lk = x - lm * KB;
            int valid = local_m0 + lm < count;
            int packed = packed_m0 + lm;
            int token = valid ? route_token[packed_route[packed]] : 0;
            la[x] = valid ? aq[(size_t)token * blocks * 32 + kb + lk] : 0;
        }
        #pragma unroll
        for (int seg = 0; seg < 8; seg++) {
            int lng = seg >> 1;
            int lb = seg & 1;
            int chunk = lid >> 5;
            int rem = lid & 31;
            int row8 = rem >> 2;
            int j = rem & 3;
            int ln = lng * 8 + row8;
            lw[ln * 32 + lb * 16 + chunk * 4 + j] =
                wq[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                     * blocks + kb / 32 + lb) * 128 + lid];
        }
        if (lid < TM * 2) {
            int lm = lid >> 1;
            int lb = lid & 1;
            int valid = local_m0 + lm < count;
            int packed = packed_m0 + lm;
            int token = valid ? route_token[packed_route[packed]] : 0;
            lad[lid] = valid
                       ? ad[(size_t)token * blocks + kb / 32 + lb] : (half)0;
            las[lid] = valid
                       ? as[(size_t)token * blocks + kb / 32 + lb] : (short)0;
        }
        if (lid < TN * 2) {
            int seg = lid >> 3;
            int row8 = lid & 7;
            int lng = seg >> 1;
            int lb = seg & 1;
            lwd[(lng * 8 + row8) * 2 + lb] =
                wd[(((size_t)expert * (n_count >> 3) + (n0 >> 3) + lng)
                    * blocks + kb / 32 + lb) * 8 + row8];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int lb = 0; lb < 2; lb++) {
            #pragma unroll
            for (int im = 0; im < 4; im++) {
                int lm = wm + im * 8;
                int correction = 8 * (int)las[lm * 2 + lb];
                float da = (float)lad[lm * 2 + lb];
                #pragma unroll
                for (int in = 0; in < 2; in++) {
                    int ln = wn + in * 16;
                    int integer = 0;
                    #pragma unroll
                    for (int c = 0; c < 4; c++) {
                        uchar4 packed = vload4(0, lw + ln * 32 + lb * 16 + c * 4);
                        char4 lo = vload4(0, la + lm * KB + lb * 32 + c * 4);
                        char4 hi = vload4(0, la + lm * KB + lb * 32 + 16 + c * 4);
                        integer += dot(packed & (uchar4)(15), lo)
                                   + dot(packed >> (uchar4)(4), hi);
                    }
                    acc[im][in] += (float)(integer - correction) * da
                                   * (float)lwd[ln * 2 + lb];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int im = 0; im < 4; im++) {
        int lm = wm + im * 8;
        if (local_m0 + lm >= count) continue;
        int gm = packed_m0 + lm;
        #pragma unroll
        for (int in = 0; in < 2; in++)
            out[(size_t)gm * n_count + n0 + wn + in * 16] = acc[im][in];
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_attn_qk(__global const float *q,
                              __global const half *k,
                              __global float *scores,
                              int m_count,
                              int n_count,
                              int dimension,
                              int heads,
                              int kv_heads,
                              int query_offset,
                              int window) {
    int subgroup = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int key = subgroup % n_count;
    int query_head = subgroup / n_count;
    int query = query_head % m_count;
    int head = query_head / m_count;
    if (head >= heads) return;
    int position = query_offset + query;
    int first = window > 0 ? max(0, position - window + 1) : 0;
    if (key < first || key > position) {
        if (lane == 0) scores[(size_t)query_head * n_count + key] = -INFINITY;
        return;
    }
    int kv_head = head * kv_heads / heads;
    float sum = 0.0f;
    for (int d = lane; d < dimension; d += 16)
        sum += q[((size_t)head * m_count + query) * dimension + d]
               * (float)k[((size_t)kv_head * n_count + key) * dimension + d];
    sum = sub_group_reduce_add(sum);
    if (lane == 0) scores[(size_t)query_head * n_count + key] = sum;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_attn_softmax(__global float *scores,
                                   int rows,
                                   int n_count) {
    int row = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    if (row >= rows) return;
    __global float *values = scores + (size_t)row * n_count;
    float maximum = -INFINITY;
    for (int key = lane; key < n_count; key += 16)
        maximum = fmax(maximum, values[key]);
    maximum = sub_group_reduce_max(maximum);
    float sum = 0.0f;
    for (int key = lane; key < n_count; key += 16) {
        float value = values[key] == -INFINITY ? 0.0f
                      : exp(values[key] - maximum);
        values[key] = value;
        sum += value;
    }
    sum = sub_group_reduce_add(sum);
    for (int key = lane; key < n_count; key += 16)
        values[key] /= sum;
}

__kernel void prefill_attn_pv(__global const float *scores,
                              __global const half *v,
                              __global float *out,
                              int m_count,
                              int n_count,
                              int dimension,
                              int heads,
                              int kv_heads) {
    __local float local_scores[128];
    int lid = get_local_id(0);
    size_t query_head = get_group_id(0);
    if (query_head >= (size_t)heads * m_count) return;
    int head = query_head / m_count;
    int kv_head = head * kv_heads / heads;
    float acc[4] = {0.0f};
    for (int key0 = 0; key0 < n_count; key0 += 128) {
        int key = key0 + lid;
        local_scores[lid] = key < n_count
                            ? scores[query_head * n_count + key] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        int owned = 0;
        for (int d = lid; d < dimension; d += 128) {
            float sum = acc[owned];
            int count = min(128, n_count - key0);
            for (int local_key = 0; local_key < count; local_key++)
                sum += local_scores[local_key]
                       * (float)v[((size_t)kv_head * n_count + key0 + local_key)
                                  * dimension + d];
            acc[owned++] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int owned = 0;
    for (int d = lid; d < dimension; d += 128)
        out[query_head * dimension + d] = acc[owned++];
}

__kernel void prefill_attn_pv4(__global const float *scores,
                               __global const half *v,
                               __global float *out,
                               int m_count,
                               int n_count,
                               int dimension,
                               int heads,
                               int kv_heads) {
    __local float local_scores[128];
    int lid = get_local_id(0);
    size_t query_head = get_group_id(0);
    if (query_head >= (size_t)heads * m_count) return;
    int head = query_head / m_count;
    int kv_head = head * kv_heads / heads;
    float acc[4][4] = {{0.0f}};
    for (int key0 = 0; key0 < n_count; key0 += 128) {
        int key = key0 + lid;
        local_scores[lid] = key < n_count
                            ? scores[query_head * n_count + key] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        int owned = 0;
        for (int d = lid; d < dimension; d += 128) {
            float sum0 = 0.0f;
            float sum1 = 0.0f;
            float sum2 = 0.0f;
            float sum3 = 0.0f;
            int count = min(128, n_count - key0);
            for (int local_key = 0; local_key < count; local_key += 4) {
                sum0 += local_scores[local_key]
                        * (float)v[((size_t)kv_head * n_count + key0 + local_key)
                                   * dimension + d];
                if (local_key + 1 < count)
                    sum1 += local_scores[local_key + 1]
                            * (float)v[((size_t)kv_head * n_count + key0
                                        + local_key + 1) * dimension + d];
                if (local_key + 2 < count)
                    sum2 += local_scores[local_key + 2]
                            * (float)v[((size_t)kv_head * n_count + key0
                                        + local_key + 2) * dimension + d];
                if (local_key + 3 < count)
                    sum3 += local_scores[local_key + 3]
                            * (float)v[((size_t)kv_head * n_count + key0
                                        + local_key + 3) * dimension + d];
            }
            int slot = (key0 >> 7) & 3;
            acc[owned++][slot] += (sum0 + sum1) + (sum2 + sum3);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int owned = 0;
    for (int d = lid; d < dimension; d += 128) {
        out[query_head * dimension + d] =
            (acc[owned][0] + acc[owned][1])
            + (acc[owned][2] + acc[owned][3]);
        owned++;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_attn_online(__global const float *q,
                                  __global const half *k,
                                  __global const half *v,
                                  __global float *out,
                                  int m_count,
                                  int n_count,
                                  int dimension,
                                  int heads,
                                  int kv_heads,
                                  int query_offset,
                                  int window) {
    int query_head = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query = query_head % m_count;
    int head = query_head / m_count;
    if (head >= heads) return;
    int kv_head = head * kv_heads / heads;
    int position = query_offset + query;
    int first = window > 0 ? max(0, position - window + 1) : 0;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc[32] = {0.0f};
    for (int key = first; key <= position; key++) {
        float score = 0.0f;
        for (int d = lane; d < dimension; d += 16)
            score += q[((size_t)head * m_count + query) * dimension + d]
                     * (float)k[((size_t)kv_head * n_count + key) * dimension + d];
        score = sub_group_reduce_add(score);
        float next_maximum = fmax(maximum, score);
        float alpha = maximum == -INFINITY ? 0.0f : exp(maximum - next_maximum);
        float beta = exp(score - next_maximum);
        denominator = denominator * alpha + beta;
        int owned = 0;
        for (int d = lane; d < dimension; d += 16) {
            acc[owned] = acc[owned] * alpha
                         + beta * (float)v[((size_t)kv_head * n_count + key)
                                           * dimension + d];
            owned++;
        }
        maximum = next_maximum;
    }
    int owned = 0;
    for (int d = lane; d < dimension; d += 16)
        out[((size_t)head * m_count + query) * dimension + d] =
            acc[owned++] / denominator;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_attn_online_b4(__global const float *q,
                                     __global const half *k,
                                     __global const half *v,
                                     __global float *out,
                                     int m_count,
                                     int n_count,
                                     int dimension,
                                     int heads,
                                     int kv_heads,
                                     int query_offset,
                                     int window) {
    int query_head = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query = query_head % m_count;
    int head = query_head / m_count;
    if (head >= heads) return;
    int kv_head = head * kv_heads / heads;
    int position = query_offset + query;
    int first = window > 0 ? max(0, position - window + 1) : 0;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc[32] = {0.0f};
    for (int key0 = first; key0 <= position; key0 += 4) {
        int keys = min(4, position - key0 + 1);
        float score[4];
        float block_maximum = -INFINITY;
        for (int local_key = 0; local_key < keys; local_key++) {
            float value = 0.0f;
            int key = key0 + local_key;
            for (int d = lane; d < dimension; d += 16)
                value += q[((size_t)head * m_count + query) * dimension + d]
                         * (float)k[((size_t)kv_head * n_count + key)
                                    * dimension + d];
            score[local_key] = sub_group_reduce_add(value);
            block_maximum = fmax(block_maximum, score[local_key]);
        }
        float next_maximum = fmax(maximum, block_maximum);
        float alpha = maximum == -INFINITY ? 0.0f
                      : exp(maximum - next_maximum);
        float beta[4];
        float block_denominator = 0.0f;
        for (int local_key = 0; local_key < keys; local_key++) {
            beta[local_key] = exp(score[local_key] - next_maximum);
            block_denominator += beta[local_key];
        }
        denominator = denominator * alpha + block_denominator;
        int owned = 0;
        for (int d = lane; d < dimension; d += 16) {
            float value = acc[owned] * alpha;
            for (int local_key = 0; local_key < keys; local_key++)
                value += beta[local_key]
                         * (float)v[((size_t)kv_head * n_count
                                     + key0 + local_key) * dimension + d];
            acc[owned++] = value;
        }
        maximum = next_maximum;
    }
    int owned = 0;
    for (int d = lane; d < dimension; d += 16)
        out[((size_t)head * m_count + query) * dimension + d] =
            acc[owned++] / denominator;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_attn_online_b8(__global const float *q,
                                     __global const half *k,
                                     __global const half *v,
                                     __global float *out,
                                     int m_count,
                                     int n_count,
                                     int dimension,
                                     int heads,
                                     int kv_heads,
                                     int query_offset,
                                     int window) {
    int query_head = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query = query_head % m_count;
    int head = query_head / m_count;
    if (head >= heads) return;
    int kv_head = head * kv_heads / heads;
    int position = query_offset + query;
    int first = window > 0 ? max(0, position - window + 1) : 0;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc[32] = {0.0f};
    for (int key0 = first; key0 <= position; key0 += 8) {
        int keys = min(8, position - key0 + 1);
        float score[8];
        float block_maximum = -INFINITY;
        for (int local_key = 0; local_key < keys; local_key++) {
            float value = 0.0f;
            int key = key0 + local_key;
            for (int d = lane; d < dimension; d += 16)
                value += q[((size_t)head * m_count + query) * dimension + d]
                         * (float)k[((size_t)kv_head * n_count + key)
                                    * dimension + d];
            score[local_key] = sub_group_reduce_add(value);
            block_maximum = fmax(block_maximum, score[local_key]);
        }
        float next_maximum = fmax(maximum, block_maximum);
        float alpha = maximum == -INFINITY ? 0.0f
                      : exp(maximum - next_maximum);
        float beta[8];
        float block_denominator = 0.0f;
        for (int local_key = 0; local_key < keys; local_key++) {
            beta[local_key] = exp(score[local_key] - next_maximum);
            block_denominator += beta[local_key];
        }
        denominator = denominator * alpha + block_denominator;
        int owned = 0;
        for (int d = lane; d < dimension; d += 16) {
            float value = acc[owned] * alpha;
            for (int local_key = 0; local_key < keys; local_key++)
                value += beta[local_key]
                         * (float)v[((size_t)kv_head * n_count
                                     + key0 + local_key) * dimension + d];
            acc[owned++] = value;
        }
        maximum = next_maximum;
    }
    int owned = 0;
    for (int d = lane; d < dimension; d += 16)
        out[((size_t)head * m_count + query) * dimension + d] =
            acc[owned++] / denominator;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_attn_online_b8_stage(__global const float *q,
                                          __global const half *k,
                                          __global const half *v,
                                          __global const half *batch_k,
                                          __global const half *batch_v,
                                          __global float *out,
                                          int m_count,
                                          int n_count,
                                          int dimension,
                                          int heads,
                                          int kv_heads,
                                          int query_offset,
                                          int window,
                                          int capacity) {
    int query_head = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query = query_head % m_count;
    int head = query_head / m_count;
    if (head >= heads) return;
    int kv_head = head * kv_heads / heads;
    int position = query_offset + query;
    int first = max(0, position - window + 1);
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc[32] = {0.0f};
    for (int key0 = first; key0 <= position; key0 += 8) {
        int keys = min(8, position - key0 + 1);
        float score[8];
        float block_maximum = -INFINITY;
        for (int local_key = 0; local_key < keys; local_key++) {
            float value = 0.0f;
            int logical_key = key0 + local_key;
            int stage_base = n_count - capacity;
            for (int d = lane; d < dimension; d += 16) {
                half key_value = k[((size_t)kv_head * capacity
                                    + logical_key - stage_base) * dimension + d];
                value += q[((size_t)head * m_count + query) * dimension + d]
                         * (float)key_value;
            }
            score[local_key] = sub_group_reduce_add(value);
            block_maximum = fmax(block_maximum, score[local_key]);
        }
        float next_maximum = fmax(maximum, block_maximum);
        float alpha = maximum == -INFINITY ? 0.0f
                      : exp(maximum - next_maximum);
        float beta[8];
        float block_denominator = 0.0f;
        for (int local_key = 0; local_key < keys; local_key++) {
            beta[local_key] = exp(score[local_key] - next_maximum);
            block_denominator += beta[local_key];
        }
        denominator = denominator * alpha + block_denominator;
        int owned = 0;
        for (int d = lane; d < dimension; d += 16) {
            float value = acc[owned] * alpha;
            for (int local_key = 0; local_key < keys; local_key++) {
                int logical_key = key0 + local_key;
                int stage_base = n_count - capacity;
                half value_source = v[((size_t)kv_head * capacity
                                        + logical_key - stage_base) * dimension + d];
                value += beta[local_key]
                         * (float)value_source;
            }
            acc[owned++] = value;
        }
        maximum = next_maximum;
    }
    int owned = 0;
    for (int d = lane; d < dimension; d += 16)
        out[((size_t)head * m_count + query) * dimension + d] =
            acc[owned++] / denominator;
    (void)n_count;
    (void)batch_k;
    (void)batch_v;
}

__kernel void prefill_swa_stage(__global const half *ring_k,
                                __global const half *ring_v,
                                __global const half *batch_k,
                                __global const half *batch_v,
                                __global half *stage_k,
                                __global half *stage_v,
                                int m_count,
                                int dimension,
                                int kv_heads,
                                int batch_start,
                                int stage_base,
                                int stage_count,
                                int capacity) {
    size_t index = get_global_id(0);
    size_t elements = (size_t)kv_heads * stage_count * dimension;
    if (index >= elements) return;
    int d = index % dimension;
    size_t row = index / dimension;
    int local_key = row % stage_count;
    int kv_head = row / stage_count;
    int key = stage_base + local_key;
    size_t source = key >= batch_start
        ? ((size_t)kv_head * m_count + key - batch_start) * dimension + d
        : ((size_t)kv_head * capacity + (key & (capacity - 1))) * dimension + d;
    stage_k[index] = key >= batch_start ? batch_k[source] : ring_k[source];
    stage_v[index] = key >= batch_start ? batch_v[source] : ring_v[source];
}

__kernel void prefill_swa_commit(__global half *ring_k,
                                 __global half *ring_v,
                                 __global const half *batch_k,
                                 __global const half *batch_v,
                                 int m_count,
                                 int dimension,
                                 int kv_heads,
                                 int batch_start,
                                 int capacity) {
    size_t index = get_global_id(0);
    size_t elements = (size_t)kv_heads * m_count * dimension;
    if (index >= elements) return;
    int d = index % dimension;
    size_t row = index / dimension;
    int query = row % m_count;
    int kv_head = row / m_count;
    size_t target = ((size_t)kv_head * capacity
                     + ((batch_start + query) & (capacity - 1))) * dimension + d;
    ring_k[target] = batch_k[index];
    ring_v[target] = batch_v[index];
}

__kernel void prefill_attn_long_init(__global half *values, ulong elements) {
    size_t index = get_global_id(0);
    if (index >= elements) return;
    uint bits = (uint)index * 1664525u + 1013904223u;
    values[index] = (half)((float)((int)(bits & 1023u) - 512) / 16384.0f);
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_attn_partial_b8(__global const float *q,
                                     __global const half *k,
                                     __global const half *v,
                                     __global float *partial,
                                     int m_count,
                                     int n_count,
                                     int dimension,
                                     int heads,
                                     int kv_heads,
                                     int query_offset,
                                     int chunk_size,
                                     int max_chunks,
                                     int partial_stride,
                                     int chunk_major) {
    int linear = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query_heads = heads * m_count;
    int chunk = chunk_major ? linear / query_heads : linear % max_chunks;
    int query_head = chunk_major ? linear - chunk * query_heads
                                 : linear / max_chunks;
    if (chunk >= max_chunks) return;
    int query = query_head % m_count;
    int head = query_head / m_count;
    if (head >= heads) return;
    int position = query_offset + query;
    int first = chunk * chunk_size;
    int last = min(position + 1, first + chunk_size);
    if (first >= last) return;
    int kv_head = head * kv_heads / heads;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc[32] = {0.0f};
    for (int key0 = first; key0 < last; key0 += 8) {
        int keys = min(8, last - key0);
        float score[8];
        float block_maximum = -INFINITY;
        for (int local_key = 0; local_key < keys; local_key++) {
            float value = 0.0f;
            int key = key0 + local_key;
            for (int d = lane; d < dimension; d += 16)
                value += q[((size_t)head * m_count + query) * dimension + d]
                         * (float)k[((size_t)kv_head * n_count + key)
                                    * dimension + d];
            score[local_key] = sub_group_reduce_add(value);
            block_maximum = fmax(block_maximum, score[local_key]);
        }
        float next_maximum = fmax(maximum, block_maximum);
        float alpha = maximum == -INFINITY ? 0.0f
                      : exp(maximum - next_maximum);
        float beta[8];
        float block_denominator = 0.0f;
        for (int local_key = 0; local_key < keys; local_key++) {
            beta[local_key] = exp(score[local_key] - next_maximum);
            block_denominator += beta[local_key];
        }
        denominator = denominator * alpha + block_denominator;
        int owned = 0;
        for (int d = lane; d < dimension; d += 16) {
            float value = acc[owned] * alpha;
            for (int local_key = 0; local_key < keys; local_key++)
                value += beta[local_key]
                         * (float)v[((size_t)kv_head * n_count
                                     + key0 + local_key) * dimension + d];
            acc[owned++] = value;
        }
        maximum = next_maximum;
    }
    __global float *target = partial
        + ((size_t)query_head * max_chunks + chunk) * partial_stride;
    if (lane == 0) {
        target[0] = maximum;
        target[1] = denominator;
    }
    int owned = 0;
    for (int d = lane; d < dimension; d += 16)
        target[16 + d] = acc[owned++];
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_attn_partial_merge(__global const float *partial,
                                        __global float *out,
                                        int m_count,
                                        int dimension,
                                        int heads,
                                        int query_offset,
                                        int chunk_size,
                                        int max_chunks,
                                        int partial_stride) {
    int query_head = get_group_id(0) * 8 + get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query = query_head % m_count;
    int head = query_head / m_count;
    if (head >= heads) return;
    int chunks = (query_offset + query + chunk_size) / chunk_size;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc[32] = {0.0f};
    for (int chunk = 0; chunk < chunks; chunk++) {
        __global const float *source = partial
            + ((size_t)query_head * max_chunks + chunk) * partial_stride;
        float chunk_maximum = source[0];
        float chunk_denominator = source[1];
        float next_maximum = fmax(maximum, chunk_maximum);
        float alpha = maximum == -INFINITY ? 0.0f
                      : exp(maximum - next_maximum);
        float beta = exp(chunk_maximum - next_maximum);
        denominator = denominator * alpha + chunk_denominator * beta;
        int owned = 0;
        for (int d = lane; d < dimension; d += 16) {
            acc[owned] = acc[owned] * alpha + source[16 + d] * beta;
            owned++;
        }
        maximum = next_maximum;
    }
    int owned = 0;
    for (int d = lane; d < dimension; d += 16)
        out[(size_t)query_head * dimension + d] = acc[owned++] / denominator;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void prefill_attn_gqa8(__global const float *q,
                                __global const half *k,
                                __global const half *v,
                                __global float *out,
                                int m_count,
                                int n_count,
                                int dimension,
                                int heads,
                                int kv_heads,
                                int query_offset,
                                int window) {
    __local float local_q[8 * 512];
    __local half local_k[16 * 512];
    __local half local_v[16 * 512];
    int lid = get_local_id(0);
    int subgroup = get_sub_group_id();
    int lane = get_sub_group_local_id();
    int query_kv = get_group_id(0);
    int query = query_kv % m_count;
    int kv_head = query_kv / m_count;
    if (kv_head >= kv_heads || dimension != 512 || heads != kv_heads * 8)
        return;
    int head = kv_head * 8 + subgroup;
    for (int x = lid; x < 8 * dimension; x += 128) {
        int local_head = x / dimension;
        int d = x - local_head * dimension;
        local_q[x] = q[((size_t)(kv_head * 8 + local_head) * m_count + query)
                       * dimension + d];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int position = query_offset + query;
    int first = window > 0 ? max(0, position - window + 1) : 0;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float acc[32] = {0.0f};
    for (int key0 = first; key0 <= position; key0 += 16) {
        int keys = min(16, position - key0 + 1);
        for (int x = lid; x < keys * dimension; x += 128) {
            int local_key = x / dimension;
            int d = x - local_key * dimension;
            size_t source = ((size_t)kv_head * n_count + key0 + local_key)
                            * dimension + d;
            local_k[x] = k[source];
            local_v[x] = v[source];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int local_key = 0; local_key < keys; local_key++) {
            float score = 0.0f;
            for (int d = lane; d < dimension; d += 16)
                score += local_q[subgroup * dimension + d]
                         * (float)local_k[local_key * dimension + d];
            score = sub_group_reduce_add(score);
            float next_maximum = fmax(maximum, score);
            float alpha = maximum == -INFINITY ? 0.0f
                          : exp(maximum - next_maximum);
            float beta = exp(score - next_maximum);
            denominator = denominator * alpha + beta;
            int owned = 0;
            for (int d = lane; d < dimension; d += 16) {
                acc[owned] = acc[owned] * alpha
                             + beta * (float)local_v[local_key * dimension + d];
                owned++;
            }
            maximum = next_maximum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int owned = 0;
    for (int d = lane; d < dimension; d += 16)
        out[((size_t)head * m_count + query) * dimension + d] =
            acc[owned++] / denominator;
}

#ifdef BENCH_CACHE_PROBE
__kernel void prefill_cache_probe(__global uint *value) {
    if (get_global_id(0) == 0) value[0] = BENCH_CACHE_PROBE;
}
#endif
