#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static long file_size(FILE *f) {
    if (fseek(f, 0, SEEK_END) != 0) return -1;
    long n = ftell(f);
    if (fseek(f, 0, SEEK_SET) != 0) return -1;
    return n;
}

static float *load(const char *path, long *size_out) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path);
        return NULL;
    }
    long n = file_size(f);
    if (n < 0) {
        fprintf(stderr, "error: cannot size %s\n", path);
        fclose(f);
        return NULL;
    }
    float *buf = malloc((size_t)n);
    if (!buf) {
        fprintf(stderr, "error: out of memory for %s\n", path);
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "error: short read on %s\n", path);
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size_out = n;
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <a.bin> <b.bin> <rel_rms_threshold>\n", argv[0]);
        return 2;
    }

    char *end;
    double thr = strtod(argv[3], &end);
    if (end == argv[3] || *end != '\0' || !(thr >= 0.0)) {
        fprintf(stderr, "error: bad threshold '%s'\n", argv[3]);
        return 2;
    }

    long na = 0, nb = 0;
    float *a = load(argv[1], &na);
    if (!a) return 2;
    float *b = load(argv[2], &nb);
    if (!b) {
        free(a);
        return 2;
    }

    if (na != nb) {
        fprintf(stderr, "error: size mismatch: %s is %ld bytes, %s is %ld bytes\n",
                argv[1], na, argv[2], nb);
        free(a);
        free(b);
        return 2;
    }
    if (na == 0 || na % 4 != 0) {
        fprintf(stderr, "error: size %ld is not a positive multiple of 4\n", na);
        free(a);
        free(b);
        return 2;
    }

    const long n = na / 4;

    double sum_d2 = 0.0;
    double sum_b2 = 0.0;
    double max_abs = 0.0;
    long max_idx = 0;
    long top_a = 0, top_b = 0;
    double best_a = (double)a[0], best_b = (double)b[0];

    for (long i = 0; i < n; i++) {
        double av = (double)a[i];
        double bv = (double)b[i];
        double d = av - bv;
        sum_d2 += d * d;
        sum_b2 += bv * bv;
        double ad = fabs(d);
        if (ad > max_abs) {
            max_abs = ad;
            max_idx = i;
        }
        if (av > best_a) {
            best_a = av;
            top_a = i;
        }
        if (bv > best_b) {
            best_b = bv;
            top_b = i;
        }
    }

    double rms_d = sqrt(sum_d2 / (double)n);
    double rms_b = sqrt(sum_b2 / (double)n);
    double rel = rms_b > 0.0 ? rms_d / rms_b : (rms_d == 0.0 ? 0.0 : INFINITY);
    int top_match = (top_a == top_b);

    printf("a          %s\n", argv[1]);
    printf("b          %s\n", argv[2]);
    printf("n          %ld\n", n);
    printf("rms_a_b    %.10e\n", rms_d);
    printf("rms_b      %.10e\n", rms_b);
    printf("rel_rms    %.10e\n", rel);
    printf("max_abs    %.10e at index %ld (a=%.8f b=%.8f)\n",
           max_abs, max_idx, (double)a[max_idx], (double)b[max_idx]);
    printf("top1_a     %ld (%.8f)\n", top_a, best_a);
    printf("top1_b     %ld (%.8f)\n", top_b, best_b);
    printf("top1_match %s\n", top_match ? "yes" : "no");
    printf("threshold  %.10e\n", thr);

    free(a);
    free(b);

    if (!(rel <= thr)) {
        printf("result     FAIL\n");
        return 1;
    }
    printf("result     PASS\n");
    return 0;
}
