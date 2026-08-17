#include "format.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int hex_equal(const uint8_t *bytes, const char *hex, size_t n) {
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < n; i++) {
        if (digits[bytes[i] >> 4] != hex[2 * i] ||
            digits[bytes[i] & 15] != hex[2 * i + 1]) return 0;
    }
    return 1;
}

int main(void) {
    int ok = 1;
    uint8_t bytes[32];
    uint8_t le[8];

    format_put_u16le(le, UINT16_C(0x1234));
    ok &= le[0] == 0x34 && le[1] == 0x12 &&
          format_get_u16le(le) == UINT16_C(0x1234);
    format_put_u32le(le, UINT32_C(0x89abcdef));
    ok &= format_get_u32le(le) == UINT32_C(0x89abcdef);
    format_put_u64le(le, UINT64_C(0x0123456789abcdef));
    ok &= format_get_u64le(le) == UINT64_C(0x0123456789abcdef);

    uint64_t value;
    ok &= format_add_u64(4, 7, &value) && value == 11;
    ok &= !format_add_u64(UINT64_MAX, 1, &value);
    ok &= format_mul_u64(7, 9, &value) && value == 63;
    ok &= !format_mul_u64(UINT64_MAX, 2, &value);
    ok &= format_align_u64(65, 64, &value) && value == 128;
    ok &= !format_align_u64(UINT64_MAX, 64, &value);

    format_sha256_bytes("", 0, bytes);
    ok &= hex_equal(bytes,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        32);
    format_sha256_bytes("abc", 3, bytes);
    ok &= hex_equal(bytes,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        32);
    format_sha256 hash;
    format_sha256_init(&hash);
    format_sha256_update(&hash, "a", 1);
    format_sha256_update(&hash, "b", 1);
    format_sha256_update(&hash, "c", 1);
    format_sha256_final(&hash, bytes);
    ok &= hex_equal(bytes,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        32);
    ok &= format_crc64(0, "123456789", 9) ==
          UINT64_C(0x6c40df5f0b497347);
    uint64_t split_crc = format_crc64(0, "1234", 4);
    split_crc = format_crc64(split_crc, "56789", 5);
    ok &= split_crc == UINT64_C(0x6c40df5f0b497347);

    FILE *file = tmpfile();
    ok &= file != NULL;
    if (file) {
        const char payload[] = "seekable format stream";
        char loaded[sizeof payload];
        uint64_t size = 0;
        ok &= format_write(file, payload, sizeof payload);
        ok &= format_file_size(file, &size) && size == sizeof payload;
        ok &= format_seek(file, 0) && format_read(file, loaded, sizeof loaded);
        ok &= memcmp(payload, loaded, sizeof payload) == 0;
        fclose(file);
    }

    printf("format: primitives %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
