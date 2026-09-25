#!/usr/bin/env bash
set -euo pipefail
umask 077
ulimit -c 0

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ${1:-} == --help ]]; then
    printf 'Usage: XENOLITH_MODEL=GGUF XENOLITH_LLAMA_SOURCE=CHECKOUT XENOLITH_BENCH_OUTPUT=FRESH_DIR bash bench/compare_pp_tg.sh\n'
    printf 'Build the two harness variants and run pp512, pp2048, and tg128 on the target machine.\n'
    printf 'To check published data without building or measuring, run: python3 -B bench/compare_pp_tg_verify.py\n'
    exit 0
fi
[[ $# == 0 ]] || { printf 'Unknown argument; use --help\n' >&2; exit 2; }
: "${XENOLITH_MODEL:?set XENOLITH_MODEL to the target GGUF}"
: "${XENOLITH_LLAMA_SOURCE:?set XENOLITH_LLAMA_SOURCE to a llama.cpp checkout}"
: "${XENOLITH_BENCH_OUTPUT:?set XENOLITH_BENCH_OUTPUT outside the source tree}"
llama_commit=434ddbbc0e30522e897670681e503b797c12b7c1
model_hash=a7c5bc715f5ff8e99a3e8901ce7d2b42b402c669bf24f7c5250747633d0f5891
token_hash=30fb9a750a9318718a373b802608d77a41c3c734928d393ddad93d00ae84952b

model=$(realpath -- "$XENOLITH_MODEL")
llama_source=$(realpath -- "$XENOLITH_LLAMA_SOURCE")
mkdir -p -- "$XENOLITH_BENCH_OUTPUT"
output=$(realpath -- "$XENOLITH_BENCH_OUTPUT")
case "$output/" in
    "$repo/"*) printf 'Keep private run logs outside the source tree\n' >&2; exit 2 ;;
    "$llama_source/"*) printf 'Keep private run logs outside the llama.cpp source tree\n' >&2; exit 2 ;;
esac
[[ ! -e "$output/runs" && ! -e "$output/llama-build" ]] || { printf 'Use a fresh output directory\n' >&2; exit 2; }
[[ -f "$model" ]] || { printf 'Model file not found\n' >&2; exit 2; }
[[ $(git -C "$llama_source" rev-parse HEAD) == "$llama_commit" ]] || { printf 'Wrong llama.cpp commit\n' >&2; exit 2; }
git -C "$llama_source" diff --quiet
git -C "$llama_source" diff --cached --quiet
[[ $(sha256sum "$model" | cut -d ' ' -f 1) == "$model_hash" ]] || { printf 'Wrong GGUF hash\n' >&2; exit 2; }

make -C "$repo" bench/compare_pp_tg_xe bench/compare_pp_tg_tokens bench/bench_guard
"$repo/bench/compare_pp_tg_tokens" > "$output/random-seed1.ids"
[[ $(sha256sum "$output/random-seed1.ids" | cut -d ' ' -f 1) == "$token_hash" ]] || { printf 'Token sequence differs\n' >&2; exit 2; }

build="$output/llama-build"
cmake -S "$llama_source" -B "$build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_VULKAN=ON -DGGML_BLAS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF
cmake --build "$build" --target llama-bench -j4
libdir="$build/bin"
gcc -O3 -march=native -std=c11 -Wall -Wextra -pthread -DLLAMA -I"$llama_source/include" -I"$llama_source/ggml/include" "$repo/bench/compare_pp_tg.c" -L"$libdir" -Wl,-rpath,"$libdir" -lllama -lggml -lggml-base -lggml-cpu -lm -o "$output/compare_pp_tg_llama"
export LD_LIBRARY_PATH="$libdir"
unset GGML_BACKEND_PATH GGML_VK_DISABLE_F16 GGML_VK_DISABLE_FUSION GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM
unset LOAD CTX REPACK SWA_FULL KV8 CPU_MOE POOLS POLL GEN_E BATCH_ALL BATCH_E EXTRAS LOGITS
export BENCH_MIN_START_GIB=24 BENCH_MIN_RUN_GIB=2 BENCH_MIN_RECOVERED_GIB=24 BENCH_MAX_TEMP_C=97
mkdir -p "$output/runs"

ready() {
    local ac available temperature value path loops=0
    while true; do
        read -r ac < /sys/class/power_supply/ACAD/online
        available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
        temperature=0
        for path in /sys/class/thermal/thermal_zone*/temp; do
            [[ -r "$path" ]] || continue
            read -r value < "$path"
            (( value > temperature )) && temperature=$value
        done
        if (( ac == 1 && available >= 25165824 && temperature <= 60000 )); then return; fi
        if (( ac == 1 && available >= 25165824 && temperature < 90000 && loops >= 12 )); then return; fi
        sleep 15
        ((loops += 1))
    done
}

run_case() {
    local label=$1
    shift
    ready
    "$repo/bench/bench_guard" "$output/runs" "$label" 2400 "$repo" -- "$@"
}

for backend in cpu-wide xe vk; do
    binary="$output/compare_pp_tg_llama"
    params=(NGL=0 FA=0 THREADS=6 BTHREADS=20 UBATCH=512)
    if [[ $backend == xe ]]; then binary="$repo/bench/compare_pp_tg_xe"; params=(); fi
    if [[ $backend == vk ]]; then params=(NGL=99 FA=0 THREADS=1 BTHREADS=1 UBATCH=1024); fi
    for depth in 512 0 2048; do
        gen=0
        if (( depth == 0 )); then gen=128; fi
        run_case "standard-$backend-pp$depth-tg$gen" env GEN="$gen" TG_ONLY=1 "${params[@]}" taskset -c 0-19 "$binary" "$model" "$output/random-seed1.ids" "$depth" 3
    done
done

python3 -B "$repo/bench/compare_pp_tg_verify.py" --runs "$output/runs"
