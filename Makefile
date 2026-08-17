CC=gcc
BUILD_COMMIT=$(shell git rev-parse --short=9 HEAD 2>/dev/null || printf unknown)
BASE_CFLAGS=-O3 -march=native -std=c11 -Wall -Wextra -pthread
NUMERIC_SOURCE_HASH=$(shell sha256sum xenolith.c xenolith.cl | sha256sum | cut -d' ' -f1)
NUMERIC_CFLAGS_HASH=$(shell printf '%s' '$(BASE_CFLAGS)' | sha256sum | cut -d' ' -f1)
CFLAGS=$(BASE_CFLAGS) -DXE_BUILD_COMMIT='"$(BUILD_COMMIT)"' \
	-DXE_NUMERIC_SOURCE_HASH='"$(NUMERIC_SOURCE_HASH)"' \
	-DXE_NUMERIC_CFLAGS_HASH='"$(NUMERIC_CFLAGS_HASH)"'
LDLIBS=-lm -lze_loader

GPU_SPV=xenolith_gpu.spv
GPU_OBJ=xenolith_gpu_spv.o

xenolith: main.o xenolith.o format.o kvstore.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -o $@ main.o xenolith.o format.o kvstore.o $(GPU_OBJ) $(LDLIBS)

main.o: main.c xenolith.h
xenolith.o: xenolith.c xenolith.h format.h
format.o: format.c format.h
kvstore.o: kvstore.c kvstore.h xenolith.h format.h

clean:
	rm -f *.o xenolith tests/certify tests/test_kv tests/test_decode \
		tests/test_model_decode tests/test_fixture_decode tests/bench_decode \
		tests/test_prefill_projection tests/test_prefill_qkv tests/test_prefill_swa \
		tests/test_prefill_dense tests/test_prefill_layer tests/test_prefill_session \
		tests/test_session_sync tests/test_format tests/test_snapshot \
		tests/test_snapshot_model \
		tests/test_kvstore \
		tests/bench_attention tests/bench_tg tests/bench_b3b tests/bench_b3b_8e \
		tests/bench_b3b.spv tests/bench_b3b_adlp.spv \
		tests/bench_prefill_gemm tests/bench_prefill_gemm_down \
		tests/bench_prefill_gemm.spv $(GPU_SPV)

$(GPU_SPV): xenolith.cl
	ocloc compile -file $< -device 0xa7a0 -spv_only -output xenolith_gpu \
		-output_no_suffix -out_dir . -options '-cl-std=CL3.0' -q

$(GPU_OBJ): $(GPU_SPV)
	ld -r -b binary -o $@ $<
	objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents $@

LLAMA_DIR=../llama.cpp
LLAMA_BUILD=$(LLAMA_DIR)/build-cpu
LLAMA_LIBDIR=$(LLAMA_BUILD)/bin
LLAMA_CFLAGS=-I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LLAMA_LDFLAGS=-L$(LLAMA_LIBDIR) -Wl,-rpath,$(abspath $(LLAMA_LIBDIR))
LLAMA_LDLIBS=-lllama -lggml -lggml-base

TEST_CFLAGS=-O2 -std=c11 -Wall -Wextra -pthread

VOCAB_DIR=$(LLAMA_DIR)/models
MODEL?=
GOLDEN_PROMPTS=short long mixed ws nl nlonly json
ORACLE_PROMPTS=short long

tests/certify: tests/certify.c xenolith.o format.o xenolith.h $(GPU_OBJ)
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/certify.c xenolith.o format.o $(GPU_OBJ) $(LDLIBS)

tests/test_format: tests/test_format.c format.o format.h
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_format.c format.o -pthread

tests/test_snapshot: tests/test_snapshot.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_snapshot.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_snapshot_model: tests/test_snapshot_model.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_snapshot_model.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_kvstore: tests/test_kvstore.c xenolith.c xenolith.h kvstore.o kvstore.c kvstore.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_kvstore.c kvstore.o format.o $(GPU_OBJ) $(LDLIBS)

tests/test_kv: tests/test_kv.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_kv.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_decode: tests/test_decode.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_decode.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_model_decode: tests/test_model_decode.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_model_decode.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_projection: tests/test_prefill_projection.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_projection.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_qkv: tests/test_prefill_qkv.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_qkv.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_swa: tests/test_prefill_swa.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_swa.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_dense: tests/test_prefill_dense.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_dense.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_layer: tests/test_prefill_layer.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_layer.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_session: tests/test_prefill_session.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_session.c format.o $(GPU_OBJ) $(LDLIBS)

tests/test_session_sync: tests/test_session_sync.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_session_sync.c format.o $(GPU_OBJ) $(LDLIBS)

tests/bench_decode: tests/bench_decode.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/bench_decode.c format.o $(GPU_OBJ) $(LDLIBS)

tests/bench_attention: tests/bench_attention.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/bench_attention.c format.o $(GPU_OBJ) $(LDLIBS)

tests/bench_tg: tests/bench_tg.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/bench_tg.c format.o $(GPU_OBJ) $(LDLIBS)

tests/bench_b3b: tests/bench_b3b.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/bench_b3b.c format.o $(GPU_OBJ) $(LDLIBS)

tests/bench_b3b_8e: tests/bench_b3b.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -DXE_WORKERS=8 -DXE_WORKER_ECORES -I. \
		-o $@ tests/bench_b3b.c format.o $(GPU_OBJ) $(LDLIBS)

tests/bench_b3b.spv: tests/bench_b3b.cl
	ocloc compile -file $< -device 0xa7a0 -spv_only -output bench_b3b \
		-output_no_suffix -out_dir tests -options '-cl-std=CL3.0' -q

bench-b3b: tests/bench_b3b tests/bench_b3b.spv

bench-b14: tests/bench_b3b tests/bench_b3b_8e tests/bench_b3b.spv

tests/bench_prefill_gemm: tests/bench_prefill_gemm.c
	$(CC) $(TEST_CFLAGS) -o $@ $< $(LDLIBS)

tests/bench_prefill_gemm_down: tests/bench_prefill_gemm.c
	$(CC) $(TEST_CFLAGS) -DBENCH_MOE_N=2816 -DBENCH_MOE_BLOCKS=22 \
		-o $@ $< $(LDLIBS)

tests/bench_prefill_gemm.spv: tests/bench_prefill_gemm.cl
	ocloc compile -file $< -device 0xa7a0 -spv_only \
		-output bench_prefill_gemm -output_no_suffix -out_dir tests \
		-options '-cl-std=CL3.0' -q

bench-prefill-gemm: tests/bench_prefill_gemm tests/bench_prefill_gemm_down \
	tests/bench_prefill_gemm.spv

tests/test_fixture_decode: tests/test_fixture_decode.c xenolith.c xenolith.h format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_fixture_decode.c format.o $(GPU_OBJ) $(LDLIBS)

check-kv: tests/test_kv
	./tests/test_kv

check: require-model tests/certify tests/test_kv tests/test_decode
	./tests/test_kv
	./tests/test_decode
	XENOLITH_MODEL="$(MODEL)" ./tests/certify

golden: require-model test-tools
	mkdir -p tests/golden tests/fixtures
	for p in $(GOLDEN_PROMPTS); do \
		XENOLITH_MODEL="$(MODEL)" tests/tokenize_prompt "$(MODEL)" tests/prompts/$$p.txt > tests/golden/$$p.ids; \
	done
	for p in $(ORACLE_PROMPTS); do \
		ids=`tests/tokenize_prompt "$(MODEL)" tests/prompts/$$p.txt 2>/dev/null`; \
		tests/dump_logits "$(MODEL)" "$$ids" tests/golden/$$p.logits; \
	done
	cp $(VOCAB_DIR)/ggml-vocab-gemma-4.gguf.inp tests/fixtures/ggml-vocab-gemma-4.gguf.inp
	cp $(VOCAB_DIR)/ggml-vocab-gemma-4.gguf.out tests/fixtures/ggml-vocab-gemma-4.gguf.out

TEST_TOOLS=tests/dump_logits tests/tokenize_prompt tests/dump_layers

test-tools: $(TEST_TOOLS)

tests/dump_logits: tests/dump_logits.c
	$(CC) $(TEST_CFLAGS) $(LLAMA_CFLAGS) -o $@ $< $(LLAMA_LDFLAGS) $(LLAMA_LDLIBS) $(LDLIBS)

tests/dump_layers: tests/dump_layers.c
	$(CC) $(TEST_CFLAGS) $(LLAMA_CFLAGS) -o $@ $< $(LLAMA_LDFLAGS) $(LLAMA_LDLIBS) $(LDLIBS)

tests/tokenize_prompt: tests/tokenize_prompt.c
	$(CC) $(TEST_CFLAGS) $(LLAMA_CFLAGS) -o $@ $< $(LLAMA_LDFLAGS) $(LLAMA_LDLIBS) $(LDLIBS)

test-tools-clean:
	rm -f $(TEST_TOOLS)

require-model:
	@test -f "$(MODEL)" || { printf '%s\n' 'Set MODEL to the path of the Gemma 4 GGUF file.' >&2; exit 2; }

.PHONY: require-model clean check check-kv golden test-tools test-tools-clean bench-b3b bench-b14 \
	bench-prefill-gemm
