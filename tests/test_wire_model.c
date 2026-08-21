#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "../wire.h"
#include "../json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(condition) do { \
    if (!(condition)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                #condition); \
    } \
} while (0)

static const char *store_fault_point;
int kvstore_fault(const char *point) {
    return store_fault_point && strcmp(store_fault_point, point) == 0;
}

typedef struct {
    char text[16384];
    size_t text_length;
    uint64_t call_ids[8];
    char call_names[8][64];
    char call_args[8][4096];
    size_t calls;
    uint32_t stop;
    wire_usage usage;
    wire_marker marker;
    int error;
    int progress_events;
    int checkpoint_attempted;
    wire_checkpoint_report checkpoint;
    int resume_attempted;
    wire_resume_report resume;
} run_result;

static void run_generation(wire *w, run_result *out, int cancel_after_text) {
    memset(out, 0, sizeof *out);
    out->marker = WIRE_MARKER_NONE;
    for (;;) {
        wire_event event;
        wire_status status = wire_next_event(w, &event);
        if (status != WIRE_OK) {
            out->error = 1;
            fprintf(stderr, "stream failed: %s\n", wire_error_text(w));
            return;
        }
        switch (event.kind) {
        case WIRE_EVENT_PROGRESS:
            out->progress_events++;
            break;
        case WIRE_EVENT_TEXT_DELTA:
            if (out->text_length + event.text_length <
                sizeof out->text - 1) {
                memcpy(out->text + out->text_length, event.text,
                       event.text_length);
                out->text_length += event.text_length;
                out->text[out->text_length] = '\0';
            }
            if (cancel_after_text) {
                wire_cancel(w);
                cancel_after_text = 0;
            }
            break;
        case WIRE_EVENT_TOOLCALL_END:
            if (out->calls < 8) {
                out->call_ids[out->calls] = event.call_id;
                snprintf(out->call_names[out->calls], 64, "%s",
                         event.call_name);
                snprintf(out->call_args[out->calls], 4096, "%s",
                         event.arguments_json);
            }
            out->calls++;
            break;
        case WIRE_EVENT_DONE:
            out->stop = event.stop;
            out->usage = event.usage;
            out->marker = event.marker;
            out->checkpoint_attempted = event.checkpoint_attempted;
            out->checkpoint = event.checkpoint;
            out->resume_attempted = event.resume_attempted;
            out->resume = event.resume;
            return;
        case WIRE_EVENT_ERROR:
            out->error = 1;
            fprintf(stderr, "stream error: %s\n",
                    event.error_text ? event.error_text : "");
            return;
        default:
            break;
        }
    }
}

static int file_contains(const char *path, const char *needle) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        fclose(f);
        return 0;
    }
    char *data = malloc((size_t)size + 1);
    if (!data || fread(data, 1, (size_t)size, f) != (size_t)size) {
        free(data);
        fclose(f);
        return 0;
    }
    data[size] = '\0';
    fclose(f);
    int found = strstr(data, needle) != NULL;
    free(data);
    return found;
}

int main(int argc, char **argv) {
    if (argc < 2 || !argv[1][0]) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    const char *model = argv[1];
    char state_dir[] = "/tmp/xenolith-test-wm-state-XXXXXX";
    char cache_dir[] = "/tmp/xenolith-test-wm-cache-XXXXXX";
    char ndjson_dir[] = "/tmp/xenolith-test-wm-ndjson-XXXXXX";
    CHECK(mkdtemp(state_dir) != NULL);
    CHECK(mkdtemp(cache_dir) != NULL);
    CHECK(mkdtemp(ndjson_dir) != NULL);

    xe_engine *e = xe_engine_open(model);
    wire *w = NULL;
    CHECK(wire_open(&w, e, state_dir, cache_dir) == WIRE_OK);
    if (!w) return 1;

    profile_tool weather = {
        "get_weather",
        "Get the current weather for a city.",
        "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":"
        "\"string\",\"description\":\"The city name, e.g. Kyoto\"}},"
        "\"required\":[\"city\"]}"
    };
    conversation_id id;
    wire_marker marker;
    CHECK(wire_session_create(w, "You are a helpful assistant with tools.",
                              &weather, 1, &id, &marker) == WIRE_OK);

    wire_gen_params params = { 1.0f, 1, 1.0f, 300, 42 };
    wire_message user;
    memset(&user, 0, sizeof user);
    user.kind = WIRE_MESSAGE_USER;
    user.text = "What's the weather in Kyoto right now?";
    CHECK(wire_append(w, &user, &marker) == WIRE_OK);
    CHECK(wire_generate(w, &params) == WIRE_OK);

    run_result turn1;
    run_generation(w, &turn1, 0);
    CHECK(!turn1.error);
    CHECK(turn1.stop == WIRE_STOP_TOOL_USE);
    CHECK(turn1.calls == 1);
    CHECK(strcmp(turn1.call_names[0], "get_weather") == 0);
    CHECK(strstr(turn1.call_args[0], "Kyoto") != NULL);
    CHECK(turn1.usage.output > 0);
    CHECK(turn1.usage.input > 0);
    CHECK(turn1.usage.cache_read == 0);
    CHECK(wire_pending_calls(w, NULL, 0) == 1);

    wire_message result;
    memset(&result, 0, sizeof result);
    result.kind = WIRE_MESSAGE_TOOL_RESULT;
    result.call_id = turn1.call_ids[0];
    result.text = "21C, light rain, humidity 81%";
    wire_marker result_marker;
    CHECK(wire_append(w, &result, &result_marker) == WIRE_OK);
    CHECK(wire_pending_calls(w, NULL, 0) == 0);

    wire_gen_params greedy = { 1.0f, 1, 1.0f, 300, 0 };
    CHECK(wire_generate(w, &greedy) == WIRE_OK);
    run_result turn2;
    run_generation(w, &turn2, 0);
    CHECK(!turn2.error);
    CHECK(turn2.stop == WIRE_STOP_STOP);
    CHECK(turn2.text_length > 0);
    CHECK(strstr(turn2.text, "21") != NULL);
    CHECK(turn2.usage.cache_read + 5 >= turn1.usage.total);
    CHECK(turn2.usage.input < 60);

    CHECK(wire_rewind(w, result_marker) == WIRE_OK);
    uint64_t cost = 12345;
    CHECK(wire_rewind_cost(w, result_marker, &cost) == WIRE_OK);
    CHECK(cost <= 1);
    CHECK(wire_generate(w, &greedy) == WIRE_OK);
    run_result turn3;
    run_generation(w, &turn3, 0);
    CHECK(!turn3.error);
    CHECK(turn3.stop == WIRE_STOP_STOP);
    CHECK(turn3.text_length == turn2.text_length &&
          memcmp(turn3.text, turn2.text, turn2.text_length) == 0);
    CHECK(turn3.usage.input <= 2);
    CHECK(turn3.usage.cache_read > 0);

    /* 3.7 finding 2: a failed save is reported, never silent. */
    wire_checkpoint_report ckpt;
    store_fault_point = "snapshot-synced";
    CHECK(wire_checkpoint(w, &ckpt) == WIRE_OK);
    CHECK(ckpt.saved == 0 && ckpt.reason == WIRE_CKPT_IO);
    CHECK(ckpt.tokens == turn3.usage.total);
    wire_open_report failed_report;
    CHECK(wire_session_open(w, &id, &failed_report) == WIRE_OK);
    CHECK(failed_report.zero_prefill == 0);
    store_fault_point = NULL;
    CHECK(wire_checkpoint(w, &ckpt) == WIRE_OK);
    CHECK(ckpt.saved == 1 && ckpt.reason == WIRE_CKPT_SAVED);
    CHECK(ckpt.tokens == turn3.usage.total);
    CHECK(wire_checkpoint(w, &ckpt) == WIRE_OK);
    CHECK(ckpt.saved == 0 && ckpt.reason == WIRE_CKPT_NOTHING_NEW);
    uint64_t tokens_before;
    tokens_before = turn3.usage.total;
    wire_close(w);
    w = NULL;
    CHECK(wire_open(&w, e, state_dir, cache_dir) == WIRE_OK);
    wire_open_report report;
    CHECK(wire_session_open(w, &id, &report) == WIRE_OK);
    CHECK(report.token_count == tokens_before);
    CHECK(report.zero_prefill == 1);
    CHECK(report.resume_stale == 0);
    CHECK(report.turn_open == 0);

    user.text = "Thanks. Reply with one short sentence.";
    CHECK(wire_append(w, &user, &marker) == WIRE_OK);
    wire_open_report stale_report;
    CHECK(wire_session_open(w, &id, &stale_report) == WIRE_OK);
    CHECK(stale_report.zero_prefill == 0 && stale_report.resume_stale == 1);
    CHECK(wire_generate(w, &greedy) == WIRE_OK);
    run_result turn4;
    run_generation(w, &turn4, 0);
    CHECK(!turn4.error);
    CHECK(turn4.usage.cache_read >= tokens_before);
    CHECK(turn4.usage.input < 40);
    CHECK(turn4.resume_attempted == 1);
    CHECK(turn4.resume.loaded == 1);
    CHECK(turn4.resume.tokens == tokens_before);

    user.text = "Tell me a very long story about the sea.";
    CHECK(wire_append(w, &user, &marker) == WIRE_OK);
    CHECK(wire_generate(w, &greedy) == WIRE_OK);
    run_result cancelled;
    run_generation(w, &cancelled, 1);
    CHECK(!cancelled.error);
    CHECK(cancelled.stop == WIRE_STOP_ABORTED);
    CHECK(cancelled.usage.output >= 1);
    wire_open_report after_cancel;
    CHECK(wire_session_open(w, &id, &after_cancel) == WIRE_OK);
    CHECK(after_cancel.turn_open == 1);

    wire_gen_params tiny = { 1.0f, 1, 1.0f, 30, 0 };
    user.text = "Never mind, just say bye.";
    CHECK(wire_append(w, &user, &marker) == WIRE_OK);
    CHECK(wire_generate(w, &tiny) == WIRE_OK);
    run_result resumed;
    run_generation(w, &resumed, 0);
    CHECK(!resumed.error);
    CHECK(resumed.stop == WIRE_STOP_STOP ||
          resumed.stop == WIRE_STOP_LENGTH);

    wire_message hello;
    memset(&hello, 0, sizeof hello);
    hello.kind = WIRE_MESSAGE_USER;
    hello.text = "Say hello.";
    wire_gen_params eph_params = { 1.0f, 1, 1.0f, 16, 0 };
    CHECK(wire_ephemeral_generate(w, "You answer with one short word.",
                                  NULL, 0, &hello, 1, &eph_params)
          == WIRE_OK);
    run_result ephemeral;
    run_generation(w, &ephemeral, 0);
    CHECK(!ephemeral.error);
    CHECK(ephemeral.marker == WIRE_MARKER_NONE);
    CHECK(ephemeral.usage.total > 0);
    CHECK(wire_session_open(w, &id, &report) == WIRE_OK);
    CHECK(report.token_count > tokens_before);

    wire_close(w);
    xe_engine_close(e);

    char requests_path[256], output_path[256], command[2048];
    snprintf(requests_path, sizeof requests_path, "%s/requests.ndjson",
             ndjson_dir);
    snprintf(output_path, sizeof output_path, "%s/output.ndjson",
             ndjson_dir);
    FILE *requests = fopen(requests_path, "w");
    CHECK(requests != NULL);
    if (requests) {
        fprintf(requests, "{\"op\":\"describe\"}\n");
        fprintf(requests,
                "{\"op\":\"create\",\"system\":\"You are a helpful "
                "assistant with tools.\",\"tools\":[{\"name\":"
                "\"get_weather\",\"description\":\"Get the current "
                "weather for a city.\",\"parameters\":{\"type\":"
                "\"object\",\"properties\":{\"city\":{\"type\":"
                "\"string\",\"description\":\"The city name, e.g. "
                "Kyoto\"}},\"required\":[\"city\"]}}]}\n");
        fprintf(requests,
                "{\"op\":\"append\",\"role\":\"user\",\"text\":"
                "\"What's the weather in Kyoto right now?\"}\n");
        fprintf(requests,
                "{\"op\":\"generate\",\"temperature\":1,\"top_k\":1,"
                "\"max_tokens\":200,\"seed\":42}\n");
        fprintf(requests, "{\"op\":\"pending\"}\n");
        fprintf(requests, "{\"op\":\"history\"}\n");
        fprintf(requests, "{\"op\":\"list\"}\n");
        fclose(requests);
    }
    snprintf(command, sizeof command,
             "XDG_STATE_HOME=%s ./xenolith wire %s --state %s/state "
             "--cache %s/cache < %s > %s 2>%s/stderr.log",
             ndjson_dir, model, ndjson_dir, ndjson_dir, requests_path,
             output_path, ndjson_dir);
    CHECK(system(command) == 0);
    CHECK(file_contains(output_path, "\"ok\":true"));
    CHECK(file_contains(output_path, "\"event\":\"start\""));
    CHECK(file_contains(output_path, "\"event\":\"toolcall_end\""));
    CHECK(file_contains(output_path, "\"name\":\"get_weather\""));
    CHECK(file_contains(output_path, "\"stop\":\"tool_use\""));
    CHECK(file_contains(output_path, "\"kind\":\"assistant\""));
    CHECK(file_contains(output_path, "\"calls\":[1]"));

    if (failures) {
        fprintf(stderr, "test_wire_model: %d failures\n", failures);
        return 1;
    }
    printf("test_wire_model: all checks passed\n");
    return 0;
}
