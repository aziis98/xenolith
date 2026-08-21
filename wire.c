#define _POSIX_C_SOURCE 200809L

#include "wire.h"
#include "format.h"
#include "json.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    WIRE_GEN_NONE = 0,
    WIRE_GEN_DURABLE = 1,
    WIRE_GEN_EPHEMERAL = 2,
    WIRE_PHASE_PREFILL = 0,
    WIRE_PHASE_DECODE = 1,
    WIRE_PREFILL_CHUNK = 512
};

typedef struct {
    uint64_t id;
    char *name;
    char *arguments;
    int complete;
} wire_gen_call;

struct wire {
    xe_engine *engine;
    profile *prof;
    conversation_store *cstore;
    kvstore *kv;
    int context;

    conversation *current;
    conversation_id current_id;
    int has_current;
    uint64_t next_call_id;
    uint64_t saved_tokens;
    int64_t saved_at;
    int ckpt_failures;        /* consecutive autosave failures (backoff) */
    int ckpt_autosave_off;    /* set on budget: the session cannot fit, stop trying */
    int kv_open_status;       /* kvstore_status of the open attempt */
    int autosave_attempted;   /* last turn ran an autosave */
    wire_checkpoint_report autosave;
    int resume_attempted;     /* last generation tried a snapshot load */
    wire_resume_report resume;

    xe_session *session;
    xe_session *ephemeral;
    xe_session *gen_session;

    int gen_kind;
    int gen_phase;
    int cancel_requested;
    int started_emitted;
    int first_sync_done;
    uint64_t generation_id;
    conversation_settings gen_settings;
    xe_sampler sampler;
    int32_t gen_max_tokens;

    int32_t *prompt;
    int prompt_length;
    int prompt_synced;
    int prompt_common;
    int framing_offset;
    int sampled_length;
    wire_usage usage;

    json_writer content;
    json_writer render;
    wire_gen_call *calls;
    size_t call_count;
    size_t call_capacity;
    int call_open;

    char error_text[512];
    json_writer tools_scratch;
    json_writer calls_scratch;
};

const char *wire_status_code(wire_status status) {
    switch (status) {
    case WIRE_OK: return "ok";
    case WIRE_CONTEXT_LENGTH_EXCEEDED: return "context_length_exceeded";
    case WIRE_SESSION_NOT_FOUND: return "session_not_found";
    case WIRE_MARKER_UNAVAILABLE: return "marker_unavailable";
    case WIRE_INVALID_ARGUMENT: return "invalid_request";
    case WIRE_BUSY: return "busy";
    case WIRE_IO: return "io_error";
    case WIRE_NOMEM: return "out_of_memory";
    }
    return "unknown";
}

const char *wire_error_text(const wire *w) {
    return w && w->error_text[0] ? w->error_text : "";
}

static wire_status wire_fail(wire *w, wire_status status,
                             const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(w->error_text, sizeof w->error_text, fmt, args);
    va_end(args);
    return status;
}

static wire_status wire_from_conversation(wire *w, conversation_status s) {
    switch (s) {
    case CONVERSATION_OK:
        return WIRE_OK;
    case CONVERSATION_MISS:
        return wire_fail(w, WIRE_SESSION_NOT_FOUND, "session not found");
    case CONVERSATION_LOCKED:
        return wire_fail(w, WIRE_BUSY, "session record is locked");
    case CONVERSATION_IO:
        return wire_fail(w, WIRE_IO,
                         "session record io failed, temporarily unavailable");
    case CONVERSATION_NOMEM:
        return wire_fail(w, WIRE_NOMEM, "out of memory");
    case CONVERSATION_DAMAGED:
        return wire_fail(w, WIRE_INVALID_ARGUMENT,
                         "session record is damaged");
    case CONVERSATION_VERSION:
        return wire_fail(w, WIRE_INVALID_ARGUMENT,
                         "session record version is unsupported");
    default:
        return wire_fail(w, WIRE_INVALID_ARGUMENT, "%s",
                         conversation_status_name(s));
    }
}

static wire_status wire_from_profile(wire *w, profile_status s) {
    switch (s) {
    case PROFILE_OK:
        return WIRE_OK;
    case PROFILE_NOMEM:
        return wire_fail(w, WIRE_NOMEM, "out of memory");
    default:
        return wire_fail(w, WIRE_INVALID_ARGUMENT, "render failed");
    }
}

static int64_t wire_now(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) return 0;
    return (int64_t)now.tv_sec * INT64_C(1000000000) + now.tv_nsec;
}

static int wire_default_cache_dir(char *out, size_t cap) {
    const char *cache = getenv("XDG_CACHE_HOME");
    int n;
    if (cache && *cache == '/') {
        n = snprintf(out, cap, "%s/xenolith/kv", cache);
    } else {
        const char *home = getenv("HOME");
        if (!home || *home != '/') return 0;
        n = snprintf(out, cap, "%s/.cache/xenolith/kv", home);
    }
    return n > 0 && (size_t)n < cap;
}

wire_status wire_open(wire **out, xe_engine *engine, const char *state_dir,
                      const char *cache_dir) {
    if (!out || !engine) return WIRE_INVALID_ARGUMENT;
    *out = NULL;
    wire *w = calloc(1, sizeof *w);
    if (!w) return WIRE_NOMEM;
    w->engine = engine;
    w->context = xe_context_size(engine);
    w->call_open = -1;
    if (profile_open(&w->prof, engine) != PROFILE_OK) {
        free(w);
        return WIRE_INVALID_ARGUMENT;
    }
    char state_path[4096];
    if (!state_dir) {
        if (!conversation_default_state_dir(state_path, sizeof state_path)) {
            profile_close(w->prof);
            free(w);
            return WIRE_IO;
        }
        state_dir = state_path;
    }
    conversation_status opened = conversation_store_open(&w->cstore,
                                                         state_dir);
    if (opened != CONVERSATION_OK) {
        wire_status status = wire_from_conversation(w, opened);
        profile_close(w->prof);
        free(w);
        return status;
    }
    char cache_path[4096];
    if (!cache_dir && wire_default_cache_dir(cache_path, sizeof cache_path))
        cache_dir = cache_path;
    w->kv_open_status = KVSTORE_OK;
    if (cache_dir) {
        w->kv_open_status = (int)kvstore_open(&w->kv, cache_dir,
                                              KVSTORE_DEFAULT_BUDGET);
        if (w->kv_open_status != KVSTORE_OK) w->kv = NULL;
    }
    *out = w;
    return WIRE_OK;
}

int wire_kvstore_open_status(const wire *w) {
    return w ? w->kv_open_status : (int)KVSTORE_INVALID_ARGUMENT;
}

const char *wire_ckpt_reason_name(wire_ckpt_reason reason) {
    switch (reason) {
    case WIRE_CKPT_SAVED: return "saved";
    case WIRE_CKPT_NO_KVSTORE: return "no_kvstore";
    case WIRE_CKPT_EMPTY: return "empty";
    case WIRE_CKPT_NOTHING_NEW: return "nothing_new";
    case WIRE_CKPT_KV_DIVERGED: return "kv_diverged";
    case WIRE_CKPT_BUDGET: return "budget";
    case WIRE_CKPT_IO: return "io";
    case WIRE_CKPT_REJECTED: return "rejected";
    case WIRE_CKPT_RECORD_FAILED: return "record_failed";
    }
    return "unknown";
}

const char *wire_resume_reason_name(wire_resume_reason reason) {
    switch (reason) {
    case WIRE_RESUME_LOADED: return "loaded";
    case WIRE_RESUME_EVICTED: return "evicted";
    case WIRE_RESUME_MODEL_MISMATCH: return "model_mismatch";
    case WIRE_RESUME_TOKEN_MISMATCH: return "token_mismatch";
    case WIRE_RESUME_IO: return "io";
    case WIRE_RESUME_REJECTED: return "rejected";
    }
    return "unknown";
}

static void wire_calls_reset(wire *w) {
    for (size_t i = 0; i < w->call_count; i++) {
        free(w->calls[i].name);
        free(w->calls[i].arguments);
    }
    w->call_count = 0;
    w->call_open = -1;
}

void wire_close(wire *w) {
    if (!w) return;
    if (w->current) conversation_close(w->current);
    if (w->session) xe_session_free(w->session);
    if (w->ephemeral) xe_session_free(w->ephemeral);
    conversation_store_close(w->cstore);
    kvstore_close(w->kv);
    profile_close(w->prof);
    wire_calls_reset(w);
    free(w->calls);
    free(w->prompt);
    json_writer_free(&w->content);
    json_writer_free(&w->render);
    json_writer_free(&w->tools_scratch);
    json_writer_free(&w->calls_scratch);
    free(w);
}

wire_status wire_describe(wire *w, wire_info *out) {
    if (!w || !out) return WIRE_INVALID_ARGUMENT;
    out->model = profile_model(w->prof);
    out->context_window = w->context;
    out->max_output = w->context;
    out->kvstore = w->kv != NULL;
    return WIRE_OK;
}

static uint32_t wire_turn(const conversation *c) {
    uint64_t count = conversation_visible_count(c);
    if (!count) return PROFILE_TURN_PADDED;
    const conversation_event *last = conversation_event_at(
        c, conversation_visible_index(c, count - 1));
    if (!last) return PROFILE_TURN_PADDED;
    switch (last->type) {
    case CONVERSATION_EVENT_MESSAGE:
        if (last->role != CONVERSATION_ROLE_ASSISTANT)
            return PROFILE_TURN_PADDED;
        if (last->token_count &&
            last->tokens[last->token_count - 1] == 106)
            return PROFILE_TURN_BARE;
        return PROFILE_TURN_OPEN;
    case CONVERSATION_EVENT_TOOL_RESULT:
        return PROFILE_TURN_OPEN;
    case CONVERSATION_EVENT_GENERATION_RESULT:
        switch (last->stop_reason) {
        case CONVERSATION_STOP_EOT_SAMPLED:
        case CONVERSATION_STOP_EOT_SYNTHETIC:
        case CONVERSATION_STOP_EOS:
            return PROFILE_TURN_BARE;
        default:
            return PROFILE_TURN_OPEN;
        }
    default:
        return PROFILE_TURN_PADDED;
    }
}

static wire_status wire_tools_json(wire *w, const profile_tool *tools,
                                   size_t tool_count) {
    json_writer_reset(&w->tools_scratch);
    json_raw(&w->tools_scratch, "[");
    for (size_t i = 0; i < tool_count; i++) {
        if (i) json_raw(&w->tools_scratch, ",");
        json_raw(&w->tools_scratch, "{\"name\":");
        json_string(&w->tools_scratch, tools[i].name,
                    strlen(tools[i].name));
        json_raw(&w->tools_scratch, ",\"description\":");
        json_string(&w->tools_scratch, tools[i].description,
                    strlen(tools[i].description));
        json_raw(&w->tools_scratch, ",\"parameters\":");
        if (tools[i].parameters_json && *tools[i].parameters_json)
            json_raw(&w->tools_scratch, tools[i].parameters_json);
        else
            json_raw(&w->tools_scratch, "null");
        json_raw(&w->tools_scratch, "}");
    }
    json_raw(&w->tools_scratch, "]");
    if (w->tools_scratch.failed)
        return wire_fail(w, WIRE_NOMEM, "out of memory");
    return WIRE_OK;
}

static wire_status wire_budget_check(wire *w, uint64_t total) {
    if (total + 1 < (uint64_t)w->context) return WIRE_OK;
    return wire_fail(w, WIRE_CONTEXT_LENGTH_EXCEEDED,
                     "prompt has %llu tokens, but the configured context "
                     "size is %d tokens",
                     (unsigned long long)total, w->context);
}

static wire_status wire_append_system_event(wire *w, conversation *c,
                                            const char *system,
                                            const profile_tool *tools,
                                            size_t tool_count,
                                            wire_marker *marker) {
    profile_render render;
    wire_status status = wire_from_profile(
        w, profile_render_system(w->prof, system, tools, tool_count,
                                 &render));
    if (status != WIRE_OK) return status;
    conversation_block blocks[2];
    uint32_t block_count = 1;
    blocks[0].format = CONVERSATION_BLOCK_TEXT;
    blocks[0].data = system ? system : "";
    blocks[0].length = system ? strlen(system) : 0;
    if (tool_count) {
        status = wire_tools_json(w, tools, tool_count);
        if (status != WIRE_OK) return status;
        blocks[1].format = CONVERSATION_BLOCK_JSON;
        blocks[1].data = w->tools_scratch.data;
        blocks[1].length = w->tools_scratch.length;
        block_count = 2;
    }
    status = wire_from_conversation(
        w, conversation_append_message(c, CONVERSATION_ROLE_SYSTEM, blocks,
                                       block_count, render.render,
                                       render.render_length, render.tokens,
                                       render.token_count));
    if (status != WIRE_OK) return status;
    if (marker) *marker = conversation_event_count(c) - 1;
    return WIRE_OK;
}

/* Consecutive transient failures push the next autosave out: 30 s, then
 * +30, +90, +210, +450 s on top of the 30 s window, capped. The explicit
 * checkpoint op bypasses this and a success resets it. */
static void wire_ckpt_backoff(wire *w, int64_t now) {
    if (w->ckpt_failures < 5) w->ckpt_failures++;
    int64_t window = INT64_C(30000000000);
    int64_t extra = window * ((INT64_C(1) << w->ckpt_failures) - 2);
    w->saved_at = now + extra;
}

static void wire_ckpt_reset(wire *w) {
    w->ckpt_failures = 0;
    w->ckpt_autosave_off = 0;
}

static void wire_checkpoint_now(wire *w, wire_checkpoint_report *out) {
    wire_checkpoint_report report;
    memset(&report, 0, sizeof report);
    if (!out) out = &report;
    memset(out, 0, sizeof *out);
    if (!w->kv) { out->reason = WIRE_CKPT_NO_KVSTORE; return; }
    if (!w->session || !w->has_current) { out->reason = WIRE_CKPT_EMPTY; return; }
    conversation *c = w->current;
    uint64_t total;
    const int32_t *tokens = conversation_tokens(c, &total);
    out->tokens = total;
    if (!total) { out->reason = WIRE_CKPT_EMPTY; return; }
    int position = xe_session_position(w->session);
    if (position <= 0) { out->reason = WIRE_CKPT_EMPTY; return; }
    if ((uint64_t)position > total) { out->reason = WIRE_CKPT_KV_DIVERGED; return; }
    xe_tokens full = { (int32_t *)tokens, (int)total, (int)total };
    if (xe_session_common(w->session, &full) < position) {
        out->reason = WIRE_CKPT_KV_DIVERGED;
        return;
    }
    if ((uint64_t)position < total) {
        xe_session_sync(w->session, &full);
        position = (int)total;
    }
    if ((uint64_t)position <= w->saved_tokens) {
        out->reason = WIRE_CKPT_NOTHING_NEW;
        return;
    }
    kvstore_save_options options;
    memset(&options, 0, sizeof options);
    options.rebuild_cost = (uint64_t)position;
    kvstore_id id;
    xe_snapshot_status snapshot_status;
    kvstore_status ks = kvstore_save(w->kv, w->session, &options, &id,
                                     &snapshot_status);
    int64_t now = wire_now();
    if (ks != KVSTORE_OK) {
        if (ks == KVSTORE_BUDGET) {
            out->reason = WIRE_CKPT_BUDGET;
            w->ckpt_autosave_off = 1;
        } else {
            out->reason = ks == KVSTORE_REJECTED ? WIRE_CKPT_REJECTED
                                                 : WIRE_CKPT_IO;
            wire_ckpt_backoff(w, now);
        }
        return;
    }
    if (conversation_append_snapshot_ref(c, &id, (uint64_t)position)
            != CONVERSATION_OK ||
        conversation_commit(c) != CONVERSATION_OK) {
        out->reason = WIRE_CKPT_RECORD_FAILED;
        wire_ckpt_backoff(w, now);
        return;
    }
    w->saved_tokens = (uint64_t)position;
    w->saved_at = now;
    wire_ckpt_reset(w);
    out->saved = 1;
    out->reason = WIRE_CKPT_SAVED;
    out->tokens = (uint64_t)position;
}

static void wire_park(wire *w, wire_checkpoint_report *out) {
    if (out) memset(out, 0, sizeof *out);
    if (!w->has_current) return;
    wire_checkpoint_now(w, out);
    conversation_close(w->current);
    w->current = NULL;
    w->has_current = 0;
}

wire_status wire_session_create(wire *w, const char *system,
                                const profile_tool *tools, size_t tool_count,
                                conversation_id *id, wire_marker *marker) {
    if (!w || !id) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind != WIRE_GEN_NONE)
        return wire_fail(w, WIRE_BUSY, "generation in progress");
    conversation *c = NULL;
    wire_status status = wire_from_conversation(
        w, conversation_create(w->cstore, &c, id));
    if (status != WIRE_OK) return status;
    wire_marker system_marker = 0;
    status = wire_append_system_event(w, c, system, tools, tool_count,
                                      &system_marker);
    if (status == WIRE_OK)
        status = wire_from_conversation(w, conversation_commit(c));
    if (status != WIRE_OK) {
        conversation_close(c);
        conversation_delete(w->cstore, id);
        return status;
    }
    wire_park(w, NULL);
    w->current = c;
    w->current_id = *id;
    w->has_current = 1;
    w->next_call_id = 1;
    w->saved_tokens = 0;
    w->saved_at = 0;
    wire_ckpt_reset(w);
    if (marker) *marker = system_marker;
    return WIRE_OK;
}

static void wire_scan_call_ids(wire *w, conversation *c) {
    uint64_t next = 1;
    uint64_t count = conversation_event_count(c);
    for (uint64_t i = 0; i < count; i++) {
        const conversation_event *ev = conversation_event_at(c, i);
        if (ev->type == CONVERSATION_EVENT_TOOL_STARTED &&
            ev->call_id >= next)
            next = ev->call_id + 1;
    }
    w->next_call_id = next;
}

static void wire_open_report_fill(wire *w, wire_open_report *out) {
    conversation *c = w->current;
    uint64_t tokens;
    conversation_tokens(c, &tokens);
    uint64_t visible = conversation_visible_count(c);
    out->token_count = tokens;
    out->marker = visible ? conversation_visible_index(c, visible - 1)
                          : WIRE_MARKER_NONE;
    out->turn_open = wire_turn(c) == PROFILE_TURN_OPEN;
    kvstore_id snapshot;
    uint64_t boundary = 0;
    int has_snapshot = conversation_snapshot_current(c, &snapshot, &boundary);
    out->zero_prefill = has_snapshot && boundary == tokens && tokens > 0;
    out->resume_stale = has_snapshot && !out->zero_prefill && boundary > 0;
    out->pending_calls = conversation_unknown_tool_calls(c, NULL, 0);
}

wire_status wire_session_open(wire *w, const conversation_id *id,
                              wire_open_report *out) {
    if (!w || !id) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind != WIRE_GEN_NONE)
        return wire_fail(w, WIRE_BUSY, "generation in progress");
    if (w->has_current &&
        memcmp(w->current_id.bytes, id->bytes, 16) == 0) {
        if (out) wire_open_report_fill(w, out);
        return WIRE_OK;
    }
    conversation *c = NULL;
    wire_status status = wire_from_conversation(
        w, conversation_open(w->cstore, id, &c));
    if (status != WIRE_OK) return status;
    wire_park(w, NULL);
    w->current = c;
    w->current_id = *id;
    w->has_current = 1;
    wire_scan_call_ids(w, c);
    kvstore_id snapshot;
    uint64_t boundary = 0;
    w->saved_tokens = conversation_snapshot_current(c, &snapshot, &boundary)
                      ? boundary : 0;
    w->saved_at = 0;
    wire_ckpt_reset(w);
    if (out) wire_open_report_fill(w, out);
    return WIRE_OK;
}

wire_status wire_session_list(wire *w, conversation_summary **out,
                              size_t *count) {
    if (!w || !out || !count) return WIRE_INVALID_ARGUMENT;
    return wire_from_conversation(
        w, conversation_list(w->cstore, out, count));
}

wire_status wire_session_stat(wire *w, const conversation_id *id,
                              conversation_summary *out) {
    if (!w || !id || !out) return WIRE_INVALID_ARGUMENT;
    conversation_summary *all = NULL;
    size_t count = 0;
    wire_status status = wire_from_conversation(
        w, conversation_list(w->cstore, &all, &count));
    if (status != WIRE_OK) return status;
    status = wire_fail(w, WIRE_SESSION_NOT_FOUND, "session not found");
    for (size_t i = 0; i < count; i++)
        if (memcmp(all[i].id.bytes, id->bytes, 16) == 0) {
            *out = all[i];
            status = WIRE_OK;
            break;
        }
    free(all);
    return status;
}

wire_status wire_session_delete(wire *w, const conversation_id *id) {
    if (!w || !id) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind != WIRE_GEN_NONE)
        return wire_fail(w, WIRE_BUSY, "generation in progress");
    if (w->has_current &&
        memcmp(w->current_id.bytes, id->bytes, 16) == 0) {
        conversation_close(w->current);
        w->current = NULL;
        w->has_current = 0;
    }
    return wire_from_conversation(w,
                                  conversation_delete(w->cstore, id));
}

size_t wire_pending_calls(wire *w, uint64_t *call_ids, size_t cap) {
    if (!w || !w->has_current) return 0;
    return conversation_unknown_tool_calls(w->current, call_ids, cap);
}

uint64_t wire_history_count(wire *w) {
    if (!w || !w->has_current) return 0;
    return conversation_visible_count(w->current);
}

static const char *wire_tool_name_of(conversation *c, uint64_t call_id) {
    uint64_t count = conversation_event_count(c);
    for (uint64_t i = count; i > 0; i--) {
        const conversation_event *ev = conversation_event_at(c, i - 1);
        if (ev->type == CONVERSATION_EVENT_TOOL_STARTED &&
            ev->call_id == call_id)
            return ev->tool;
    }
    return NULL;
}

wire_status wire_history_at(wire *w, uint64_t position,
                            wire_history_entry *out) {
    if (!w || !out) return WIRE_INVALID_ARGUMENT;
    if (!w->has_current)
        return wire_fail(w, WIRE_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;
    if (position >= conversation_visible_count(c))
        return wire_fail(w, WIRE_INVALID_ARGUMENT,
                         "history position out of range");
    uint64_t index = conversation_visible_index(c, position);
    const conversation_event *ev = conversation_event_at(c, index);
    memset(out, 0, sizeof *out);
    out->marker = index;
    if (ev->block_count) {
        out->text = ev->blocks[0].data;
        out->text_length = ev->blocks[0].length;
    }
    if (ev->block_count > 1) {
        out->extra_json = ev->blocks[1].data;
        out->extra_length = ev->blocks[1].length;
    }
    switch (ev->type) {
    case CONVERSATION_EVENT_MESSAGE:
        out->role = ev->role;
        out->kind = ev->role == CONVERSATION_ROLE_SYSTEM
                    ? WIRE_MESSAGE_SYSTEM
                    : ev->role == CONVERSATION_ROLE_ASSISTANT
                    ? WIRE_MESSAGE_ASSISTANT : WIRE_MESSAGE_USER;
        break;
    case CONVERSATION_EVENT_GENERATION_RESULT:
        out->role = CONVERSATION_ROLE_ASSISTANT;
        out->kind = WIRE_MESSAGE_ASSISTANT;
        out->stop_reason = ev->stop_reason;
        break;
    case CONVERSATION_EVENT_TOOL_RESULT:
        out->role = CONVERSATION_ROLE_TOOL;
        out->kind = WIRE_MESSAGE_TOOL_RESULT;
        out->call_id = ev->call_id;
        out->tool_status = ev->tool_status;
        out->tool_name = wire_tool_name_of(c, ev->call_id);
        break;
    default:
        return wire_fail(w, WIRE_INVALID_ARGUMENT, "unexpected event");
    }
    return WIRE_OK;
}

wire_status wire_append(wire *w, const wire_message *message,
                        wire_marker *out) {
    if (!w || !message) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind != WIRE_GEN_NONE)
        return wire_fail(w, WIRE_BUSY, "generation in progress");
    if (!w->has_current)
        return wire_fail(w, WIRE_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;
    uint64_t tokens;
    conversation_tokens(c, &tokens);
    profile_render render;
    wire_status status;
    conversation_block block;
    block.format = CONVERSATION_BLOCK_TEXT;
    block.data = message->text ? message->text : "";
    block.length = message->text ? strlen(message->text) : 0;

    if (message->kind == WIRE_MESSAGE_USER) {
        if (!message->text)
            return wire_fail(w, WIRE_INVALID_ARGUMENT, "missing text");
        status = wire_from_profile(
            w, profile_render_user(w->prof, message->text, wire_turn(c),
                                   &render));
        if (status != WIRE_OK) return status;
        status = wire_budget_check(w, tokens + render.token_count);
        if (status != WIRE_OK) return status;
        status = wire_from_conversation(
            w, conversation_append_message(c, CONVERSATION_ROLE_USER,
                                           &block, 1, render.render,
                                           render.render_length,
                                           render.tokens,
                                           render.token_count));
    } else if (message->kind == WIRE_MESSAGE_TOOL_RESULT) {
        if (!message->text)
            return wire_fail(w, WIRE_INVALID_ARGUMENT, "missing text");
        if (wire_turn(c) != PROFILE_TURN_OPEN)
            return wire_fail(w, WIRE_INVALID_ARGUMENT,
                             "no open model turn for a tool result");
        const char *name = wire_tool_name_of(c, message->call_id);
        if (!name)
            return wire_fail(w, WIRE_INVALID_ARGUMENT,
                             "unknown tool call id %llu",
                             (unsigned long long)message->call_id);
        status = wire_from_profile(
            w, profile_render_tool_result(w->prof, name, message->text,
                                          &render));
        if (status != WIRE_OK) return status;
        status = wire_budget_check(w, tokens + render.token_count);
        if (status != WIRE_OK) return status;
        uint32_t tool_status = message->tool_status
                               ? message->tool_status
                               : CONVERSATION_TOOL_OK;
        status = wire_from_conversation(
            w, conversation_append_tool_result(c, message->call_id,
                                               tool_status, &block, 1,
                                               render.render,
                                               render.render_length,
                                               render.tokens,
                                               render.token_count));
    } else {
        return wire_fail(w, WIRE_INVALID_ARGUMENT,
                         "unsupported message kind for append");
    }
    if (status != WIRE_OK) return status;
    status = wire_from_conversation(w, conversation_commit(c));
    if (status != WIRE_OK) return status;
    if (out) *out = conversation_event_count(c) - 1;
    return WIRE_OK;
}

static void wire_settings_defaults(conversation_settings *settings) {
    settings->temperature = 1.0f;
    settings->top_k = 64;
    settings->top_p = 0.95f;
    settings->max_tokens = 0;
    settings->sampler_abi = CONVERSATION_SAMPLER_ABI;
    settings->rng_seed = 1;
    settings->rng_state = 1;
}

static int wire_settings_apply(conversation_settings *settings,
                               const wire_gen_params *params) {
    int changed = 0;
    if (!params) return 0;
    if (params->temperature >= 0.0f &&
        params->temperature != settings->temperature) {
        settings->temperature = params->temperature;
        changed = 1;
    }
    if (params->top_k >= 0 && params->top_k != settings->top_k) {
        settings->top_k = params->top_k;
        changed = 1;
    }
    if (params->top_p > 0.0f && params->top_p != settings->top_p) {
        settings->top_p = params->top_p;
        changed = 1;
    }
    if (params->max_tokens >= 0 &&
        params->max_tokens != settings->max_tokens) {
        settings->max_tokens = params->max_tokens;
        changed = 1;
    }
    if (params->rng_seed) {
        settings->rng_seed = params->rng_seed;
        settings->rng_state = params->rng_seed;
        changed = 1;
    }
    return changed;
}

static wire_status wire_prompt_reserve(wire *w) {
    if (w->prompt) return WIRE_OK;
    w->prompt = malloc((size_t)w->context * sizeof(*w->prompt));
    if (!w->prompt) return wire_fail(w, WIRE_NOMEM, "out of memory");
    return WIRE_OK;
}

static void wire_gen_reset(wire *w) {
    w->cancel_requested = 0;
    w->started_emitted = 0;
    w->first_sync_done = 0;
    w->sampled_length = 0;
    memset(&w->usage, 0, sizeof w->usage);
    json_writer_reset(&w->content);
    json_writer_reset(&w->render);
    wire_calls_reset(w);
    profile_parser_reset(w->prof);
}

static wire_status wire_best_snapshot(conversation *c, uint64_t limit,
                                      kvstore_id *id, uint64_t *boundary) {
    uint64_t best = 0;
    int found = 0;
    uint64_t epoch = conversation_epoch_current(c);
    uint64_t count = conversation_event_count(c);
    for (uint64_t i = 0; i < count; i++) {
        const conversation_event *ev = conversation_event_at(c, i);
        if (ev->type != CONVERSATION_EVENT_SNAPSHOT_REF) continue;
        if (ev->epoch != epoch) continue;
        if (ev->snapshot_boundary > limit) continue;
        if (ev->snapshot_boundary < best) continue;
        best = ev->snapshot_boundary;
        *id = ev->snapshot;
        found = 1;
    }
    if (!found) return WIRE_SESSION_NOT_FOUND;
    *boundary = best;
    return WIRE_OK;
}

static wire_status wire_gen_prepare(wire *w, xe_session *session,
                                    conversation *c) {
    w->gen_session = session;
    w->autosave_attempted = 0;
    w->resume_attempted = 0;
    xe_tokens prompt = { w->prompt, w->prompt_length, w->context };
    int common = xe_session_common(session, &prompt);
    if (c && w->kv) {
        kvstore_id id;
        uint64_t boundary = 0;
        if (wire_best_snapshot(c, (uint64_t)w->prompt_length, &id,
                               &boundary) == WIRE_OK &&
            boundary > (uint64_t)common) {
            xe_tokens expected = { w->prompt, (int)boundary,
                                   (int)boundary };
            xe_snapshot_status snapshot_status;
            kvstore_status ks = kvstore_load(w->kv, &id, session, &expected,
                                             &snapshot_status);
            w->resume_attempted = 1;
            memset(&w->resume, 0, sizeof w->resume);
            w->resume.tokens = boundary;
            if (ks == KVSTORE_OK) {
                common = xe_session_common(session, &prompt);
                w->resume.loaded = 1;
                w->resume.reason = WIRE_RESUME_LOADED;
            } else if (ks == KVSTORE_MISS) {
                w->resume.reason = WIRE_RESUME_EVICTED;
            } else if (ks == KVSTORE_REJECTED) {
                switch (snapshot_status) {
                case XE_SNAPSHOT_MODEL_MISMATCH:
                    w->resume.reason = WIRE_RESUME_MODEL_MISMATCH; break;
                case XE_SNAPSHOT_TOKEN_MISMATCH:
                    w->resume.reason = WIRE_RESUME_TOKEN_MISMATCH; break;
                case XE_SNAPSHOT_IO:
                    w->resume.reason = WIRE_RESUME_IO; break;
                default:
                    w->resume.reason = WIRE_RESUME_REJECTED; break;
                }
            } else {
                w->resume.reason = WIRE_RESUME_IO;
            }
        }
    }
    w->prompt_common = common;
    w->prompt_synced = common;
    w->gen_phase = WIRE_PHASE_PREFILL;
    return WIRE_OK;
}

wire_status wire_generate(wire *w, const wire_gen_params *params) {
    if (!w) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind != WIRE_GEN_NONE)
        return wire_fail(w, WIRE_BUSY, "generation in progress");
    if (!w->has_current)
        return wire_fail(w, WIRE_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;

    conversation_settings settings;
    int stored = conversation_get_settings(c, &settings);
    if (!stored) wire_settings_defaults(&settings);
    if (settings.sampler_abi != CONVERSATION_SAMPLER_ABI)
        return wire_fail(w, WIRE_INVALID_ARGUMENT,
                         "sampler abi mismatch");
    int changed = wire_settings_apply(&settings, params);
    if (!stored || changed) {
        wire_status status = wire_from_conversation(
            w, conversation_append_settings(c, &settings));
        if (status != WIRE_OK) return status;
    }

    uint64_t tokens;
    const int32_t *projection = conversation_tokens(c, &tokens);
    profile_render render;
    wire_status status = wire_from_profile(
        w, profile_render_reply_open(w->prof, wire_turn(c), &render));
    if (status != WIRE_OK) return status;
    status = wire_budget_check(w, tokens + render.token_count);
    if (status != WIRE_OK) return status;
    status = wire_prompt_reserve(w);
    if (status != WIRE_OK) return status;
    if (tokens) memcpy(w->prompt, projection,
                       (size_t)tokens * sizeof(*w->prompt));
    if (render.token_count)
        memcpy(w->prompt + tokens, render.tokens,
               (size_t)render.token_count * sizeof(*w->prompt));
    w->framing_offset = (int)tokens;
    w->prompt_length = (int)(tokens + render.token_count);

    w->generation_id = conversation_event_count(c);
    conversation_generation generation = { w->generation_id, settings };
    status = wire_from_conversation(
        w, conversation_append_generation_started(c, &generation));
    if (status == WIRE_OK)
        status = wire_from_conversation(w, conversation_commit(c));
    if (status != WIRE_OK) return status;

    wire_gen_reset(w);
    if (json_rawn(&w->render, render.render,
                  render.render_length) == 0)
        return wire_fail(w, WIRE_NOMEM, "out of memory");
    w->gen_settings = settings;
    w->sampler.temperature = settings.temperature;
    w->sampler.top_k = settings.top_k;
    w->sampler.top_p = settings.top_p;
    w->sampler.rng_state = settings.rng_state;
    w->gen_max_tokens = settings.max_tokens;
    if (!w->session) w->session = xe_session_new(w->engine);
    w->gen_kind = WIRE_GEN_DURABLE;
    return wire_gen_prepare(w, w->session, c);
}

wire_status wire_ephemeral_generate(wire *w, const char *system,
                                    const profile_tool *tools,
                                    size_t tool_count,
                                    const wire_message *messages,
                                    size_t count,
                                    const wire_gen_params *params) {
    if (!w) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind != WIRE_GEN_NONE)
        return wire_fail(w, WIRE_BUSY, "generation in progress");
    wire_status status = wire_prompt_reserve(w);
    if (status != WIRE_OK) return status;

    profile_render render;
    status = wire_from_profile(
        w, profile_render_system(w->prof, system, tools, tool_count,
                                 &render));
    if (status != WIRE_OK) return status;
    int length = 0;
    if ((int)render.token_count >= w->context)
        return wire_budget_check(w, render.token_count);
    memcpy(w->prompt, render.tokens,
           (size_t)render.token_count * sizeof(*w->prompt));
    length = (int)render.token_count;

    uint32_t turn = PROFILE_TURN_PADDED;
    for (size_t i = 0; i < count; i++) {
        const wire_message *m = &messages[i];
        switch (m->kind) {
        case WIRE_MESSAGE_USER:
            if (!m->text)
                return wire_fail(w, WIRE_INVALID_ARGUMENT, "missing text");
            status = wire_from_profile(
                w, profile_render_user(w->prof, m->text, turn, &render));
            turn = PROFILE_TURN_PADDED;
            break;
        case WIRE_MESSAGE_ASSISTANT:
            status = wire_from_profile(
                w, profile_render_assistant(w->prof, m->text, m->calls,
                                            m->call_count, turn, &render));
            turn = m->call_count ? PROFILE_TURN_OPEN : PROFILE_TURN_BARE;
            break;
        case WIRE_MESSAGE_TOOL_RESULT:
            if (turn != PROFILE_TURN_OPEN)
                return wire_fail(w, WIRE_INVALID_ARGUMENT,
                                 "no open model turn for a tool result");
            if (!m->tool_name || !m->text)
                return wire_fail(w, WIRE_INVALID_ARGUMENT,
                                 "missing tool name or text");
            status = wire_from_profile(
                w, profile_render_tool_result(w->prof, m->tool_name,
                                              m->text, &render));
            break;
        default:
            return wire_fail(w, WIRE_INVALID_ARGUMENT,
                             "unsupported message kind");
        }
        if (status != WIRE_OK) return status;
        status = wire_budget_check(w,
                                   (uint64_t)length + render.token_count);
        if (status != WIRE_OK) return status;
        memcpy(w->prompt + length, render.tokens,
               (size_t)render.token_count * sizeof(*w->prompt));
        length += (int)render.token_count;
    }
    status = wire_from_profile(
        w, profile_render_reply_open(w->prof, turn, &render));
    if (status != WIRE_OK) return status;
    status = wire_budget_check(w, (uint64_t)length + render.token_count);
    if (status != WIRE_OK) return status;
    memcpy(w->prompt + length, render.tokens,
           (size_t)render.token_count * sizeof(*w->prompt));
    length += (int)render.token_count;

    w->framing_offset = length;
    w->prompt_length = length;
    wire_gen_reset(w);
    conversation_settings settings;
    wire_settings_defaults(&settings);
    wire_settings_apply(&settings, params);
    w->gen_settings = settings;
    w->sampler.temperature = settings.temperature;
    w->sampler.top_k = settings.top_k;
    w->sampler.top_p = settings.top_p;
    w->sampler.rng_state = settings.rng_state;
    w->gen_max_tokens = settings.max_tokens;
    if (!w->ephemeral) w->ephemeral = xe_session_new(w->engine);
    w->gen_kind = WIRE_GEN_EPHEMERAL;
    return wire_gen_prepare(w, w->ephemeral, NULL);
}

wire_status wire_cancel(wire *w) {
    if (!w) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind == WIRE_GEN_NONE)
        return wire_fail(w, WIRE_INVALID_ARGUMENT,
                         "no generation in progress");
    w->cancel_requested = 1;
    return WIRE_OK;
}

static wire_status wire_call_register(wire *w, const char *name) {
    if (w->call_count == w->call_capacity) {
        size_t capacity = w->call_capacity ? w->call_capacity * 2 : 4;
        wire_gen_call *grown = realloc(w->calls,
                                       capacity * sizeof(*grown));
        if (!grown) return wire_fail(w, WIRE_NOMEM, "out of memory");
        w->calls = grown;
        w->call_capacity = capacity;
    }
    wire_gen_call *call = &w->calls[w->call_count];
    call->id = w->next_call_id++;
    call->name = strdup(name);
    call->arguments = NULL;
    call->complete = 0;
    if (!call->name) return wire_fail(w, WIRE_NOMEM, "out of memory");
    w->call_open = (int)w->call_count;
    w->call_count++;
    return WIRE_OK;
}

static int wire_render_token(wire *w, int32_t token) {
    char buffer[512];
    int bytes = xe_detokenize(w->engine, token, buffer, (int)sizeof buffer);
    if (bytes > 0) return json_rawn(&w->render, buffer, (size_t)bytes);
    int piece_length = 0;
    const char *piece = xe_token_piece(w->engine, token, &piece_length);
    if (piece && piece_length > 0)
        return json_rawn(&w->render, piece, (size_t)piece_length);
    return 1;
}

static wire_status wire_calls_json(wire *w) {
    json_writer_reset(&w->calls_scratch);
    json_raw(&w->calls_scratch, "[");
    for (size_t i = 0; i < w->call_count; i++) {
        if (i) json_raw(&w->calls_scratch, ",");
        json_raw(&w->calls_scratch, "{\"id\":");
        json_u64(&w->calls_scratch, w->calls[i].id);
        json_raw(&w->calls_scratch, ",\"name\":");
        json_string(&w->calls_scratch, w->calls[i].name,
                    strlen(w->calls[i].name));
        json_raw(&w->calls_scratch, ",\"arguments\":");
        json_raw(&w->calls_scratch, w->calls[i].arguments
                                    ? w->calls[i].arguments : "{}");
        json_raw(&w->calls_scratch, "}");
    }
    json_raw(&w->calls_scratch, "]");
    if (w->calls_scratch.failed)
        return wire_fail(w, WIRE_NOMEM, "out of memory");
    return WIRE_OK;
}

static void wire_gen_teardown(wire *w) {
    if (w->gen_kind == WIRE_GEN_EPHEMERAL && w->ephemeral) {
        xe_session_free(w->ephemeral);
        w->ephemeral = NULL;
    }
    w->gen_kind = WIRE_GEN_NONE;
    w->gen_session = NULL;
}

static wire_status wire_gen_error(wire *w, wire_status status,
                                  wire_event *out) {
    wire_gen_teardown(w);
    memset(out, 0, sizeof *out);
    out->kind = WIRE_EVENT_ERROR;
    out->error = status;
    out->error_text = w->error_text;
    return WIRE_OK;
}

static wire_status wire_finalize(wire *w, uint32_t wire_stop_reason,
                                 uint32_t record_stop, wire_event *out) {
    if (w->call_open >= 0 && !w->calls[w->call_open].complete) {
        free(w->calls[w->call_count - 1].name);
        free(w->calls[w->call_count - 1].arguments);
        w->call_count--;
        w->call_open = -1;
    }
    w->usage.output = (uint64_t)w->sampled_length;
    wire_marker marker = WIRE_MARKER_NONE;

    if (w->gen_kind == WIRE_GEN_DURABLE) {
        conversation *c = w->current;
        conversation_block blocks[2];
        uint32_t block_count = 1;
        blocks[0].format = CONVERSATION_BLOCK_TEXT;
        blocks[0].data = w->content.data ? w->content.data : "";
        blocks[0].length = w->content.length;
        if (w->call_count) {
            wire_status status = wire_calls_json(w);
            if (status != WIRE_OK) return wire_gen_error(w, status, out);
            blocks[1].format = CONVERSATION_BLOCK_JSON;
            blocks[1].data = w->calls_scratch.data;
            blocks[1].length = w->calls_scratch.length;
            block_count = 2;
        }
        const int32_t *event_tokens = w->prompt + w->framing_offset;
        uint32_t event_count = (uint32_t)(w->prompt_length -
                                          w->framing_offset +
                                          w->sampled_length);
        wire_status status = wire_from_conversation(
            w, conversation_append_generation_result(
                   c, w->generation_id, record_stop, w->sampler.rng_state,
                   blocks, block_count, w->render.data ? w->render.data : "",
                   w->render.length, event_tokens, event_count));
        if (status != WIRE_OK) return wire_gen_error(w, status, out);
        marker = conversation_event_count(c) - 1;
        for (size_t i = 0; i < w->call_count; i++) {
            conversation_tool_call call;
            memset(&call, 0, sizeof call);
            call.call_id = w->calls[i].id;
            call.server = "";
            call.tool = w->calls[i].name;
            call.arguments = (const uint8_t *)(w->calls[i].arguments
                                               ? w->calls[i].arguments
                                               : "{}");
            call.arguments_length = strlen((const char *)call.arguments);
            format_sha256 hasher;
            format_sha256_init(&hasher);
            format_sha256_update(&hasher, call.tool, strlen(call.tool));
            format_sha256_update(&hasher, call.arguments,
                                 call.arguments_length);
            format_sha256_final(&hasher, call.fingerprint);
            status = wire_from_conversation(
                w, conversation_append_tool_started(c, &call));
            if (status != WIRE_OK) return wire_gen_error(w, status, out);
        }
        status = wire_from_conversation(w, conversation_commit(c));
        if (status != WIRE_OK) return wire_gen_error(w, status, out);
        uint64_t total;
        conversation_tokens(c, &total);
        w->usage.total = total;
        if (w->kv && !w->ckpt_autosave_off &&
            conversation_autosave_due(total, w->saved_tokens,
                                      w->saved_at, wire_now())) {
            w->autosave_attempted = 1;
            wire_checkpoint_now(w, &w->autosave);
        }
    } else {
        w->usage.total = (uint64_t)(w->prompt_length + w->sampled_length);
    }

    wire_usage usage = w->usage;
    wire_gen_teardown(w);
    memset(out, 0, sizeof *out);
    out->kind = WIRE_EVENT_DONE;
    out->stop = wire_stop_reason;
    out->usage = usage;
    out->marker = marker;
    out->checkpoint_attempted = w->autosave_attempted;
    out->checkpoint = w->autosave;
    out->resume_attempted = w->resume_attempted;
    out->resume = w->resume;
    return WIRE_OK;
}

wire_status wire_next_event(wire *w, wire_event *out) {
    if (!w || !out) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind == WIRE_GEN_NONE)
        return wire_fail(w, WIRE_INVALID_ARGUMENT,
                         "no generation in progress");
    memset(out, 0, sizeof *out);

    if (!w->started_emitted) {
        w->started_emitted = 1;
        out->kind = WIRE_EVENT_START;
        return WIRE_OK;
    }

    for (;;) {
        if (w->cancel_requested)
            return wire_finalize(w, WIRE_STOP_ABORTED,
                                 CONVERSATION_STOP_CANCELLED, out);

        if (w->gen_phase == WIRE_PHASE_PREFILL) {
            int target = w->prompt_synced + WIRE_PREFILL_CHUNK;
            if (target > w->prompt_length) target = w->prompt_length;
            if (!w->first_sync_done || w->prompt_synced < w->prompt_length) {
                xe_tokens prefix = { w->prompt, target, w->context };
                xe_sync_report report;
                xe_session_sync_report(w->gen_session, &prefix, &report);
                if (!w->first_sync_done) {
                    w->usage.cache_read = (uint64_t)report.reused;
                    w->first_sync_done = 1;
                }
                w->usage.input += (uint64_t)report.prefilled;
                w->prompt_synced = target;
                if (w->prompt_synced >= w->prompt_length)
                    w->gen_phase = WIRE_PHASE_DECODE;
                if (w->usage.input > 0 &&
                    (uint64_t)w->prompt_length > w->usage.cache_read) {
                    out->kind = WIRE_EVENT_PROGRESS;
                    out->prefilled = w->usage.input;
                    out->prefill_total = (uint64_t)w->prompt_length -
                                         w->usage.cache_read;
                    return WIRE_OK;
                }
                continue;
            }
            w->gen_phase = WIRE_PHASE_DECODE;
            continue;
        }

        if (w->gen_max_tokens > 0 &&
            w->sampled_length >= w->gen_max_tokens)
            return wire_finalize(w, WIRE_STOP_LENGTH,
                                 CONVERSATION_STOP_LIMIT, out);
        if (w->prompt_length + w->sampled_length >= w->context - 1)
            return wire_finalize(w, WIRE_STOP_LENGTH,
                                 CONVERSATION_STOP_LIMIT, out);

        int32_t token = xe_session_next(w->gen_session, &w->sampler);
        profile_parse_event event;
        profile_status parsed = profile_parser_feed(w->prof, token, &event);
        if (parsed != PROFILE_OK)
            return wire_gen_error(
                w, wire_fail(w, WIRE_NOMEM, "out of memory"), out);

        if (event.kind == PROFILE_PARSE_STOP) {
            if (event.include_token) {
                w->prompt[w->prompt_length + w->sampled_length] = token;
                w->sampled_length++;
                if (!wire_render_token(w, token))
                    return wire_gen_error(
                        w, wire_fail(w, WIRE_NOMEM, "out of memory"), out);
            }
            switch (event.stop_reason) {
            case PROFILE_STOP_TOOL_CALLS:
                if (w->call_count)
                    return wire_finalize(w, WIRE_STOP_TOOL_USE,
                                         CONVERSATION_STOP_TOOL_CALLS, out);
                return wire_finalize(w, WIRE_STOP_STOP,
                                     CONVERSATION_STOP_TOOL_CALLS, out);
            case PROFILE_STOP_EOS:
                return wire_finalize(w, WIRE_STOP_STOP,
                                     CONVERSATION_STOP_EOS, out);
            default:
                return wire_finalize(w, WIRE_STOP_STOP,
                                     CONVERSATION_STOP_EOT_SAMPLED, out);
            }
        }

        w->prompt[w->prompt_length + w->sampled_length] = token;
        w->sampled_length++;
        if (!wire_render_token(w, token))
            return wire_gen_error(
                w, wire_fail(w, WIRE_NOMEM, "out of memory"), out);
        xe_tokens prefix = { w->prompt,
                             w->prompt_length + w->sampled_length,
                             w->context };
        xe_session_sync(w->gen_session, &prefix);

        switch (event.kind) {
        case PROFILE_PARSE_TEXT:
            if (!json_rawn(&w->content, event.text, event.text_length))
                return wire_gen_error(
                    w, wire_fail(w, WIRE_NOMEM, "out of memory"), out);
            out->kind = WIRE_EVENT_TEXT_DELTA;
            out->text = event.text;
            out->text_length = event.text_length;
            return WIRE_OK;
        case PROFILE_PARSE_CALL_START: {
            wire_status status = wire_call_register(w, event.call_name);
            if (status != WIRE_OK) return wire_gen_error(w, status, out);
            out->kind = WIRE_EVENT_TOOLCALL_START;
            out->call_id = w->calls[w->call_count - 1].id;
            out->call_name = w->calls[w->call_count - 1].name;
            return WIRE_OK;
        }
        case PROFILE_PARSE_CALL_END: {
            if (w->call_open < 0)
                return wire_gen_error(
                    w, wire_fail(w, WIRE_INVALID_ARGUMENT,
                                 "call end without start"), out);
            wire_gen_call *call = &w->calls[w->call_open];
            call->arguments = strdup(event.arguments_json
                                     ? event.arguments_json : "{}");
            call->complete = 1;
            w->call_open = -1;
            if (!call->arguments)
                return wire_gen_error(
                    w, wire_fail(w, WIRE_NOMEM, "out of memory"), out);
            out->kind = WIRE_EVENT_TOOLCALL_END;
            out->call_id = call->id;
            out->call_name = call->name;
            out->arguments_json = call->arguments;
            return WIRE_OK;
        }
        default:
            continue;
        }
    }
}

wire_status wire_rewind(wire *w, wire_marker marker) {
    if (!w) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind != WIRE_GEN_NONE)
        return wire_fail(w, WIRE_BUSY, "generation in progress");
    if (!w->has_current)
        return wire_fail(w, WIRE_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;
    uint64_t position;
    if (!conversation_visible_position(c, marker, &position))
        return wire_fail(w, WIRE_MARKER_UNAVAILABLE,
                         "marker %llu is not addressable",
                         (unsigned long long)marker);
    if (position + 1 == conversation_visible_count(c)) return WIRE_OK;
    wire_status status = wire_from_conversation(
        w, conversation_append_rewind(c, marker));
    if (status != WIRE_OK) return status;
    status = wire_from_conversation(w, conversation_commit(c));
    if (status != WIRE_OK) return status;
    kvstore_id snapshot;
    uint64_t boundary = 0;
    w->saved_tokens = conversation_snapshot_current(c, &snapshot, &boundary)
                      ? boundary : 0;
    wire_ckpt_reset(w);
    return WIRE_OK;
}

wire_status wire_rewind_cost(wire *w, wire_marker marker,
                             uint64_t *prefill_tokens) {
    if (!w || !prefill_tokens) return WIRE_INVALID_ARGUMENT;
    if (!w->has_current)
        return wire_fail(w, WIRE_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;
    uint64_t position;
    if (!conversation_visible_position(c, marker, &position))
        return wire_fail(w, WIRE_MARKER_UNAVAILABLE,
                         "marker %llu is not addressable",
                         (unsigned long long)marker);
    uint64_t boundary = conversation_visible_boundary(c, position);
    if (w->session && boundary) {
        uint64_t total;
        const int32_t *tokens = conversation_tokens(c, &total);
        if (boundary <= total) {
            xe_tokens prefix = { (int32_t *)tokens, (int)boundary,
                                 (int)boundary };
            int common = xe_session_common(w->session, &prefix);
            int frontier = xe_session_position(w->session);
            if ((uint64_t)common == boundary &&
                (uint64_t)frontier >= boundary &&
                (uint64_t)frontier - boundary <= 1024) {
                *prefill_tokens = 0;
                return WIRE_OK;
            }
        }
    }
    kvstore_id id;
    uint64_t snapshot_boundary = 0;
    if (wire_best_snapshot(c, boundary, &id, &snapshot_boundary)
            == WIRE_OK) {
        *prefill_tokens = boundary - snapshot_boundary;
        return WIRE_OK;
    }
    *prefill_tokens = boundary;
    return WIRE_OK;
}

wire_status wire_rebuild(wire *w, const char *system,
                         const profile_tool *tools, size_t tool_count,
                         const wire_message *messages, size_t count,
                         wire_marker *out) {
    if (!w) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind != WIRE_GEN_NONE)
        return wire_fail(w, WIRE_BUSY, "generation in progress");
    if (!w->has_current)
        return wire_fail(w, WIRE_SESSION_NOT_FOUND, "no open session");
    conversation *c = w->current;
    wire_status status = wire_from_conversation(
        w, conversation_append_rewind(c, CONVERSATION_REWIND_ALL));
    if (status != WIRE_OK) return status;
    status = wire_from_conversation(w, conversation_append_cache_epoch(c));
    if (status != WIRE_OK) return status;
    status = wire_append_system_event(w, c, system, tools, tool_count,
                                      NULL);
    if (status != WIRE_OK) return status;

    uint32_t turn = PROFILE_TURN_PADDED;
    profile_render render;
    for (size_t i = 0; i < count; i++) {
        const wire_message *m = &messages[i];
        uint64_t tokens;
        conversation_tokens(c, &tokens);
        conversation_block blocks[2];
        uint32_t block_count = 1;
        blocks[0].format = CONVERSATION_BLOCK_TEXT;
        blocks[0].data = m->text ? m->text : "";
        blocks[0].length = m->text ? strlen(m->text) : 0;
        switch (m->kind) {
        case WIRE_MESSAGE_USER:
            if (!m->text)
                return wire_fail(w, WIRE_INVALID_ARGUMENT, "missing text");
            status = wire_from_profile(
                w, profile_render_user(w->prof, m->text, turn, &render));
            if (status != WIRE_OK) return status;
            status = wire_budget_check(w, tokens + render.token_count);
            if (status != WIRE_OK) return status;
            status = wire_from_conversation(
                w, conversation_append_message(c, CONVERSATION_ROLE_USER,
                                               blocks, 1, render.render,
                                               render.render_length,
                                               render.tokens,
                                               render.token_count));
            if (status != WIRE_OK) return status;
            turn = PROFILE_TURN_PADDED;
            break;
        case WIRE_MESSAGE_ASSISTANT: {
            status = wire_from_profile(
                w, profile_render_assistant(w->prof, m->text, m->calls,
                                            m->call_count, turn, &render));
            if (status != WIRE_OK) return status;
            status = wire_budget_check(w, tokens + render.token_count);
            if (status != WIRE_OK) return status;
            wire_calls_reset(w);
            for (size_t j = 0; j < m->call_count; j++) {
                status = wire_call_register(w, m->calls[j].name);
                if (status != WIRE_OK) return status;
                w->calls[j].arguments = strdup(
                    m->calls[j].arguments_json
                    ? m->calls[j].arguments_json : "{}");
                w->calls[j].complete = 1;
                if (!w->calls[j].arguments)
                    return wire_fail(w, WIRE_NOMEM, "out of memory");
            }
            w->call_open = -1;
            if (m->call_count) {
                status = wire_calls_json(w);
                if (status != WIRE_OK) return status;
                blocks[1].format = CONVERSATION_BLOCK_JSON;
                blocks[1].data = w->calls_scratch.data;
                blocks[1].length = w->calls_scratch.length;
                block_count = 2;
            }
            status = wire_from_conversation(
                w, conversation_append_message(
                       c, CONVERSATION_ROLE_ASSISTANT, blocks, block_count,
                       render.render, render.render_length, render.tokens,
                       render.token_count));
            if (status != WIRE_OK) return status;
            for (size_t j = 0; j < w->call_count; j++) {
                conversation_tool_call call;
                memset(&call, 0, sizeof call);
                call.call_id = w->calls[j].id;
                call.server = "";
                call.tool = w->calls[j].name;
                call.arguments = (const uint8_t *)w->calls[j].arguments;
                call.arguments_length = strlen(w->calls[j].arguments);
                format_sha256 hasher;
                format_sha256_init(&hasher);
                format_sha256_update(&hasher, call.tool,
                                     strlen(call.tool));
                format_sha256_update(&hasher, call.arguments,
                                     call.arguments_length);
                format_sha256_final(&hasher, call.fingerprint);
                status = wire_from_conversation(
                    w, conversation_append_tool_started(c, &call));
                if (status != WIRE_OK) return status;
            }
            turn = m->call_count ? PROFILE_TURN_OPEN : PROFILE_TURN_BARE;
            break;
        }
        case WIRE_MESSAGE_TOOL_RESULT: {
            if (turn != PROFILE_TURN_OPEN)
                return wire_fail(w, WIRE_INVALID_ARGUMENT,
                                 "no open model turn for a tool result");
            if (!m->text)
                return wire_fail(w, WIRE_INVALID_ARGUMENT, "missing text");
            uint64_t pending[64];
            size_t pending_count = conversation_unknown_tool_calls(
                c, pending, sizeof pending / sizeof pending[0]);
            if (!pending_count)
                return wire_fail(w, WIRE_INVALID_ARGUMENT,
                                 "no pending tool call");
            uint64_t call_id = pending[0];
            const char *name = wire_tool_name_of(c, call_id);
            if (m->tool_name) {
                size_t k = 0;
                for (; k < pending_count; k++) {
                    const char *candidate = wire_tool_name_of(c,
                                                              pending[k]);
                    if (candidate && strcmp(candidate,
                                            m->tool_name) == 0) {
                        call_id = pending[k];
                        name = candidate;
                        break;
                    }
                }
                if (k == pending_count)
                    return wire_fail(w, WIRE_INVALID_ARGUMENT,
                                     "no pending call for tool %s",
                                     m->tool_name);
            }
            if (!name)
                return wire_fail(w, WIRE_INVALID_ARGUMENT,
                                 "unknown tool call");
            status = wire_from_profile(
                w, profile_render_tool_result(w->prof, name, m->text,
                                              &render));
            if (status != WIRE_OK) return status;
            status = wire_budget_check(w, tokens + render.token_count);
            if (status != WIRE_OK) return status;
            uint32_t tool_status = m->tool_status ? m->tool_status
                                                  : CONVERSATION_TOOL_OK;
            status = wire_from_conversation(
                w, conversation_append_tool_result(
                       c, call_id, tool_status, blocks, 1, render.render,
                       render.render_length, render.tokens,
                       render.token_count));
            if (status != WIRE_OK) return status;
            break;
        }
        default:
            return wire_fail(w, WIRE_INVALID_ARGUMENT,
                             "unsupported message kind");
        }
    }
    status = wire_from_conversation(w, conversation_commit(c));
    if (status != WIRE_OK) return status;
    w->saved_tokens = 0;
    w->saved_at = 0;
    wire_ckpt_reset(w);
    if (out) *out = conversation_event_count(c) - 1;
    return WIRE_OK;
}

wire_status wire_checkpoint(wire *w, wire_checkpoint_report *out) {
    if (!w) return WIRE_INVALID_ARGUMENT;
    if (w->gen_kind != WIRE_GEN_NONE)
        return wire_fail(w, WIRE_BUSY, "generation in progress");
    if (!w->has_current)
        return wire_fail(w, WIRE_SESSION_NOT_FOUND, "no open session");
    /* Explicit: bypass the autosave backoff and re-arm it on success. */
    w->ckpt_autosave_off = 0;
    int64_t now = wire_now();
    if (w->saved_at > now) w->saved_at = now;
    wire_checkpoint_now(w, out);
    return WIRE_OK;
}
