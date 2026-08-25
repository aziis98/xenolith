#ifndef WIRE_H
#define WIRE_H

#include "xenolith.h"
#include "kvstore.h"
#include "conversation.h"
#include "profile.h"

#include <stddef.h>
#include <stdint.h>

typedef struct wire wire;
typedef uint64_t wire_marker;

#define WIRE_MARKER_NONE UINT64_MAX

typedef enum {
    WIRE_OK,
    WIRE_CONTEXT_LENGTH_EXCEEDED,
    WIRE_SESSION_NOT_FOUND,
    WIRE_MARKER_UNAVAILABLE,
    WIRE_INVALID_ARGUMENT,
    WIRE_BUSY,
    WIRE_IO,
    WIRE_NOMEM
} wire_status;

typedef struct {
    const char *model;
    int context_window;
    int max_output;
    int kvstore;            /* 1 when the snapshot store opened */
    int reasoning;
} wire_info;

/* Why a checkpoint did not save. Checkpoints are never fatal (the
 * transcript is the source of truth and replay always works); the
 * report makes the outcome visible to the client, the log and tests. */
typedef enum {
    WIRE_CKPT_SAVED,
    WIRE_CKPT_NO_KVSTORE,    /* store never opened: permanent for the process */
    WIRE_CKPT_EMPTY,         /* nothing computed yet (no tokens, or KV not loaded) */
    WIRE_CKPT_NOTHING_NEW,   /* already saved up to the current boundary */
    WIRE_CKPT_KV_DIVERGED,   /* KV is not a prefix of the transcript (transient) */
    WIRE_CKPT_BUDGET,        /* snapshot does not fit the store budget: permanent for the session */
    WIRE_CKPT_IO,            /* store write failed */
    WIRE_CKPT_REJECTED,      /* engine refused the snapshot */
    WIRE_CKPT_RECORD_FAILED  /* snapshot written, transcript record failed */
} wire_ckpt_reason;

typedef struct {
    int saved;
    wire_ckpt_reason reason;
    uint64_t tokens;        /* boundary saved, or current size when not saved */
} wire_checkpoint_report;

/* Why a resume did not load the snapshot it was expected to. */
typedef enum {
    WIRE_RESUME_LOADED,
    WIRE_RESUME_EVICTED,        /* snapshot file gone from the store */
    WIRE_RESUME_MODEL_MISMATCH,
    WIRE_RESUME_TOKEN_MISMATCH, /* transcript no longer matches the snapshot */
    WIRE_RESUME_IO,
    WIRE_RESUME_REJECTED        /* any other engine refusal */
} wire_resume_reason;

typedef struct {
    int loaded;
    wire_resume_reason reason;
    uint64_t tokens;        /* boundary loaded or attempted */
} wire_resume_report;

const char *wire_ckpt_reason_name(wire_ckpt_reason reason);
const char *wire_resume_reason_name(wire_resume_reason reason);

typedef enum {
    WIRE_MESSAGE_USER = 1,
    WIRE_MESSAGE_TOOL_RESULT = 2,
    WIRE_MESSAGE_ASSISTANT = 3,
    WIRE_MESSAGE_SYSTEM = 4
} wire_message_kind;

typedef struct {
    uint32_t kind;
    const char *text;
    const char *reasoning;
    uint64_t call_id;
    uint32_t tool_status;
    const char *tool_name;
    const profile_call *calls;
    size_t call_count;
} wire_message;

typedef struct {
    float temperature;
    int32_t top_k;
    float top_p;
    int32_t max_tokens;
    uint64_t rng_seed;
    int reasoning_set;
    uint32_t reasoning_effort;
    int reasoning_history_set;
    uint32_t reasoning_history;
    int reasoning_budget_set;
    int32_t reasoning_budget;
} wire_gen_params;

typedef struct {
    uint64_t input;
    uint64_t cache_read;
    uint64_t output;
    uint64_t total;
    uint64_t reasoning;
    uint64_t replayed;
    uint64_t shadow_prefilled;
    uint64_t shadow_background;
    uint64_t shadow_remaining;
    uint64_t shadow_kv_bytes;
    uint64_t shadow_wait_us;
} wire_usage;

typedef enum {
    WIRE_EVENT_START = 1,
    WIRE_EVENT_PROGRESS = 2,
    WIRE_EVENT_TEXT_DELTA = 3,
    WIRE_EVENT_TOOLCALL_START = 4,
    WIRE_EVENT_TOOLCALL_END = 5,
    WIRE_EVENT_DONE = 6,
    WIRE_EVENT_ERROR = 7,
    WIRE_EVENT_REASONING_DELTA = 8
} wire_event_kind;

typedef enum {
    WIRE_STOP_STOP = 1,
    WIRE_STOP_TOOL_USE = 2,
    WIRE_STOP_LENGTH = 3,
    WIRE_STOP_ABORTED = 4
} wire_stop;

typedef struct {
    uint32_t kind;
    const uint8_t *text;
    size_t text_length;
    uint64_t call_id;
    const char *call_name;
    const char *arguments_json;
    uint64_t prefilled;
    uint64_t prefill_total;
    uint32_t stop;
    uint32_t reasoning_close;
    wire_usage usage;
    wire_marker marker;
    uint32_t error;
    const char *error_text;
    uint64_t error_tokens;     /* error: context_length_exceeded detail */
    uint64_t error_context;
    int checkpoint_attempted;          /* done: autosave ran at end of turn */
    wire_checkpoint_report checkpoint;
    int resume_attempted;              /* done: a snapshot load was tried */
    wire_resume_report resume;
} wire_event;

typedef struct {
    uint64_t token_count;
    wire_marker marker;
    int turn_open;
    int zero_prefill;
    int resume_stale;       /* a snapshot exists but behind the current boundary */
    size_t pending_calls;
} wire_open_report;

typedef struct {
    uint32_t kind;
    uint32_t role;
    const char *text;
    uint64_t text_length;
    const char *reasoning;
    uint64_t reasoning_length;
    const char *extra_json;
    uint64_t extra_length;
    uint64_t call_id;
    uint32_t tool_status;
    const char *tool_name;
    uint32_t stop_reason;
    wire_marker marker;
} wire_history_entry;

wire_status wire_open(wire **out, xe_engine *engine, const char *state_dir,
                      const char *cache_dir);
void wire_close(wire *w);

wire_status wire_describe(wire *w, wire_info *out);

wire_status wire_session_create(wire *w, const char *system,
                                const profile_tool *tools, size_t tool_count,
                                conversation_id *id, wire_marker *marker);
wire_status wire_session_open(wire *w, const conversation_id *id,
                              wire_open_report *out);
wire_status wire_session_list(wire *w, conversation_summary **out,
                              size_t *count);
wire_status wire_session_stat(wire *w, const conversation_id *id,
                              conversation_summary *out);
wire_status wire_session_delete(wire *w, const conversation_id *id);

size_t wire_pending_calls(wire *w, uint64_t *call_ids, size_t cap);
uint64_t wire_history_count(wire *w);
wire_status wire_history_at(wire *w, uint64_t position,
                            wire_history_entry *out);

wire_status wire_append(wire *w, const wire_message *message,
                        wire_marker *out);
wire_status wire_generate(wire *w, const wire_gen_params *params);
wire_status wire_ephemeral_generate(wire *w, const char *system,
                                    const profile_tool *tools,
                                    size_t tool_count,
                                    const wire_message *messages,
                                    size_t count,
                                    const wire_gen_params *params);
wire_status wire_next_event(wire *w, wire_event *out);
wire_status wire_cancel(wire *w);

wire_status wire_rewind(wire *w, wire_marker marker);
wire_status wire_rewind_cost(wire *w, wire_marker marker,
                             uint64_t *prefill_tokens);
wire_status wire_rebuild(wire *w, const char *system,
                         const profile_tool *tools, size_t tool_count,
                         const wire_message *messages, size_t count,
                         wire_marker *out);
wire_status wire_checkpoint(wire *w, wire_checkpoint_report *out);
/* Status of the snapshot store at open; KVSTORE_OK when it works. */
int wire_kvstore_open_status(const wire *w);

const char *wire_status_code(wire_status status);
const char *wire_error_text(const wire *w);
/* Structured detail of the last context_length_exceeded failure; 0 when
 * the last failure was something else. */
int wire_error_detail(const wire *w, uint64_t *tokens, uint64_t *context);

#endif
