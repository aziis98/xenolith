#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "../conversation.h"
#include "../format.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *conversation_fault_point;

int conversation_fault(const char *point) {
    return conversation_fault_point &&
           strcmp(conversation_fault_point, point) == 0;
}

static void remove_tree(const char *path) {
    DIR *directory = opendir(path);
    if (directory) {
        struct dirent *entry;
        while ((entry = readdir(directory)) != NULL) {
            if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, ".."))
                continue;
            char child[1024];
            int n = snprintf(child, sizeof child, "%s/%s", path,
                             entry->d_name);
            if (n <= 0 || (size_t)n >= sizeof child) continue;
            struct stat st;
            if (lstat(child, &st) == 0 && S_ISDIR(st.st_mode))
                remove_tree(child);
            else
                unlink(child);
        }
        closedir(directory);
    }
    rmdir(path);
}

static void id_hex(const conversation_id *id, char out[33]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++) {
        out[2 * i] = hex[id->bytes[i] >> 4];
        out[2 * i + 1] = hex[id->bytes[i] & 15];
    }
    out[32] = '\0';
}

static void record_path(const char *root, const conversation_id *id,
                        char out[1280]) {
    char hex[33];
    id_hex(id, hex);
    snprintf(out, 1280, "%s/sessions/%s/record.xcr", root, hex);
}

static int copy_record(const char *from_root, const char *to_root,
                       const conversation_id *id, int64_t truncate_to) {
    char source[1280], target[1280], hex[33];
    record_path(from_root, id, source);
    id_hex(id, hex);
    snprintf(target, sizeof target, "%s/sessions/%s", to_root, hex);
    char component[1280];
    snprintf(component, sizeof component, "%s/sessions", to_root);
    mkdir(to_root, 0700);
    mkdir(component, 0700);
    mkdir(target, 0700);
    snprintf(target, sizeof target, "%s/sessions/%s/record.xcr", to_root,
             hex);
    FILE *in = fopen(source, "rb");
    FILE *out = fopen(target, "wb");
    if (!in || !out) {
        if (in) fclose(in);
        if (out) fclose(out);
        return 0;
    }
    uint8_t buffer[4096];
    size_t n;
    int64_t remaining = truncate_to;
    while ((n = fread(buffer, 1, sizeof buffer, in)) > 0) {
        size_t take = n;
        if (truncate_to >= 0) {
            if (remaining <= 0) break;
            if ((int64_t)take > remaining) take = (size_t)remaining;
            remaining -= (int64_t)take;
        }
        if (fwrite(buffer, 1, take, out) != take) {
            fclose(in);
            fclose(out);
            return 0;
        }
    }
    fclose(in);
    return fclose(out) == 0;
}

static int flip_byte(const char *root, const conversation_id *id,
                     int64_t offset) {
    char path[1280];
    record_path(root, id, path);
    FILE *file = fopen(path, "r+b");
    if (!file) return 0;
    if (offset < 0) {
        fseek(file, 0, SEEK_END);
        offset += ftell(file);
    }
    int ok = fseek(file, (long)offset, SEEK_SET) == 0;
    int c = fgetc(file);
    ok = ok && c != EOF && fseek(file, (long)offset, SEEK_SET) == 0 &&
         fputc(c ^ 1, file) != EOF;
    return fclose(file) == 0 && ok;
}

static int64_t record_size(const char *root, const conversation_id *id) {
    char path[1280];
    record_path(root, id, path);
    struct stat st;
    return stat(path, &st) == 0 ? (int64_t)st.st_size : -1;
}

static int check_events_equal(const conversation_event *a,
                              const conversation_event *b) {
    if (a->type != b->type || a->role != b->role ||
        a->block_count != b->block_count ||
        a->render_length != b->render_length ||
        a->token_count != b->token_count) return 0;
    if (memcmp(a->render, b->render, (size_t)a->render_length) != 0) return 0;
    if (memcmp(a->tokens, b->tokens,
               (size_t)a->token_count * sizeof(*a->tokens)) != 0) return 0;
    for (uint32_t i = 0; i < a->block_count; i++) {
        if (a->blocks[i].format != b->blocks[i].format ||
            a->blocks[i].length != b->blocks[i].length ||
            memcmp(a->blocks[i].data, b->blocks[i].data,
                   (size_t)a->blocks[i].length) != 0) return 0;
    }
    return 1;
}

static int test_basic(const char *root) {
    conversation_store *store = NULL;
    int ok = conversation_store_open(&store, root) == CONVERSATION_OK;
    conversation *c = NULL;
    conversation_id id;
    ok &= conversation_create(store, &c, &id) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;

    ok &= conversation_append_title(c, "prova \xc3\xa8 unicode") ==
          CONVERSATION_OK;
    ok &= conversation_append_workspace(c, "/workspace/project") ==
          CONVERSATION_OK;
    conversation_settings settings = { 0.7f, 64, 0.95f, 512,
                                       CONVERSATION_SAMPLER_ABI, 42, 42 };
    ok &= conversation_append_settings(c, &settings) == CONVERSATION_OK;

    conversation_block blocks[2] = {
        { CONVERSATION_BLOCK_TEXT, "ciao mondo", 10 },
        { CONVERSATION_BLOCK_BINARY, "\x00\x01\x02\xff", 4 }
    };
    int32_t user_tokens[5] = { 2, 105, 4640, 236, 108 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, blocks, 2,
                                      "<user>ciao mondo</user>", 23,
                                      user_tokens, 5) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;

    conversation_generation generation = { 1, settings };
    ok &= conversation_append_generation_started(c, &generation) ==
          CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    ok &= conversation_generation_interrupted(c) == 1;
    int32_t reply_tokens[3] = { 900, 901, 106 };
    conversation_block reply_block = { CONVERSATION_BLOCK_TEXT, "ecco", 4 };
    ok &= conversation_append_generation_result(
              c, 1, CONVERSATION_STOP_EOT_SAMPLED, 777, &reply_block, 1,
              "ecco<eot>", 9, reply_tokens, 3) == CONVERSATION_OK;
    ok &= conversation_generation_interrupted(c) == 0;
    ok &= conversation_commit(c) == CONVERSATION_OK;

    uint64_t count = 0;
    const int32_t *tokens = conversation_tokens(c, &count);
    ok &= count == 8 && tokens[0] == 2 && tokens[5] == 900 &&
          tokens[7] == 106;
    conversation_settings loaded;
    ok &= conversation_get_settings(c, &loaded) && loaded.rng_state == 777 &&
          loaded.rng_seed == 42;
    uint64_t events = conversation_event_count(c);

    conversation_close(c);
    conversation *reopened = NULL;
    ok &= conversation_open(store, &id, &reopened) == CONVERSATION_OK;
    ok &= memcmp(conversation_get_id(reopened)->bytes, id.bytes, 16) == 0;
    ok &= conversation_event_count(reopened) == events;
    ok &= strcmp(conversation_get_title(reopened), "prova \xc3\xa8 unicode")
          == 0;
    ok &= strcmp(conversation_get_workspace(reopened), "/workspace/project") == 0;
    const int32_t *replayed = conversation_tokens(reopened, &count);
    ok &= count == 8 &&
          memcmp(replayed, tokens ? tokens : replayed, 0) == 0;
    ok &= replayed[0] == 2 && replayed[4] == 108 && replayed[5] == 900 &&
          replayed[6] == 901 && replayed[7] == 106;
    ok &= conversation_get_settings(reopened, &loaded) &&
          loaded.rng_state == 777;
    const conversation_event *message = conversation_event_at(reopened, 3);
    ok &= message && message->type == CONVERSATION_EVENT_MESSAGE &&
          message->role == CONVERSATION_ROLE_USER &&
          message->block_count == 2 &&
          message->blocks[1].length == 4 &&
          memcmp(message->blocks[1].data, "\x00\x01\x02\xff", 4) == 0 &&
          message->render_length == 23 &&
          memcmp(message->render, "<user>ciao mondo</user>", 23) == 0;
    const conversation_event *result = conversation_event_at(reopened, 5);
    ok &= result && result->type == CONVERSATION_EVENT_GENERATION_RESULT &&
          result->stop_reason == CONVERSATION_STOP_EOT_SAMPLED &&
          result->rng_after == 777 && result->token_count == 3;
    ok &= conversation_updated(reopened) >= conversation_created(reopened);

    conversation *locked = NULL;
    ok &= conversation_open(store, &id, &locked) == CONVERSATION_LOCKED;

    conversation_close(reopened);
    conversation_store_close(store);
    printf("conversation: create append commit reopen replay lock %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static int test_replay_equality(const char *root, const char *copy_root) {
    conversation_store *store = NULL;
    int ok = conversation_store_open(&store, root) == CONVERSATION_OK;
    conversation *c = NULL;
    conversation_id id;
    ok &= conversation_create(store, &c, &id) == CONVERSATION_OK;
    conversation_block block = { CONVERSATION_BLOCK_JSON,
                                 "{\"k\":\"v\"}", 9 };
    int32_t tokens[2] = { 11, 12 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_SYSTEM, &block, 1,
                                      "sys", 3, tokens, 2) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;

    ok &= copy_record(root, copy_root, &id, -1);
    conversation_store *copy_store = NULL;
    ok &= conversation_store_open(&copy_store, copy_root) == CONVERSATION_OK;
    conversation *copy = NULL;
    ok &= conversation_open(copy_store, &id, &copy) == CONVERSATION_OK;
    ok &= conversation_event_count(copy) == conversation_event_count(c);
    for (uint64_t i = 0; ok && i < conversation_event_count(c); i++)
        ok &= check_events_equal(conversation_event_at(c, i),
                                 conversation_event_at(copy, i));
    conversation_close(copy);
    conversation_store_close(copy_store);
    conversation_close(c);
    conversation_store_close(store);
    printf("conversation: byte-exact replay across copies %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static int test_crash_tails(const char *root, const char *copy_root) {
    conversation_store *store = NULL;
    int ok = conversation_store_open(&store, root) == CONVERSATION_OK;
    conversation *c = NULL;
    conversation_id id;
    ok &= conversation_create(store, &c, &id) == CONVERSATION_OK;
    int32_t tokens[2] = { 5, 6 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "a", 1, tokens, 2) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    int64_t committed_size = record_size(root, &id);
    int32_t more[1] = { 7 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "b", 1, more, 1) == CONVERSATION_OK;
    int64_t uncommitted_size = record_size(root, &id);
    ok &= uncommitted_size > committed_size;

    for (int64_t cut = committed_size; ok && cut <= uncommitted_size;
         cut += (cut == committed_size) ? 7 : 11) {
        remove_tree(copy_root);
        ok &= copy_record(root, copy_root, &id, cut);
        conversation_store *copy_store = NULL;
        ok &= conversation_store_open(&copy_store, copy_root) ==
              CONVERSATION_OK;
        conversation *copy = NULL;
        ok &= conversation_open(copy_store, &id, &copy) == CONVERSATION_OK;
        uint64_t count = 0;
        conversation_tokens(copy, &count);
        ok &= count == 2;
        ok &= conversation_uncommitted(copy) == 0;
        int32_t after[1] = { 9 };
        ok &= conversation_append_message(copy, CONVERSATION_ROLE_USER, NULL,
                                          0, "c", 1, after, 1) ==
              CONVERSATION_OK;
        ok &= conversation_commit(copy) == CONVERSATION_OK;
        conversation_close(copy);
        copy = NULL;
        ok &= conversation_open(copy_store, &id, &copy) == CONVERSATION_OK;
        conversation_tokens(copy, &count);
        ok &= count == 3;
        conversation_close(copy);
        conversation_store_close(copy_store);
    }
    conversation_close(c);
    conversation_store_close(store);
    printf("conversation: torn tails resume at last commit %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static int test_damage(const char *root, const char *copy_root) {
    conversation_store *store = NULL;
    int ok = conversation_store_open(&store, root) == CONVERSATION_OK;
    conversation *c = NULL;
    conversation_id id;
    ok &= conversation_create(store, &c, &id) == CONVERSATION_OK;
    int32_t tokens[2] = { 5, 6 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "aaaa", 4, tokens, 2) ==
          CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    int64_t first_commit = record_size(root, &id);
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "bbbb", 4, tokens, 2) ==
          CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    conversation_close(c);

    remove_tree(copy_root);
    ok &= copy_record(root, copy_root, &id, -1);
    ok &= flip_byte(copy_root, &id, 64 + 40);
    conversation_store *copy_store = NULL;
    ok &= conversation_store_open(&copy_store, copy_root) == CONVERSATION_OK;
    conversation *copy = NULL;
    ok &= conversation_open(copy_store, &id, &copy) == CONVERSATION_DAMAGED;
    ok &= copy == NULL;
    conversation_summary *summaries = NULL;
    size_t count = 0;
    ok &= conversation_list(copy_store, &summaries, &count) ==
          CONVERSATION_OK;
    ok &= count == 1 && summaries[0].resumable == 0;
    free(summaries);
    conversation_store_close(copy_store);

    remove_tree(copy_root);
    ok &= copy_record(root, copy_root, &id, -1);
    ok &= flip_byte(copy_root, &id, first_commit + 8);
    copy_store = NULL;
    ok &= conversation_store_open(&copy_store, copy_root) == CONVERSATION_OK;
    copy = NULL;
    ok &= conversation_open(copy_store, &id, &copy) == CONVERSATION_DAMAGED;
    conversation_store_close(copy_store);

    remove_tree(copy_root);
    ok &= copy_record(root, copy_root, &id, -1);
    ok &= flip_byte(copy_root, &id, -3);
    copy_store = NULL;
    ok &= conversation_store_open(&copy_store, copy_root) == CONVERSATION_OK;
    copy = NULL;
    ok &= conversation_open(copy_store, &id, &copy) == CONVERSATION_OK;
    uint64_t token_count = 0;
    conversation_tokens(copy, &token_count);
    ok &= token_count == 2;
    conversation_close(copy);
    conversation_store_close(copy_store);
    conversation_store_close(store);
    printf("conversation: committed damage vs final torn commit %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static int test_embedded_frame_tail(const char *root, const char *copy_root) {
    conversation_store *store = NULL;
    int ok = conversation_store_open(&store, root) == CONVERSATION_OK;
    conversation *c = NULL;
    conversation_id id;
    ok &= conversation_create(store, &c, &id) == CONVERSATION_OK;
    int32_t tokens[1] = { 1 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "m", 1, tokens, 1) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    int64_t committed = record_size(root, &id);
    ok &= committed > 64;

    char path[1280];
    record_path(root, &id, path);
    size_t image_length = (size_t)(committed - 64);
    uint8_t *image = malloc(image_length);
    ok &= image != NULL;
    FILE *file = fopen(path, "rb");
    ok &= file && fseek(file, 64, SEEK_SET) == 0 &&
          fread(image, 1, image_length, file) == image_length;
    if (file) fclose(file);

    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      image, image_length, tokens, 1) ==
          CONVERSATION_OK;
    int64_t uncommitted = record_size(root, &id);
    free(image);

    remove_tree(copy_root);
    ok &= copy_record(root, copy_root, &id, uncommitted - 5);
    conversation_store *copy_store = NULL;
    ok &= conversation_store_open(&copy_store, copy_root) == CONVERSATION_OK;
    conversation *copy = NULL;
    ok &= conversation_open(copy_store, &id, &copy) == CONVERSATION_OK;
    uint64_t count = 0;
    conversation_tokens(copy, &count);
    ok &= count == 1;
    conversation_close(copy);
    conversation_store_close(copy_store);
    conversation_close(c);
    conversation_store_close(store);
    printf("conversation: embedded frame image stays torn tail %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static int test_tools(const char *root, const char *copy_root) {
    conversation_store *store = NULL;
    int ok = conversation_store_open(&store, root) == CONVERSATION_OK;
    conversation *c = NULL;
    conversation_id id;
    ok &= conversation_create(store, &c, &id) == CONVERSATION_OK;
    conversation_tool_call call;
    memset(&call, 0, sizeof call);
    call.call_id = 9;
    call.server = "native";
    call.tool = "read_file";
    call.arguments = (const uint8_t *)"{\"path\":\"/etc/hostname\"}";
    call.arguments_length = 24;
    memset(call.fingerprint, 0xab, 32);
    ok &= conversation_append_tool_started(c, &call) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;

    remove_tree(copy_root);
    ok &= copy_record(root, copy_root, &id, -1);
    conversation_store *copy_store = NULL;
    ok &= conversation_store_open(&copy_store, copy_root) == CONVERSATION_OK;
    conversation *crashed = NULL;
    ok &= conversation_open(copy_store, &id, &crashed) == CONVERSATION_OK;
    uint64_t pending[4];
    ok &= conversation_unknown_tool_calls(crashed, pending, 4) == 1 &&
          pending[0] == 9;
    const conversation_event *started = conversation_event_at(crashed, 0);
    ok &= started && started->type == CONVERSATION_EVENT_TOOL_STARTED &&
          strcmp(started->server, "native") == 0 &&
          strcmp(started->tool, "read_file") == 0 &&
          started->arguments_length == 24 &&
          started->fingerprint[0] == 0xab;
    conversation_close(crashed);
    conversation_store_close(copy_store);

    int32_t tool_tokens[2] = { 40, 41 };
    ok &= conversation_append_tool_result(c, 9, CONVERSATION_TOOL_OK, NULL, 0,
                                          "out", 3, tool_tokens, 2) ==
          CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    ok &= conversation_unknown_tool_calls(c, NULL, 0) == 0;
    ok &= conversation_append_tool_result(c, 9, CONVERSATION_TOOL_OK, NULL, 0,
                                          "again", 5, NULL, 0) ==
          CONVERSATION_FORMAT;
    ok &= conversation_append_tool_result(c, 77, CONVERSATION_TOOL_OK, NULL,
                                          0, "ghost", 5, NULL, 0) ==
          CONVERSATION_FORMAT;
    int32_t next[1] = { 50 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "next", 4, next, 1) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    conversation_close(c);
    conversation *reopened = NULL;
    ok &= conversation_open(store, &id, &reopened) == CONVERSATION_OK;
    uint64_t count = 0;
    conversation_tokens(reopened, &count);
    ok &= count == 3;
    ok &= conversation_unknown_tool_calls(reopened, NULL, 0) == 0;
    conversation_close(reopened);
    conversation_store_close(store);
    printf("conversation: tool lifecycle outcome-unknown no-repeat %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static int test_snapshot_epoch(const char *root) {
    conversation_store *store = NULL;
    int ok = conversation_store_open(&store, root) == CONVERSATION_OK;
    conversation *c = NULL;
    conversation_id id;
    ok &= conversation_create(store, &c, &id) == CONVERSATION_OK;
    int32_t tokens[4] = { 1, 2, 3, 4 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "m", 1, tokens, 4) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;

    kvstore_id snapshot;
    memset(snapshot.bytes, 0x5a, 16);
    ok &= conversation_append_snapshot_ref(c, &snapshot, 9) ==
          CONVERSATION_FORMAT;
    ok &= conversation_append_snapshot_ref(c, &snapshot, 4) ==
          CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    kvstore_id current;
    uint64_t boundary = 0;
    ok &= conversation_snapshot_current(c, &current, &boundary) &&
          boundary == 4 && memcmp(current.bytes, snapshot.bytes, 16) == 0;
    ok &= conversation_epoch_current(c) == 0;

    ok &= conversation_append_cache_epoch(c) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    ok &= conversation_epoch_current(c) == 1;
    ok &= conversation_snapshot_current(c, NULL, NULL) == 0;

    kvstore_id second;
    memset(second.bytes, 0x66, 16);
    ok &= conversation_append_snapshot_ref(c, &second, 2) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    conversation_close(c);
    conversation *reopened = NULL;
    ok &= conversation_open(store, &id, &reopened) == CONVERSATION_OK;
    ok &= conversation_epoch_current(reopened) == 1;
    ok &= conversation_snapshot_current(reopened, &current, &boundary) &&
          boundary == 2 && memcmp(current.bytes, second.bytes, 16) == 0;
    conversation_close(reopened);
    conversation_store_close(store);
    printf("conversation: snapshot refs and cache epoch strip %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static int test_list_delete(const char *root) {
    conversation_store *store = NULL;
    int ok = conversation_store_open(&store, root) == CONVERSATION_OK;
    conversation_summary *summaries = NULL;
    size_t count = 0;
    conversation *a = NULL, *b = NULL;
    conversation_id id_a, id_b;
    ok &= conversation_create(store, &a, &id_a) == CONVERSATION_OK;
    ok &= conversation_append_title(a, "prima") == CONVERSATION_OK;
    ok &= conversation_commit(a) == CONVERSATION_OK;
    ok &= conversation_create(store, &b, &id_b) == CONVERSATION_OK;
    ok &= conversation_commit(b) == CONVERSATION_OK;
    ok &= conversation_list(store, &summaries, &count) == CONVERSATION_OK;
    ok &= count == 2;
    int seen_a = 0, seen_b = 0;
    for (size_t i = 0; i < count; i++) {
        if (!memcmp(summaries[i].id.bytes, id_a.bytes, 16)) {
            seen_a = 1;
            ok &= strcmp(summaries[i].title, "prima") == 0 &&
                  summaries[i].resumable == 1;
        }
        if (!memcmp(summaries[i].id.bytes, id_b.bytes, 16)) seen_b = 1;
    }
    ok &= seen_a && seen_b;
    free(summaries);
    conversation_close(a);
    conversation_close(b);
    ok &= conversation_delete(store, &id_a) == CONVERSATION_OK;
    ok &= conversation_delete(store, &id_a) == CONVERSATION_MISS;
    ok &= conversation_list(store, &summaries, &count) == CONVERSATION_OK;
    ok &= count == 1 && memcmp(summaries[0].id.bytes, id_b.bytes, 16) == 0;
    free(summaries);
    conversation *gone = NULL;
    ok &= conversation_open(store, &id_a, &gone) == CONVERSATION_MISS;
    conversation_store_close(store);
    printf("conversation: list and trash delete %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static int test_rejections(const char *root) {
    conversation_store *store = NULL;
    int ok = conversation_store_open(&store, root) == CONVERSATION_OK;
    conversation *c = NULL;
    conversation_id id;
    ok &= conversation_create(store, &c, &id) == CONVERSATION_OK;
    char *huge = malloc(CONVERSATION_TITLE_MAX + 2);
    ok &= huge != NULL;
    if (huge) {
        memset(huge, 'x', CONVERSATION_TITLE_MAX + 1);
        huge[CONVERSATION_TITLE_MAX + 1] = '\0';
        ok &= conversation_append_title(c, huge) == CONVERSATION_LIMIT;
        free(huge);
    }
    ok &= conversation_append_title(c, "bad \xff utf8") ==
          CONVERSATION_LIMIT;
    ok &= conversation_append_workspace(c, "relative/path") ==
          CONVERSATION_LIMIT;
    int32_t bad_token[1] = { 262144 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "x", 1, bad_token, 1) ==
          CONVERSATION_INVALID_ARGUMENT;
    int64_t before = record_size(root, &id);

    conversation_fault_point = "frame-written";
    int32_t good[1] = { 3 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "x", 1, good, 1) == CONVERSATION_IO;
    conversation_fault_point = NULL;
    ok &= record_size(root, &id) == before;
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "x", 1, good, 1) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    conversation_close(c);
    conversation *reopened = NULL;
    ok &= conversation_open(store, &id, &reopened) == CONVERSATION_OK;
    uint64_t count = 0;
    conversation_tokens(reopened, &count);
    ok &= count == 1;

    conversation_fault_point = "commit-synced";
    int32_t extra[1] = { 4 };
    ok &= conversation_append_message(reopened, CONVERSATION_ROLE_USER, NULL,
                                      0, "y", 1, extra, 1) ==
          CONVERSATION_OK;
    ok &= conversation_commit(reopened) == CONVERSATION_IO;
    conversation_fault_point = NULL;
    ok &= conversation_append_message(reopened, CONVERSATION_ROLE_USER, NULL,
                                      0, "z", 1, extra, 1) ==
          CONVERSATION_INVALID_ARGUMENT;
    conversation_close(reopened);
    conversation_store_close(store);
    printf("conversation: rejections and fault injection %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static int test_state_dir(void) {
    char out[512];
    int ok = setenv("XDG_STATE_HOME", "/custom/state", 1) == 0;
    ok &= conversation_default_state_dir(out, sizeof out) &&
          strcmp(out, "/custom/state/xenolith") == 0;
    ok &= unsetenv("XDG_STATE_HOME") == 0;
    const char *home = getenv("HOME");
    if (home && *home == '/') {
        char expect[512];
        snprintf(expect, sizeof expect, "%s/.local/state/xenolith", home);
        ok &= conversation_default_state_dir(out, sizeof out) &&
              strcmp(out, expect) == 0;
    }
    ok &= conversation_default_state_dir(out, 4) == 0;
    printf("conversation: default state dir %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static int test_autosave_policy(void) {
    int64_t second = INT64_C(1000000000);
    int ok = conversation_autosave_due(600, 0, 0, 31 * second) == 1;
    ok &= conversation_autosave_due(500, 0, 0, 31 * second) == 0;
    ok &= conversation_autosave_due(600, 0, 0, 29 * second) == 0;
    ok &= conversation_autosave_due(4096, 4000, 0, 31 * second) == 0;
    ok &= conversation_autosave_due(5024, 4096, 0, 31 * second) == 0;
    ok &= conversation_autosave_due(5120, 4096, 0, 31 * second) == 1;
    ok &= conversation_autosave_due(5120, 4096, 20 * second,
                                    45 * second) == 0;
    ok &= conversation_autosave_due(100, 200, 0, 31 * second) == 0;
    printf("conversation: autosave policy %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

static int test_unknown_events(const char *root, const char *copy_root) {
    conversation_store *store = NULL;
    int ok = conversation_store_open(&store, root) == CONVERSATION_OK;
    conversation *c = NULL;
    conversation_id id;
    ok &= conversation_create(store, &c, &id) == CONVERSATION_OK;
    int32_t tokens[1] = { 8 };
    ok &= conversation_append_message(c, CONVERSATION_ROLE_USER, NULL, 0,
                                      "m", 1, tokens, 1) == CONVERSATION_OK;
    ok &= conversation_commit(c) == CONVERSATION_OK;
    conversation_close(c);

    for (int committed = 0; ok && committed <= 1; committed++) {
        remove_tree(copy_root);
        ok &= copy_record(root, copy_root, &id, -1);
        char path[1280];
        record_path(copy_root, &id, path);
        FILE *file = fopen(path, "r+b");
        ok &= file != NULL;
        if (!file) break;
        fseek(file, 0, SEEK_END);
        long size = ftell(file);
        uint8_t old_commit[152];
        ok &= size > 152 && fseek(file, size - 152, SEEK_SET) == 0 &&
              fread(old_commit, 1, sizeof old_commit, file) ==
              sizeof old_commit;
        fseek(file, 0, SEEK_END);
        uint8_t payload[8] = { 0 };
        uint8_t frame[32];
        uint8_t seed[24];
        memcpy(seed, id.bytes, 16);
        format_put_u64le(seed + 16, (uint64_t)size);
        format_put_u32le(frame, 99);
        format_put_u32le(frame + 4, 1);
        format_put_u64le(frame + 8, format_get_u64le(old_commit + 8));
        format_put_u64le(frame + 16, sizeof payload);
        uint64_t crc = format_crc64(0, seed, sizeof seed);
        crc = format_crc64(crc, frame, 24);
        crc = format_crc64(crc, payload, sizeof payload);
        format_put_u64le(frame + 24, crc);
        ok &= fwrite(frame, 1, sizeof frame, file) == sizeof frame;
        ok &= fwrite(payload, 1, sizeof payload, file) == sizeof payload;
        if (committed) {
            format_sha256 hasher;
            uint8_t chain_new[32];
            format_sha256_init(&hasher);
            format_sha256_update(&hasher, old_commit + 152 - 32, 32);
            format_sha256_update(&hasher, frame, sizeof frame);
            format_sha256_update(&hasher, payload, sizeof payload);
            format_sha256_final(&hasher, chain_new);
            uint8_t commit[152];
            memcpy(commit, old_commit, sizeof commit);
            memcpy(commit + 152 - 32, chain_new, 32);
            format_put_u64le(seed + 16, (uint64_t)size + 40);
            uint64_t commit_crc = format_crc64(0, seed, sizeof seed);
            commit_crc = format_crc64(commit_crc, commit, 24);
            commit_crc = format_crc64(commit_crc, commit + 32, 120);
            format_put_u64le(commit + 24, commit_crc);
            ok &= fwrite(commit, 1, sizeof commit, file) == sizeof commit;
        }
        ok &= fclose(file) == 0;
        conversation_store *copy_store = NULL;
        ok &= conversation_store_open(&copy_store, copy_root) ==
              CONVERSATION_OK;
        conversation *copy = NULL;
        conversation_status opened = conversation_open(copy_store, &id,
                                                       &copy);
        if (committed) {
            ok &= opened == CONVERSATION_VERSION && copy == NULL;
            conversation_summary *summaries = NULL;
            size_t count = 0;
            ok &= conversation_list(copy_store, &summaries, &count) ==
                  CONVERSATION_OK;
            ok &= count == 1 && summaries[0].resumable == 0;
            free(summaries);
        } else {
            ok &= opened == CONVERSATION_OK && copy != NULL;
            uint64_t count = 0;
            conversation_tokens(copy, &count);
            ok &= count == 1;
            conversation_close(copy);
        }
        conversation_store_close(copy_store);
    }
    conversation_store_close(store);
    printf("conversation: unknown critical and optional events %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

int main(void) {
    char root[] = "/tmp/xenolith-conversation-XXXXXX";
    if (!mkdtemp(root)) return 1;
    char sub[8][1024];
    for (int i = 0; i < 8; i++)
        snprintf(sub[i], sizeof sub[i], "%s/case%d", root, i);
    char copy_root[1024];
    snprintf(copy_root, sizeof copy_root, "%s/copies", root);

    int ok = test_basic(sub[0]);
    ok &= test_replay_equality(sub[1], copy_root);
    remove_tree(copy_root);
    ok &= test_crash_tails(sub[2], copy_root);
    remove_tree(copy_root);
    ok &= test_damage(sub[3], copy_root);
    remove_tree(copy_root);
    {
        char embedded_root[1040];
        snprintf(embedded_root, sizeof embedded_root, "%s/embedded", root);
        ok &= test_embedded_frame_tail(embedded_root, copy_root);
    }
    remove_tree(copy_root);
    ok &= test_tools(sub[4], copy_root);
    remove_tree(copy_root);
    ok &= test_snapshot_epoch(sub[5]);
    ok &= test_list_delete(sub[6]);
    ok &= test_rejections(sub[7]);
    ok &= test_state_dir();
    ok &= test_autosave_policy();
    char unknown_root[1024];
    snprintf(unknown_root, sizeof unknown_root, "%s/unknown", root);
    ok &= test_unknown_events(unknown_root, copy_root);
    remove_tree(root);
    return ok ? 0 : 1;
}
