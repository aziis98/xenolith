#ifndef XENOLITH_INTERNAL_H
#define XENOLITH_INTERNAL_H

#include "xenolith.h"

int xe_session_anchor_capture(xe_session *s);
int xe_session_anchor_restore(xe_session *s);
void xe_session_anchor_clear(xe_session *s);
int xe_session_anchor_valid(const xe_session *s);
int xe_session_token_near_top(xe_session *s, int32_t token,
                              int max_rank, float max_margin);

#endif
