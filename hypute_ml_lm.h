#ifndef HYPUTE_ML_LM_H
#define HYPUTE_ML_LM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t vocab_size;
    size_t embed_dim;
    size_t hidden_dim;
    size_t num_layers;
} HyputeLMConfig;

typedef struct HyputeLMModel HyputeLMModel;

/* ── Lifecycle ────────────────────────────────────────────────────────────── */

HyputeLMModel* hypute_lm_create(const HyputeLMConfig* cfg);
void           hypute_lm_destroy(HyputeLMModel* m);

/* ── Weights ─────────────────────────────────────────────────────────────── */

int  hypute_lm_load_weights      (HyputeLMModel* m, const char* path);
int  hypute_lm_save_weights      (HyputeLMModel* m, const char* path);
void hypute_lm_randomize_weights (HyputeLMModel* m, uint64_t seed);

/* ── Inference ───────────────────────────────────────────────────────────── */

/** Reset hidden/cell state to zero (call before each new sequence). */
void hypute_lm_reset_state(HyputeLMModel* m);

/**
 * Feed one token and get next-token logits.
 * @return logits_size on success, negative on error.
 */
int hypute_lm_step(HyputeLMModel* m,
                   uint32_t       token_id,
                   float*         logits,
                   size_t         logits_size);

/* ── Telemetry ───────────────────────────────────────────────────────────── */

double hypute_lm_last_step_ns  (const HyputeLMModel* m);
size_t hypute_lm_memory_bytes  (const HyputeLMModel* m);
size_t hypute_lm_param_count   (const HyputeLMModel* m);

#ifdef __cplusplus
}
#endif
#endif /* HYPUTE_ML_LM_H */
