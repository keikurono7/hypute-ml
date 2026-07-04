#ifndef HYPUTE_ML_H
#define HYPUTE_ML_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Model configuration. All dimensions must be set before calling
 *        hypute_ml_create(). Zero values are invalid.
 */
typedef struct {
    size_t vocab_size;      /* Embedding vocabulary (items, tokens, users) */
    size_t embedding_dim;   /* Input embedding width                       */
    size_t hidden_dim;      /* Hidden layer width                          */
    size_t num_layers;      /* Number of hidden layers                     */
    size_t output_dim;      /* Output projection width (classes/items)     */
} HyputeMLConfig;

/** Opaque model handle. */
typedef struct HyputeMLModel HyputeMLModel;

/* ── Lifecycle ─────────────────────────────────────────────────────────────── */

/** Allocate and initialise a model. Returns NULL on invalid config. */
HyputeMLModel* hypute_ml_create(const HyputeMLConfig* config);

/** Free all resources. */
void hypute_ml_destroy(HyputeMLModel* model);

/**
 * @brief Switch inference mode.
 * @param use_dense  0 = default mode, 1 = dense mode.
 *                   Use dense mode for models trained with standard backprop.
 */
void hypute_ml_set_dense_mode(HyputeMLModel* model, int use_dense);

/* ── Weights ───────────────────────────────────────────────────────────────── */

/**
 * @brief Load weights from a .hml binary file.
 * @return 0 on success, negative error code on failure.
 */
int hypute_ml_load_weights(HyputeMLModel* model, const char* path);

/**
 * @brief Save current weights to a .hml binary file.
 * @return 0 on success, negative error code on failure.
 */
int hypute_ml_save_weights(HyputeMLModel* model, const char* path);

/**
 * @brief Fill all weights with deterministic random values.
 *        Useful for benchmarking without a trained checkpoint.
 */
void hypute_ml_randomize_weights(HyputeMLModel* model, uint64_t seed);

/* ── Inference ─────────────────────────────────────────────────────────────── */

/**
 * @brief Run a forward pass.
 *
 * @param input_ids    Array of input IDs (tokens, item IDs, user IDs, …).
 * @param num_inputs   Number of input IDs (sequence / context length).
 * @param output_scores  Caller-allocated buffer for output scores.
 * @param output_count Number of scores to write (capped at output_dim).
 * @return Number of scores written, or negative on error.
 */
int hypute_ml_infer(HyputeMLModel*  model,
                    const uint64_t* input_ids,
                    size_t          num_inputs,
                    float*          output_scores,
                    size_t          output_count);

/* ── Telemetry ─────────────────────────────────────────────────────────────── */

/** Wall-clock latency of the last hypute_ml_infer() call, in nanoseconds. */
double hypute_ml_last_latency_ns(const HyputeMLModel* model);

/** Throughput implied by last latency: 1e9 / latency_ns (inferences/sec). */
double hypute_ml_throughput_ops(const HyputeMLModel* model);

/** Total model memory in bytes (weights + state + working buffers). */
size_t hypute_ml_memory_bytes(const HyputeMLModel* model);

/** Total compressed parameter count across all weight matrices. */
size_t hypute_ml_param_count(const HyputeMLModel* model);

/** Number of active compute units during the last inference. */
size_t hypute_ml_activation_count(const HyputeMLModel* model);

/** Total compute slots evaluated during the last inference. */
size_t hypute_ml_compute_slots(const HyputeMLModel* model);

#ifdef __cplusplus
}
#endif

#endif /* HYPUTE_ML_H */
