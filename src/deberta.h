#ifndef DEBERTA_H
#define DEBERTA_H

#include "ggml.h"
#include "ggml-backend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <map>

#define KEY_FTYPE "general.file_type"
#define KEY_NAME "general.name"
#define KEY_DESCRIPTION "general.description"

#define KEY_PAD_ID "tokenizer.ggml.padding_token_id"
#define KEY_UNK_ID "tokenizer.ggml.unknown_token_id"
#define KEY_BOS_ID "tokenizer.ggml.bos_token_id"
#define KEY_EOS_ID "tokenizer.ggml.eos_token_id"
#define KEY_WORD_PREFIX "tokenizer.ggml.word_prefix"
#define KEY_SUBWORD_PREFIX "tokenizer.ggml.subword_prefix"
#define KEY_TOKEN_LIST "tokenizer.ggml.tokens"

#define DEBERTA_API __attribute__ ((visibility ("default")))

#ifdef __cplusplus
extern "C" {
#endif


typedef int32_t deberta_token;
typedef std::vector<deberta_token> deberta_tokens;
typedef std::vector<deberta_tokens> deberta_batch;
typedef std::string deberta_string;
typedef std::vector<deberta_string> deberta_strings;



struct deberta_hparams {
  int32_t n_vocab;
  int32_t n_max_tokens;
  int32_t n_embed;
  int32_t n_intermediate;
  int32_t n_head;
  int32_t n_layer;
  float_t layer_norm_eps;
};

struct deberta_layer {
  struct ggml_tensor *q_w;
  struct ggml_tensor *q_b;
  struct ggml_tensor *k_w;
  struct ggml_tensor *k_b;
  struct ggml_tensor *v_w;
  struct ggml_tensor *v_b;
  struct ggml_tensor *o_w;
  struct ggml_tensor *o_b;
  struct ggml_tensor *ln_att_w;
  struct ggml_tensor *ln_att_b;
  struct ggml_tensor *ff_i_w;
  struct ggml_tensor *ff_i_b;
  struct ggml_tensor *ff_o_w;
  struct ggml_tensor *ff_o_b;
  struct ggml_tensor *ln_out_w;
  struct ggml_tensor *ln_out_b;
};

struct lm_head {
  struct ggml_tensor *b;
  struct ggml_tensor *d_w;
  struct ggml_tensor *d_b;
  struct ggml_tensor *ln_w;
  struct ggml_tensor *ln_b;
};

struct mask_predictions {
  struct ggml_tensor *d_w;
  struct ggml_tensor *d_b;
  struct ggml_tensor *ln_w;
  struct ggml_tensor *ln_b;
  struct ggml_tensor *clf_w;
  struct ggml_tensor *clf_b;
};

struct deberta_vocab {
  deberta_token pad_id;
  deberta_token unk_id;
  deberta_token bos_id;
  deberta_token eos_id;
  int32_t max_token_length = 1024;

  std::string word_prefix;
  std::string subword_prefix;
  
  std::vector<std::string> tokens;

  std::map<std::string, deberta_token> token_to_id;

  std::map<deberta_token, std::string> id_to_token;
};


struct deberta_model {
  deberta_hparams hparams;

  struct ggml_tensor *word_embeddings;
  struct ggml_tensor *positional_embeddings;
  struct ggml_tensor *ln_e_w;
  struct ggml_tensor *ln_e_b;
  
  std::vector<deberta_layer> layers;
  struct ggml_tensor *rel_embeddings;
  struct ggml_tensor *ln_enc_w;
  struct ggml_tensor *ln_enc_b;
  // struct ggml_tensor *clf_w;
  // struct ggml_tensor *clf_b;
  
  // struct lm_head lm;
  // struct mask_predictions mask;
};


struct deberta_ctx {
  deberta_model model;
  deberta_vocab vocab;

  struct ggml_context * ctx_data = NULL;
  struct gguf_context *ctx_gguf;

  std::vector<uint8_t> buf_compute_meta;

  ggml_backend_t backend = NULL;
  ggml_backend_buffer_t params_buffer = NULL;
  ggml_backend_buffer_t compute_buffer = NULL;
  ggml_gallocr_t *compute_alloc = NULL;
};


DEBERTA_API struct deberta_ctx *deberta_load_from_file(
    const char *fname,
    bool use_cpu
    );

DEBERTA_API void deberta_tokens_debug(
    struct deberta_ctx *ctx
    );

DEBERTA_API deberta_tokens tokenizer_encode(
    struct deberta_ctx *ctx,
    std::string text
    );


#ifdef __cplusplus
}
#endif

#endif
