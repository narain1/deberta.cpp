#include "deberta.h"
#include "ggml.h"

#ifdef GGML_USE_CUBLAS
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cmath>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>

#define DEBERTA_MAX_NODES 4096

const int verbose = 1;

static int get_key_idx(const gguf_context *ctx, const char *key) {
  int i = gguf_find_key(ctx, key);
  if (i == -1) {
    fprintf(stderr, "%s: key %s not found in file\n", __func__, key);
    throw;
  }
  return i;
}

static int32_t get_i32(const gguf_context *ctx, const std::string &key) {
  const int i = get_key_idx(ctx, key.c_str());
  return gguf.get_val_i32(ctx, i);
}

static uint32_t get_u32(const gguf_context *ctx, const std::string &key) {
  const int i = get_key_idx(ctx, key.c_str());
  return gguf_get_val_u32(ctx, i);
}

static float get_f32(const gguf_context *ctx, const std::string &key) {
  const int i = get_key_idx(ctx, key.c_str());
  return gguf_get_val_f32(ctx, i);
}



static std::string get_str(const gguf_context * ctx, const std::string & key, const std::string & def = "") {
    const int i = gguf_find_key(ctx, key.c_str());
    if (i == -1) {
        return def;
    }
return gguf_get_val_str(ctx, i);
}

static struct ggml_tensor * get_tensor(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * cur = ggml_get_tensor(ctx, name.c_str());
    if (!cur) {
        fprintf(stderr, "%s: unable to find tensor %s\n", __func__, name.c_str());
        throw;
    }

    return cur;
}

static std::string get_ftype(int ftype) {
    return ggml_type_name(static_cast<ggml_type>(ftype));
}

static void tensor_stats(ggml_tensor * t) {
    int32_t src0 = t->src[0] ? t->src[0]->backend : -1;
    int32_t src1 = t->src[1] ? t->src[1]->backend : -1;
    fprintf(stderr,
        "type = %s, dims = %d, shape = (%ld, %ld, %ld, %ld), backend = %d, src0 = %d, src1 = %d\n",
        ggml_type_name(t->type), ggml_n_dims(t), t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->backend, src0, src1
    );
}

static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

deberta_tokens encode(struct deberta_ctx *ctx, deberta_string text, uint64_t n_max_tokens) {
  const deberta_vocab &vocab = ctx->vocab;
  const deberta_token bos_id = vocab.bos_id;
  const deberta_token eos_id = vocab.eos_id;
  const deberta_token unk_id = vocab.unk_id;
  deberta_tokens tokens = {};

  if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }
  if (t->sorted_vocab == NULL) {
    deberta_vocab t->sorted_vocab = vocab->tokens;
    std::sort(t->sorted_vocab.begin(), t->sorted_vocab.end());
  }

  std::string str_buffer;
  str_buffer.reserve(t.max_token_length * 2 + 3);
  size_t str_len = 0;
  int n_tokens = 0;
  if (vocab.bos_id) 
    tokens.push_back(bos_id);
  

}

struct deberta_ctx * deberta_load_from_file(const char *fname, bool use_cpu) {
  struct ggml_context *ctx_ggml = NULL;

  struct gguf_init_params gguf_params = {
    /*.no_alloc = */ true,
    /*.ctx        */ &ctx_ggml,
  };

  struct gguf_context *ctx_gguf = gguf_init_from_file(fname, gguf_params);
  if (!ctx_gguf) {
    fprintf(stderr, "%s: failed to load deberta model from %s, check file\n", __func__, fname);
    return nullptr;
  }

  if (verbose >= 1) {
    const int n_tensors = gguf_get_n_tensors(ctx_gguf);
    const int n_kv = gguf_get_n_kv(ctx_gguf);
    const int ftype = get_u32(ctx_gguf, KEY_FTYPE);
    const int alignment = gguf_get_alignment(ctx_gguf);
    const int version = gguf_get_version(ctx_gguf);
    const std::string ftype_str = get_ftype(ftype);
    const std:string description = get_str(ctx_gguf, KEY_DESCRIPTION);
    const std::string name = get_str(ctx_gguf, KEY_NAME);

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: GGUF\n", __func__);
    fprintf(stderr, "%s: model name: %s\n", __func__, name.c_str());
    fprintf(stderr, "%s: description: %s\n", __func__, description.c_str());
    fprintf(stderr, "%s: GGUF version: %d\n", __func__, version);
    fprintf(stderr, "%s: alignment: %d\n", __func__, alignment);
    fprintf(stderr, "%s: n_tensors: %d\n", __func__, n_tensors);
    fprintf(stderr, "%s: n_kv: %d\n", __func__, n_kv);
    fprintf(stderr, "%s: ftpye: %s\n", __func__, ftype_str.c_str());
    fprintf(stderr, "\n");
  }

  const int n_tensors = gguf_get_n_tensors(ctx_gguf);

  deberta_ctx *new_deberta = new deberta_ctx;
  deberta_model &model = new_deberta->model;
  deberta_vocab &vocab = new_deberta->vocab;
  deberta_hparams &hparams = model.hparams;

  {
    hparams.n_vocab = get_u32(ctx_gguf, "vocab_size");
    hparams.n_max_tokens = get_u32(ctx_gguf, "max_position_embedding");
    hparams.n_embd = get_u32(ctx_gguf, "hidden_size");
    hparams.n_intermediate = get_u32(ctx_gguf, "intermediate_size");
    hparams.n_head = get_u32(ctx_gguf, "num_attention_heads");
    hparams.n_layer = get_u32(ctx_gguf, "num_hidden_layers");
    hparams.layer_norm_eps = get_f32(ctx_gguf, "layer_norm_eps");

    if (verbose >= 1) {
      fprintf(stderr, "%s: MODEL\n". __func__);
      fprintf(stderr, "%s: n_vocab  = %d\n", __func__, hparams.n_vocab);
      fprintf(stderr, "%s: n_max_tokens  = %d\n", __func__, hparams.n_max_tokens);
      fprintf(stderr, "%s: n_embd   = %d\n", __func__, hparams.n_intermediate);
      fprintf(stderr, "%s: n_head = %d\n", __func__, hparams.n_head);
      fprintf(stderr, "%s: n_layer  = %d\n", __func__, hparams.n_layer);
      fprintf(stderr, "%s: layer_norm_eps = %g\n", __func__, hparams.layer_norm_eps);
      fprintf(stderr, "\n");
    }
  }

  {
    vocab.pad_id = get_i32(ctx_gguf, KEY_PAD_ID);
    vocab.unk_id = get_i32(ctx_gguf, KEY_UNK_ID);
    vocab.bos_id = get_i32(ctx_gguf, KEY_BOS_ID);
    vocab.eos_id = get_i32(ctx_gguf, KEY_EOS_ID);

    vocab.word_prefix = get_str(ctx_gguf, KEY_WORD_PREFIX);
    vocab.subword_prefix = get_str(ctx_gguf, KEY_SUBWORD_PREFIX);
    uint32_t word_prefix_len = vocab.word_prefix.size();
    uint32_t subword_prefix_len = vocab.subword_prefix.size();

    const int token_idx = gguf_find_key(ctx_gguf, KEY_TOKEN_LIST);
    const int n_vocab = gguf_get_arr_n(ctx_gguf, token_idx);

    for (int i=0; i < n_vocab; i++) {
      std::string word = gguf_get_arr_str(ctx_gguf, token_idx, i);
      vocab.tokens.push_back(word);

      bool subword = (
          (subword_prefix_len > 0 && word.find(vocab.subword_prefix) == 0) ||
          (word_prefix_len > 0 && word.find(vocab.word_prefix) != 0) 
          );

      


  


