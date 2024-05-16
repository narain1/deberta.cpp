#include "deberta.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

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
#include <iostream>
#include <regex>

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
  return gguf_get_val_i32(ctx, i);
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


void deberta_free(deberta_ctx *ctx) {
  ggml_free(ctx->ctx_data);
  gguf_free(ctx->ctx_gguf);

  ggml_backend_buffer_free(ctx->weights_buffer);
  ggml_backend_free(ctx->backend);
  // gml_gallocr_free(ctx->compute_alloc);
  delete ctx;
}

struct deberta_ctx * deberta_load_from_file(const char *fname, bool use_cpu) {
  struct ggml_context *meta = NULL;

  struct gguf_init_params gguf_params = {
    /*.no_alloc = */ true,
    /*.ctx        */ &meta,
  };

  struct gguf_context *ctx = gguf_init_from_file(fname, gguf_params);
  if (!ctx) {
    fprintf(stderr, "%s: failed to load deberta model from %s, check file\n", __func__, fname);
    return nullptr;
  }

  if (verbose >= 1) {
    const int n_tensors = gguf_get_n_tensors(ctx);
    const int n_kv = gguf_get_n_kv(ctx);
    const int ftype = get_u32(ctx, KEY_FTYPE);
    const int alignment = gguf_get_alignment(ctx);
    const int version = gguf_get_version(ctx);
    const std::string ftype_str = get_ftype(ftype);
    const int idx_descr = get_key_idx(ctx, KEY_DESCRIPTION);
    const std::string description = gguf_get_val_str(ctx, idx_descr);
    const int idx_name = gguf_find_key(ctx, KEY_NAME);
    const std::string name = gguf_get_val_str(ctx, idx_name); 

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

  const int n_tensors = gguf_get_n_tensors(ctx);

  deberta_ctx *new_deberta = new deberta_ctx;
  deberta_model &model = new_deberta->model;
  deberta_vocab &vocab = new_deberta->vocab;
  deberta_hparams &hparams = model.hparams;

  {
    hparams.n_vocab = get_u32(ctx, "vocab_size");
    hparams.n_max_tokens = get_u32(ctx, "max_position_embedding");
    hparams.n_embed = get_u32(ctx, "hidden_size");
    hparams.n_intermediate = get_u32(ctx, "intermediate_size");
    hparams.n_head = get_u32(ctx, "num_attention_heads");
    hparams.n_layer = get_u32(ctx, "num_hidden_layers");
    hparams.layer_norm_eps = get_f32(ctx, "layer_norm_eps");

    if (verbose >= 1) {
      fprintf(stderr, "%s: MODEL\n", __func__);
      fprintf(stderr, "%s: n_vocab  = %d\n", __func__, hparams.n_vocab);
      fprintf(stderr, "%s: n_max_tokens  = %d\n", __func__, hparams.n_max_tokens);
      fprintf(stderr, "%s: n_embd  = %d\n", __func__, hparams.n_embed);
      fprintf(stderr, "%s: n_intermediate   = %d\n", __func__, hparams.n_intermediate);
      fprintf(stderr, "%s: n_head = %d\n", __func__, hparams.n_head);
      fprintf(stderr, "%s: n_layer  = %d\n", __func__, hparams.n_layer);
      fprintf(stderr, "%s: layer_norm_eps = %g\n", __func__, hparams.layer_norm_eps);
      fprintf(stderr, "\n");
    }
  }

  {
    vocab.pad_id = get_i32(ctx, KEY_PAD_ID);
    vocab.unk_id = get_i32(ctx, KEY_UNK_ID);
    vocab.bos_id = get_i32(ctx, KEY_BOS_ID);
    vocab.eos_id = get_i32(ctx, KEY_EOS_ID);

    const int token_idx = gguf_find_key(ctx, KEY_TOKEN_LIST);
    const int n_vocab = gguf_get_arr_n(ctx, token_idx);

    for (int i=0; i < n_vocab; i++) {
      std::string word = gguf_get_arr_str(ctx, token_idx, i);
      vocab.tokens.push_back(word);
      vocab.token_to_id[word] = i;
      vocab.id_to_token[i] = word;
    }

    if (verbose >= 1) {
      fprintf(stderr, "%s: TOKENIZER\n", __func__);
      fprintf(stderr, "%s: vocab size: %d\n", __func__, n_vocab);
      fprintf(stderr, "%s: pad_id = %d\n", __func__, vocab.pad_id);
      fprintf(stderr, "%s: unk_id = %d\n", __func__, vocab.unk_id);
      fprintf(stderr, "%s: bos_id = %d\n", __func__, vocab.bos_id);
      fprintf(stderr, "%s: eos_id = %d\n", __func__, vocab.eos_id);
      fprintf(stderr, "\n");
    }
  }

  size_t buffer_size = 32*1024;
  {
    for (int i=0; i<n_tensors; ++i) {
      const char *name = gguf_get_tensor_name(ctx, i);
      const size_t offset = gguf_get_tensor_offset(ctx, i);
      struct ggml_tensor *cur = ggml_get_tensor(meta, name);
      size_t tensor_size = ggml_nbytes(cur);
      buffer_size += tensor_size;
      if (verbose >= 2) {
        fprintf(stderr, "%s: tensor[%d]: type = %s, n_dims = %d, name = %s, offset=%zu, type=%d\n", __func__, i,
            ggml_type_name(cur->type), ggml_n_dims(cur), cur->name, offset, cur->type);
      }
    }
  }

  if (!new_deberta->backend) {
    new_deberta->backend = ggml_backend_cpu_init();
    fprintf(stderr, "%s: using CPU backend\n", __func__);
  }

  {
    std::vector<uint8_t> read_buf;
    struct ggml_init_params ggml_params = {
      /* mem_size = */ (n_tensors + 1) * ggml_tensor_overhead(),
      /* mem_buffer = */ NULL,
      /* no_alloc = */ true,
    };

    new_deberta->ctx_data = ggml_init(ggml_params);
    if (!new_deberta->ctx_data) {
      fprintf(stderr, "%s: ggml_init() failed\n", __func__);
      delete new_deberta;
      return nullptr;
    }

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
      fprintf(stderr, "cannot open model file for loading tensors\n");
      delete new_deberta;
      return nullptr;
    }

    for (int i=0; i<n_tensors; ++i) {
      const char *name = gguf_get_tensor_name(ctx, i);
      struct ggml_tensor *ten = ggml_get_tensor(meta, name);
      struct ggml_tensor *cur = ggml_dup_tensor(new_deberta->ctx_data, ten);
      ggml_set_name(cur, name);
    }

    new_deberta->weights_buffer = ggml_backend_alloc_ctx_tensors(new_deberta->ctx_data, new_deberta->backend);
    // ggml_gallocr_t *alloc = ggml_allocr_new_from_buffer(new_deberta->weights_buffer);

    for (int i=0; i < n_tensors; ++i) {
      const char *name = gguf_get_tensor_name(ctx, i);
      struct ggml_tensor *cur = ggml_get_tensor(new_deberta->ctx_data, name);
      // ggml_tallocr_alloc(alloc, cur);

      const size_t offset = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, i);
      fin.seekg(offset, std::ios::beg);
      if (!fin) {
        fprintf(stderr, "%s: failed to seek for tensor %s\n", __func__, name);
        deberta_free(new_deberta);
        return nullptr;
      }

      int num_bytes = ggml_nbytes(cur);
      if (ggml_backend_buffer_is_host(new_deberta->weights_buffer)) {
        fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
      } else {
        read_buf.resize(num_bytes);
        fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
        ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
      }
    }

    fin.close();
  }

  {
    model.word_embeddings = get_tensor(new_deberta->ctx_data, "deberta.embeddings.word_embeddings.weight");
    model.positional_embeddings = get_tensor(new_deberta->ctx_data, "deberta.embeddings.position_embeddings.weight");
    model.ln_e_w = get_tensor(new_deberta->ctx_data, "deberta.embeddings.LayerNorm.weight");
    model.ln_e_b = get_tensor(new_deberta->ctx_data, "deberta.embeddings.LayerNorm.bias");

    model.layers.resize(hparams.n_layer);
    for (int i=0; i<hparams.n_layer; ++i) {
      deberta_layer &layer = model.layers[i];
      std::string pre = "deberta.encoder.layer." + std::to_string(i) + ".";

      layer.q_w = get_tensor(new_deberta->ctx_data, pre + "attention.self.query_proj.weight");
      layer.q_b = get_tensor(new_deberta->ctx_data, pre + "attention.self.query_proj.bias");
      layer.k_w = get_tensor(new_deberta->ctx_data, pre + "attention.self.key_proj.weight");
      layer.k_b = get_tensor(new_deberta->ctx_data, pre + "attention.self.key_proj.bias");
      layer.v_w = get_tensor(new_deberta->ctx_data, pre + "attention.self.value_proj.weight");
      layer.v_b = get_tensor(new_deberta->ctx_data, pre + "attention.self.value_proj.bias");

      layer.o_w = get_tensor(new_deberta->ctx_data, pre + "attention.output.dense.weight");
      layer.o_b = get_tensor(new_deberta->ctx_data, pre + "attention.output.dense.bias");

      layer.ln_att_w = get_tensor(new_deberta->ctx_data, pre + "attention.output.LayerNorm.weight");
      layer.ln_att_b = get_tensor(new_deberta->ctx_data, pre + "attention.output.LayerNorm.bias");

      layer.ff_i_w = get_tensor(new_deberta->ctx_data, pre + "intermediate.dense.weight");
      layer.ff_i_b = get_tensor(new_deberta->ctx_data, pre + "intermediate.dense.bias");

      layer.ff_o_w = get_tensor(new_deberta->ctx_data, pre + "output.dense.weight");
      layer.ff_o_b = get_tensor(new_deberta->ctx_data, pre + "output.dense.bias");

      layer.ln_out_w = get_tensor(new_deberta->ctx_data, pre + "output.LayerNorm.weight");
      layer.ln_out_b = get_tensor(new_deberta->ctx_data, pre + "output.LayerNorm.bias");
    }

    {
      model.rel_embeddings = get_tensor(new_deberta->ctx_data, "deberta.encoder.rel_embeddings.weight");
      model.ln_enc_w = get_tensor(new_deberta->ctx_data, "deberta.encoder.LayerNorm.weight");
      model.ln_enc_b = get_tensor(new_deberta->ctx_data, "deberta.encoder.LayerNorm.bias");
    }

    // {
    //   model.lm.b = get_tensor(new_deberta->ctx_data, "lm_predictions.lm_head.bias");
    //   model.lm.d_w = get_tensor(new_deberta->ctx_data, "lm_predictions.lm_head.dense.weight");
    //   model.lm.d_b = get_tensor(new_deberta->ctx_data, "lm_predictions.lm_head.dense.bias");
    //   model.lm.ln_w = get_tensor(new_deberta->ctx_data, "lm_predictions.lm_head.LayerNorm.weight");
    //   model.lm.ln_b = get_tensor(new_deberta->ctx_data, "lm_predictions.lm_head.LayerNorm.bias");
    // }

    // {
    //   model.mask.d_w = get_tensor(new_deberta->ctx_data, "mask_predictions.dense.weight");
    //  model.mask.d_b = get_tensor(new_deberta->ctx_data, "mask_predictions.dense.bias");
    //  model.mask.ln_w = get_tensor(new_deberta->ctx_data, "mask_predictions.LayerNorm.weight");
    //  model.mask.ln_b = get_tensor(new_deberta->ctx_data, "mask_predictions.LayerNorm.bias");
    //  model.mask.clf_w = get_tensor(new_deberta->ctx_data, "mask_predictions.classifier.weight");
    //  model.mask.clf_b = get_tensor(new_deberta->ctx_data, "mask_predictions.classifer.bias");
   // }
  }

    // comp model arch
    // model.clf_w = get_tensor(new_deberta->ctx_data, "classifier.weight");
    // model.clf_b = get_tensor(new_deberta->ctx_data, "classifier.bias");
  ggml_free(meta);
  gguf_free(ctx);

  return new_deberta;
}


void deberta_tokens_debug(struct deberta_ctx *ctx) {
  const deberta_vocab &vocab = ctx->vocab;
  std::vector<int> a = {1 ,279 ,51888 ,12629 ,265 ,10766 ,718 ,45118 ,268 ,294 ,1007 ,13190 ,33606 ,264 ,4468 ,2445 ,15293 ,2};
  for (auto i: a) {
    std::cout << vocab.tokens[i] << std::endl;
  }
}

std::string sentence_piece_normalization(const std::string text) {
  std::string normalized_text = text;
  normalized_text = std::regex_replace(normalized_text, std::regex("[\\n\\t]+"), " ");
  normalized_text = std::regex_replace(normalized_text, std::regex("[\\x00-\\x1F\\x7F]+"), "");
  normalized_text = std::regex_replace(normalized_text, std::regex("\\s+"), " ");
  if (!normalized_text.empty() && normalized_text[normalized_text.size() - 1] == ' ') {
    normalized_text.erase(normalized_text.size() - 1);
  }
  if (!normalized_text.empty() && normalized_text[0] == ' ') {
    normalized_text.erase(0, 1);
  }
  return normalized_text;
}

deberta_tokens tokenizer_encode(struct deberta_ctx *ctx, const std::string x) {
  std::string normalized = sentence_piece_normalization(x);
  // const deberta_vocab &vocab = ctx->vocab;
  const int32_t max_token_length = ctx->vocab.max_token_length;
  const std::map<deberta_token, std::string> &id_to_token = ctx->vocab.id_to_token;
  const std::map<std::string, deberta_token> &vocab = ctx->vocab.token_to_id;
  std::string processed = "▁" + std::regex_replace(normalized, std::regex(" "), "▁") + "▁";
  deberta_tokens acc = {ctx->vocab.bos_id};
  size_t start_idx = 0;
  while (start_idx < processed.size()) {
    std::string buffer = processed.substr(start_idx, 1);
    size_t match_idx = 0;
    for (size_t idx=1; idx < std::min(max_token_length, static_cast<deberta_token>(processed.size() - start_idx)); ++idx) {
      std::string temp = processed.substr(start_idx, idx);
      if (vocab.find(temp) != vocab.end()) {
        buffer = temp;
        match_idx = idx;
      }
    }
    std::cout << vocab.at(processed.substr(start_idx, match_idx)) << std::endl;
    acc.push_back(vocab.at(processed.substr(start_idx, match_idx)));
    start_idx += match_idx;
  }
  acc.back() = 2;
  return acc;
}

