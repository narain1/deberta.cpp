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

void deberta_deallocate_buffers(deberta_ctx *ctx) {
  if (ctx->compute_buffer) {
    ggml_backend_buffer_free(ctx->compute_buffer);
    ctx->compute_buffer = NULL;
  }
  if (ctx->compute_alloc) {
    ggml_gallocr_free(ctx->compute_alloc);
    ctx->compute_alloc = NULL;
  }
}

void deberta_allocate_buffers(deberta_ctx *ctx, int32_t n_max_tokens, int32_t bs) {
  deberta_deallocate_buffers(ctx);

  ctx->buf_compute_meta.resize(GGML_DEFAULT_GRAPH_SIZE * ggml_tensor_overhead() + ggml_graph_overhead());
  ctx->compute_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));

  deberta_tokens tokens(n_max_tokens);
  deberta_batch batch;
  for (int i=0; i<bs; ++i) {
    batch.push_back(tokens);
  }
  ggml_cgraph *gf = deberta_build_graph(ctx, batch, true);

  size_t compute_memory_buffer_size = ggml_allocr_alloc_graph(ctx->compute_alloc, gf);
  ggml_gallocr_free(ctx->compute_alloc);

  ctx->compute_buffer = ggml_backend_alloc_buffer(ctx->backend, compute_memory_buffer_size);
  ctx->compute_alloc = ggml_allocr_new_from_buffer(ctx->compute_buffer);

  if (verbose >= 1) {
    fprintf(stderr, "%s: compute allocated memory: %.2f MB\n\n", __func__, compute_memory_buffer_size / 1024.0 / 1024.0);
  }
}


void deberta_free(deberta_ctx *ctx) {
  ggml_free(ctx->ctx_data);
  gguf_free(ctx->ctx_gguf);

  ggml_backend_buffer_free(ctx->weights_buffer);
  ggml_backend_free(ctx->backend);
  //ggml_gallocr_free(ctx->compute_alloc);
  delete ctx;
}

struct deberta_ctx * deberta_load_from_file(const char *fname, bool use_cpu) {
  struct ggml_context *ctx_ggml = NULL;

  struct gguf_init_params gguf_params = {
    /*.no_alloc = */ true,
    /*.ctx        */ &ctx_ggml,
  };

  struct gguf_context *ctx_gguf = gguf_init_from_file(fname, gguf_params);
  if (!ctx_ggml) {
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
    const int idx_descr = get_key_idx(ctx_gguf, KEY_DESCRIPTION);
    const std::string description = gguf_get_val_str(ctx_gguf, idx_descr);
    const int idx_name = gguf_find_key(ctx_gguf, KEY_NAME);
    const std::string name = gguf_get_val_str(ctx_gguf, idx_name); 

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
    hparams.n_embed = get_u32(ctx_gguf, "hidden_size");
    hparams.n_intermediate = get_u32(ctx_gguf, "intermediate_size");
    hparams.n_head = get_u32(ctx_gguf, "num_attention_heads");
    hparams.n_layer = get_u32(ctx_gguf, "num_hidden_layers");
    hparams.layer_norm_eps = get_f32(ctx_gguf, "layer_norm_eps");

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
    vocab.pad_id = get_i32(ctx_gguf, KEY_PAD_ID);
    vocab.unk_id = get_i32(ctx_gguf, KEY_UNK_ID);
    vocab.bos_id = get_i32(ctx_gguf, KEY_BOS_ID);
    vocab.eos_id = get_i32(ctx_gguf, KEY_EOS_ID);

    const int token_idx = gguf_find_key(ctx_gguf, KEY_TOKEN_LIST);
    const int n_vocab = gguf_get_arr_n(ctx_gguf, token_idx);

    for (int i=0; i < n_vocab; i++) {
      std::string word = gguf_get_arr_str(ctx_gguf, token_idx, i);
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
      const char *name = gguf_get_tensor_name(ctx_gguf, i);
      const size_t offset = gguf_get_tensor_offset(ctx_gguf, i);
      struct ggml_tensor *cur = ggml_get_tensor(ctx_ggml, name);
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
      const char *name = gguf_get_tensor_name(ctx_gguf, i);
      struct ggml_tensor *ten = ggml_get_tensor(ctx_ggml, name);
      struct ggml_tensor *cur = ggml_dup_tensor(new_deberta->ctx_data, ten);
      ggml_set_name(cur, name);
    }

    new_deberta->weights_buffer = ggml_backend_alloc_ctx_tensors(new_deberta->ctx_data, new_deberta->backend);
    // ggml_gallocr_t *alloc = ggml_allocr_new_from_buffer(new_deberta->weights_buffer);

    for (int i=0; i < n_tensors; ++i) {
      const char *name = gguf_get_tensor_name(ctx_gguf, i);
      struct ggml_tensor *cur = ggml_get_tensor(new_deberta->ctx_data, name);
      // ggml_tallocr_alloc(alloc, cur);

      const size_t offset = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
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
  }

  {
    model.rel_embeddings = get_tensor(new_deberta->ctx_data, "deberta.encoder.rel_embeddings.weight");
    model.ln_enc_w = get_tensor(new_deberta->ctx_data, "deberta.encoder.LayerNorm.weight");
    model.ln_enc_b = get_tensor(new_deberta->ctx_data, "deberta.encoder.LayerNorm.bias");
    model.clf_w = get_tensor(new_deberta->ctx_data, "classifier.weight");
    model.clf_b = get_tensor(new_deberta->ctx_data, "classifier.bias");
  }

  ggml_free(ctx_ggml);
  gguf_free(ctx_gguf);

  return new_deberta;
}



