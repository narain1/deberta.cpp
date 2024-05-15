#include "deberta.h"
#include "ggml.h"

#include <unistd.h>
#include <stdio.h>
#include <vector>
#include <string>

struct deberta_options {
  const char *model = nullptr;
  const char *prompt = nullptr;
  int32_t n_max_tokens = 0;
  int32_t batch_size = 32;
  bool use_cpu = true;
  bool normalize = true;
  int32_t n_threads =6;
};

void deberta_print_usage(char **argv, const deberta_options &options) {
  fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -r --raw              don't normalize embeddings (default: normalize embeddings)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", options.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model GGUF path\n");
    fprintf(stderr, "  -n N, --max-tokens N  number of tokens to generate (default: max)\n");
    fprintf(stderr, "  -b BATCH_SIZE, --batch-size BATCH_SIZE\n");
    fprintf(stderr, "                        batch size to use when executing model\n");
    fprintf(stderr, "  -c, --cpu             use CPU backend (default: use CUDA if available)\n");
    fprintf(stderr, "\n");
}

int main() {
  deberta_ctx *ctx;
  std::string file_name = "src/deberta.ggml";
  {
    ctx = deberta_load_from_file(file_name.c_str(), true);
    if (ctx == nullptr) {
      fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, file_name);
      return 1;
    }

    // deberta_allocate_buffers(ctx, 4096, 1);
  }
  return 0;
}


