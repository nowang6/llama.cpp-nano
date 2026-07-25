#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct quant_option {
    const char * name;
    llama_ftype  ftype;
};

static const std::vector<quant_option> QUANT_OPTIONS = {
    { "Q4_0",        LLAMA_FTYPE_MOSTLY_Q4_0   },
    { "Q4_1",        LLAMA_FTYPE_MOSTLY_Q4_1   },
    { "Q5_0",        LLAMA_FTYPE_MOSTLY_Q5_0   },
    { "Q5_1",        LLAMA_FTYPE_MOSTLY_Q5_1   },
    { "Q8_0",        LLAMA_FTYPE_MOSTLY_Q8_0   },
    { "Q2_K",        LLAMA_FTYPE_MOSTLY_Q2_K   },
    { "Q3_K_S",      LLAMA_FTYPE_MOSTLY_Q3_K_S },
    { "Q3_K_M",      LLAMA_FTYPE_MOSTLY_Q3_K_M },
    { "Q3_K_L",      LLAMA_FTYPE_MOSTLY_Q3_K_L },
    { "Q4_K_S",      LLAMA_FTYPE_MOSTLY_Q4_K_S },
    { "Q4_K_M",      LLAMA_FTYPE_MOSTLY_Q4_K_M },
    { "Q5_K_S",      LLAMA_FTYPE_MOSTLY_Q5_K_S },
    { "Q5_K_M",      LLAMA_FTYPE_MOSTLY_Q5_K_M },
    { "Q6_K",        LLAMA_FTYPE_MOSTLY_Q6_K   },
    { "IQ2_XXS",     LLAMA_FTYPE_MOSTLY_IQ2_XXS },
    { "IQ2_XS",      LLAMA_FTYPE_MOSTLY_IQ2_XS  },
    { "IQ2_S",       LLAMA_FTYPE_MOSTLY_IQ2_S   },
    { "IQ3_XXS",     LLAMA_FTYPE_MOSTLY_IQ3_XXS },
    { "IQ3_S",       LLAMA_FTYPE_MOSTLY_IQ3_S   },
    { "IQ3_M",       LLAMA_FTYPE_MOSTLY_IQ3_M   },
    { "IQ1_S",       LLAMA_FTYPE_MOSTLY_IQ1_S   },
    { "IQ4_NL",      LLAMA_FTYPE_MOSTLY_IQ4_NL  },
    { "IQ4_XS",      LLAMA_FTYPE_MOSTLY_IQ4_XS  },
};

static void print_usage(const char * prog) {
    fprintf(stderr, "usage: %s <input.gguf> <output.gguf> <type> [nthreads]\n\n", prog);
    fprintf(stderr, "quantization types:\n");
    for (const auto & opt : QUANT_OPTIONS) {
        fprintf(stderr, "  %-10s", opt.name);
    }
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        print_usage(argc > 0 ? argv[0] : "llama-quantize");
        return 1;
    }

    const std::string fname_inp  = argv[1];
    const std::string fname_out  = argv[2];
    const std::string quant_type = argv[3];

    llama_ftype ftype = LLAMA_FTYPE_ALL_F32;
    bool found = false;
    for (const auto & opt : QUANT_OPTIONS) {
        if (quant_type == opt.name) {
            ftype = opt.ftype;
            found = true;
            break;
        }
    }
    if (!found) {
        fprintf(stderr, "error: unknown quantization type '%s'\n", quant_type.c_str());
        print_usage(argv[0]);
        return 1;
    }

    auto params = llama_model_quantize_default_params();
    params.ftype = ftype;
    if (argc >= 5) {
        params.nthread = std::stoi(argv[4]);
    }

    fprintf(stderr, "quantizing %s → %s  type=%s  nthread=%d\n",
            fname_inp.c_str(), fname_out.c_str(), quant_type.c_str(), params.nthread);

    uint32_t ret = llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), &params);
    if (ret != 0) {
        fprintf(stderr, "quantization failed with code %u\n", ret);
        return ret;
    }

    fprintf(stderr, "done: %s\n", fname_out.c_str());
    return 0;
}
