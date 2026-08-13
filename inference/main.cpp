#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>


static std::string apply_chat_template(const std::string & user_msg) {
    return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
           "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
           "<|im_start|>assistant\n";
}

int main(int argc, char ** argv) {
    // 解析命令行参数
    std::string model_path = "/home/niwang/code/llama.cpp-nano/models/model-Q4_K_M.gguf";
    std::string user_prompt = "你好";
    std::string prompt = apply_chat_template(user_prompt);
    int n_predict = 256;
    float temperature = 0.6f;

    // 加载后端 (CPU only)
    ggml_backend_load_all();

    // 加载模型
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0; // CPU only

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "%s: 错误: 无法加载模型 %s\n", __func__, model_path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 获取 BOS token
    llama_token bos_id = llama_vocab_bos(vocab);

    // 分词 prompt (add_bos=true)
    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: 错误: 分词失败\n", __func__);
        return 1;
    }

    // 初始化上下文
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt_tokens + n_predict - 1;
    ctx_params.n_batch = n_prompt_tokens;
    ctx_params.no_perf = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr, "%s: 错误: 创建上下文失败\n", __func__);
        return 1;
    }

    // 初始化采样器（temperature + top-k + top-p，Qwen3 默认值）
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);

    if (temperature <= 0.0f) {
        // temperature=0 时使用贪婪采样
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        // 标准采样：temperature → top-k → top-p → 分布采样
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(20));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    // 准备 batch
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // 主循环
    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt_tokens + n_predict; ) {
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s: 解码失败\n", __func__);
            return 1;
        }

        n_pos += batch.n_tokens;

        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: 错误: token 转换失败\n", __func__);
                return 1;
            }
            printf("%s", std::string(buf, n).c_str());
            fflush(stdout);

            batch = llama_batch_get_one(&new_token_id, 1);
            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "\n解码: %d tokens / %.2f 秒, 速度: %.2f t/s\n",
            n_decode, (t_main_end - t_main_start) / 1000000.0f,
            n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
