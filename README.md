# llama-nano

从 [llama.cpp](https://github.com/ggerganov/llama.cpp) 精简而来的最小化 LLM 推理项目。

**仅支持 X86 CPU + Qwen3-0.6B 模型**，移除所有 GPU 后端。

项目分为两个子目录：
- **`model_converter/`** — 模型导出、格式转换（HF→GGUF），纯 Python
- **`inference/`** — 推理引擎 + 量化工具（C++ 编译运行）

# 编译

```bash
cd inference
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++
cmake --build build -j$(nproc)
```

编译产物在 `inference/build/bin/` 下：
- `llama-nano` — 推理 CLI
- `llama-quantize` — 量化工具

# 模型转换与量化

## 模型转换

将 HuggingFace 格式的模型权重转换为 GGUF 格式：

```bash
cd model_converter
python convert_hf_to_gguf.py ../models/Qwen3-0.6B --outfile ../models/model.gguf --outtype f16
```
支持的 `--outtype` 选项：`f32` / `f16` / `bf16` / `q8_0` / `tq1_0` / `tq2_0`

### 模型量化
先用 `--outtype f16` 转换得到全精度 GGUF，再用 `llama-quantize` 工具进行量化：
```bash
# 量化为 Q4_K_M（推荐，~4.8 bit，体积约 1/4）
./inference/build/bin/llama-quantize ../models/model.gguf ../models/model-Q4_K_M.gguf Q4_K_M
```

可选的量化类型：

| 类型 | 大小 vs F16 | 质量 | 说明 |
|------|------------|------|------|
| **Q4_K_M** | ~25% | ⭐⭐⭐ | 推荐首选 |
| Q4_K_S | ~23% | ⭐⭐ | 更小但略差 |
| Q4_0 | ~23% | ⭐⭐ | 基础 4-bit 对称量化 |
| Q5_K_M | ~32% | ⭐⭐⭐ | 5-bit，质量更好 |
| Q8_0 | ~50% | ⭐⭐⭐⭐ | 8-bit，损失极小 |
| IQ4_NL | ~24% | ⭐⭐⭐ | 重要性感知量化 |
| IQ4_XS | ~22% | ⭐⭐⭐ | 更小的重要性感知量化 |

完整类型列表可通过 `./inference/build/bin/llama-quantize` 查看。

# 推理

```bash
cd inference
./build/bin/llama-nano -m ../models/model-Q4_K_M.gguf -n 256 "你好"
```

参数说明：
- `-m <path>` — GGUF 模型文件路径（**必需**）
- `-n <n>` — 最大生成 token 数（默认 256）
- `-t <temp>` — 采样温度（默认 0.6，设为 0 启用贪心采样）

# 模型文件解析
```sh
#直接查看 GGUF 内容
python gguf-py/examples/reader.py ../models/model-Q4_K_M.gguf

# Dump完整信息
python gguf-py/gguf/scripts/gguf_dump.py \
  ../models/model-Q4_K_M.gguf
```


## 项目结构

```
model_converter/          # 模型转换（纯 Python）
├── convert_hf_to_gguf.py # HF → GGUF 主脚本
├── conversion/           # 模型架构转换实现
├── gguf-py/              # GGUF 格式读写库
└── pyproject.toml        # Python 项目配置

inference/               # 推理引擎 + 量化工具
├── CMakeLists.txt        # 构建配置
├── main.cpp              # 推理 CLI（~200 行）
├── quantize.cpp          # 量化工具 CLI
├── include/llama.h       # 公开 C API 头文件
├── src/                  # LLaMA 模型加载/推理/采样/量化
├── ggml/                 # 张量计算库（仅 CPU 后端）
└── cmake/                # CMake 模块

models/                   # GGUF 模型文件（共享输出目录）
```