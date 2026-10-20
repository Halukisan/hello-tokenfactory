# 第 8 章　vLLM 快速上手与单机多卡部署

> **本章要回答的问题**
> 1. 怎么用最短路径把 vLLM 装好、跑起来，又不被 CUDA / PyTorch 版本坑死？
> 2. 离线批量推理和对外提供 API 服务，分别怎么写最简单的代码？
> 3. `vllm serve` 启动后那一长串参数，哪些是新手必须懂的？调错了会怎样？
> 4. 在内网、无外网的环境里，模型权重怎么拿、服务怎么部署？

难度 ★☆☆　|　硬件要求：单卡 24GB（如 4090 / L20）即可　|　⏭️ 已熟 Docker 部署者可跳至 8.4 节

---

## 8.1 环境准备与安装

![一条命令，起一个能对外服务的推理引擎](./assets/ch08-quickstart.png)

vLLM 的安装看似一行 `pip install`，实际踩坑大多发生在「装上了却跑不起来」。这一节先把三种安装路径的取舍讲清，再集中处理版本匹配和国内网络两个高频痛点。建议把本章实验环境固定下来，后面第 6–8 章都复用，避免每章重新折腾环境。

### 8.1.1 三种安装方式怎么选

| 方式 | 适用场景 | 优点 | 代价 |
|---|---|---|---|
| **pip 安装**（官方 wheel） | 有网、环境干净、想快速试 | 最快，一条命令 | 对系统 CUDA / 驱动版本敏感 |
| **Docker 镜像**（如 `vllm/vllm-openai`） | 生产、环境一致性要求高 | 免去环境差异，部署即一致 | 需要宿主机装好 NVIDIA Container Toolkit |
| **源码编译** | 改引擎源码、追最新未发布特性、定制 kernel | 能用上尚未发版的能力 | 编译耗时、依赖复杂，新手慎入 |

> **作者经验**：90% 的读者用 Docker 就能解决「在我机器上能跑」的问题。pip 方式适合本地开发调试，源码编译只在你要给 vLLM 提 PR 或追某个刚合入的特性时才需要。不要为了「显得专业」而一上来就源码编译——编译失败带来的挫败感不值当。

**关于 conda / venv**：无论哪种方式，强烈建议装在隔离的虚拟环境里，不要污染系统 Python。vLLM 对 PyTorch 版本敏感，和你在做的其他项目共用环境，迟早互相覆盖。

```bash
python -m venv vllm-env && source vllm-env/bin/activate   # 或 conda create -n vllm python=3.11
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 8.1.2 CUDA、驱动与 PyTorch 的版本匹配

这是新手第一个大坑。三者是**硬绑定**关系，不是「能用就行」：

- **NVIDIA 驱动**决定了你能用的最高 CUDA 版本（驱动版本号对应一个 CUDA 上限）。
- **PyTorch wheel** 是编译在**特定 CUDA 版本**之上的（如 `cu124`、`cu128`）。
- **vLLM wheel** 依赖特定 PyTorch 版本，进而间接锁定 CUDA。

一个常见失败场景：宿主机驱动只支持到 CUDA 12.1，却装了 `cu124` 版的 PyTorch，结果运行时报 `CUDA error: no kernel image available` 或直接找不到 `libcudart.so`。

**稳妥的做法**：先用 `nvidia-smi` 看右上角的「CUDA Version」，它代表驱动支持的最高 CUDA。再据此选 PyTorch / vLLM 的 wheel 版本。版本矩阵以 vLLM 官方文档为准（适用于 vLLM v0.20+ 的配套通常是 PyTorch 2.11+ 与 CUDA 12.4+）。

```bash
# 先看驱动支持的最高 CUDA 版本
nvidia-smi | head -5      # 关注右上角 "CUDA Version: xx.x"

# 再看已装 PyTorch 实际绑定的 CUDA
python -c "import torch; print(torch.version.cuda, torch.__version__)"
```

> **反模式**：看到报错就无脑升级驱动。生产机的驱动版本往往受运维规范约束，升级驱动可能牵连同机其他服务。正确顺序是「先看驱动上限 → 选匹配的 wheel」，而不是「先装最新 wheel → 再被迫升驱动」。

**常见安装失败对照**：

| 现象 | 可能原因 | 解法 |
|---|---|---|
| `torch.cuda.is_available()` 返回 False | PyTorch 装成了 CPU 版，或 CUDA 不匹配 | 重装对应 `cu` 版本的 PyTorch |
| `ImportError: libcudart.so.x` 找不到 | 系统 CUDA runtime 与 wheel 不一致 | 装匹配的 NVIDIA CUDA Toolkit 或用 Docker |
| 安装时卡在编译 flash-attn | 缺少编译工具链 / 网络超时 | 换预编译 wheel，或 Docker 镜像已内置 |
| `vllm` 命令不存在 | 装到了错误环境 | `which python` 确认，激活正确 venv |

### 8.1.3 国内网络：镜像加速与离线安装

境外 HuggingFace 下载慢甚至不通，是第二大坑。两招解决：

**第一招，pip 换源**（安装阶段）：

```bash
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**第二招，模型权重走 ModelScope**（下载阶段，详见 8.5 节）。模型权重才是真正的大文件，pip 源帮不上这个忙。

**离线安装**：完全无外网的机器，先在能联网的机器上 `pip download vllm`（加 `--platform` / `--python-version` 指定目标环境）把 wheel 和依赖打包，再到内网 `pip install --no-index --find-links ./wheels vllm`。注意 `pip download` 要带上 `--extra-index-url` 指向 PyTorch 的 cu 源，否则 PyTorch 这个大依赖会漏掉。

> **作者经验**：离线包最容易被忽略的是 flash-attn、xformers 这类带编译产物的依赖。如果目标机架构（CPU 指令集、glibc 版本）和下载机不一致，拷过去的 wheel 可能装得上跑不了。离线环境首推 Docker——把整个镜像在联网机 `docker save` 成 tar，内网 `docker load`，连带 CUDA runtime 一起搬过去，最稳。

### 8.1.4 三选一决策清单

把上面三类安装方式落成一张可执行的判断表，照着选就不会错：

| 你的处境 | 选哪个 | 理由 |
|---|---|---|
| 本地开发、有网、想快速试 | pip + venv | 一条命令，改代码方便 |
| 生产部署、要环境一致、有 Docker | Docker 镜像 | 免去「在我机器能跑」 |
| 要改 vLLM 引擎源码 / 追未发版特性 | 源码编译 | 只有这种才需要 |
| 完全无外网 | Docker + `docker save/load` | 连带 CUDA runtime 一起搬 |
| 内网但能传文件 | pip 离线包 + 本地权重 | 轻量，但依赖架构一致 |

> **一句话原则**：能用 Docker 就用 Docker，能 pip 就别编译，除非你在改引擎本身。

### 8.1.5 升级 vLLM 的注意事项

vLLM 迭代极快，升级时几个坑要提前知道：第一，**V1 引擎在 v0.8 后成为默认**，老脚本里若显式关过 V0 参数（如 `--disable-...`）升级后可能失效或行为变化；第二，**CLI 参数名会随版本调整**，升级后先 `vllm serve --help` 比对一遍，别凭记忆敲；第三，**量化格式支持矩阵会变**，旧版本能加载的格式新版可能改了名字或默认后端。升级策略建议：先在测试机用相同模型和参数跑一遍冒烟测试，确认接口与精度无回归再上生产。

---

## 8.2 离线推理：LLM 类

「离线推理」指进程内直接调用 `LLM` 类批量生成，不启 HTTP 服务。适合批处理任务：文档摘要、数据标注、评测跑分。它的好处是没有网络开销、没有多进程调度，是验证「模型本身能不能跑、效果对不对」的最快路径。

### 8.2.1 最小可运行示例

```python
# ch06-vllm-quickstart/offline_inference.py  （vLLM v0.20+ 适用）
from vllm import LLM, SamplingParams

# 模型：0.8B 级小模型（显存占用极低，适合先跑通链路）
model = LLM(model="Qwen/Qwen3.5-0.8B", gpu_memory_utilization=0.90)

# 采样参数：温度 0.7，最多生成 256 个 token
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

prompts = [
    "用一句话解释什么是 KV Cache。",
    "如何提高大模型推理的吞吐？",
]

outputs = model.generate(prompts, sampling_params)
for out in outputs:
    print(f"=== Prompt: {out.prompt[:20]}...")
    print(out.outputs[0].text)
```

跑起来前确认显存够：Qwen3.5-0.8B 的 BF16 权重约 1.7GB，单卡 24GB 余量充足（连 KV Cache 都绰绰有余）。如果报 OOM，先加 `--gpu-memory-utilization 0.85` 这类参数（见 6.4），或换更小模型。

### 8.2.2 采样参数与输出结构

`SamplingParams` 除了 `temperature` / `top_p` / `max_tokens`，还有几个常被忽略但很实用：

- `top_k`：每步只在概率最高的 k 个 token 里采，配合 `top_p` 更稳。
- `repetition_penalty`：抑制重复生成，长文本、摘要场景建议设 1.05–1.15。
- `stop`：自定义停止串，比如生成代码时遇到 ```` ``` ```` 就停。
- `n`：一次要几个候选；配合 `best_of` 可做简单重排（注意这会乘算显存与耗时）。

输出结构值得记住：`outputs` 是 `RequestOutput` 列表，每个里有 `prompt`、`finished`、`outputs[0].text`、`outputs[0].token_ids`、`outputs[0].logprobs`。做评测或后处理时，`token_ids` 比 `text` 更可靠（避免 detokenizer 细节差异）。

### 8.2.3 批量推理

`model.generate` 天然接受 list，内部走连续批处理（continuous batching，第 5 章会展开原理），你不需要自己手搓 batch。但要注意：**批的大小由引擎按显存动态决定，不是你传多少条就同时跑多少条。** 传 1000 条 prompt，引擎会自动排队，你拿到的仍是 1000 条结果。

```python
# 大数据集分批传入即可，vLLM 自己管理调度
outputs = model.generate(prompts, sampling_params)  # prompts 可以是很长的一个列表
```

> **作者经验**：离线推理最常见的性能误区是「我传了 N 条，怎么单条延迟还这么高」。因为引擎在攒够显存允许的并发才会一起算，短任务批量反而可能因为排队等凑批而变慢。离线场景要的是总吞吐，别用单条延迟衡量它。如果单条极长、彼此独立，可以考虑 `LLM.generate` 的异步接口或拆成多进程。

### 8.2.4 流式输出

离线 `LLM` 类默认是「算完一起返回」。需要逐 token 流式，用 `llm.generate` 配合参数 `stream=True` 拿到迭代器：

```python
# ch06-vllm-quickstart/stream_output.py
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3.5-0.8B")
params = SamplingParams(temperature=0.7, max_tokens=256)

# stream=True 返回生成器，逐 token 产出
for output in llm.generate("讲一个关于缓存的三行小故事。", params, stream=True):
    print(output.outputs[0].text, end="", flush=True)
```

> **注意**：离线流的用途主要是「本地看效果」，真正的流式服务请走 6.3 的 OpenAI 兼容接口——那是带 SSE 的 HTTP 流，前端能直接消费。

---

## 8.3 OpenAI 兼容 API Server

对外提供服务的正道是起一个 HTTP Server。`vllm serve` 直接给出一个 OpenAI 兼容接口，意味着你现有的 OpenAI SDK 代码**只改一行 `base_url` 就能迁移**，这是 vLLM 最实在的吸引力之一。本章所有迁移收益，都建立在「接口契约一致」这件事上。

### 8.3.1 启动与验证

```bash
# 启动一个 OpenAI 兼容服务（vLLM v0.20+）
vllm serve Qwen/Qwen3.5-0.8B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.90
```

启动后先验证服务活着：

```bash
# 列出已加载模型
curl http://localhost:8000/v1/models

# 最简 chat 请求
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 128
  }'
```

### 8.3.2 用 OpenAI SDK 直连

```python
# ch06-vllm-quickstart/openai_client.py
from openai import OpenAI

# 关键就这一行：把 base_url 指向本地 vLLM
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",          # vLLM 默认不校验 key
)

resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[{"role": "user", "content": "用一句话解释 PagedAttention。"}],
    max_tokens=128,
    stream=True,             # 流式
)
for chunk in resp:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

> **作者经验**：迁移到 vLLM 时，真正的改动量通常就是 `base_url` 一行。但有两个地方会「编译过了、运行却不对」：一是 `api_key` 必须显式给（哪怕是 `"EMPTY"`，否则 SDK 默认去读环境变量）；二是流式返回结构里 `delta.content` 可能为空（role 占位），遍历时务必判空，否则生产日志里会刷一堆 `None`。

### 8.3.3 服务化必须知道的几件事

**健康检查**：`GET /health` 返回 200 表示进程存活（注意它不等同于「能正常推理」，只说明 API server 起来了）。配合 K8s 探针用。

**多模型**：一个 `vllm serve` 进程默认只加载一个模型。要服务多个模型，要么起多个进程各自占卡，要么用 `--served-model-name` 给同一权重起不同别名。真正的多模型异构，建议用多个实例 + 网关路由（第 20 章）。

**并发真相**：`vllm serve` 起来后，能同时处理的请求数由 6.4 的 `max-num-seqs` 决定，不是「来多少接多少」。超过上限的请求会在引擎内部排队，表现为 TTFT 升高而非被拒绝——这一点对估容量很关键。

**鉴权**：生产环境默认 `api_key="EMPTY"` 等于无鉴权。需要鉴权时，用反向代理（如 Nginx / 网关）加一层，或启用 vLLM 的 API key 校验机制，别把裸服务直接暴露公网。

### 8.3.4 常用接口清单

| 接口 | 用途 | 备注 |
|---|---|---|
| `POST /v1/chat/completions` | 对话式生成（带 messages 结构） | 最常用 |
| `POST /v1/completions` | 续写式生成（裸 prompt） | 评测 / 老代码 |
| `GET /v1/models` | 列出可用模型 | 健康检查可用 |
| `POST /v1/embeddings` | 向量化（需 Embedding 模型） | 第 9.6 节 |
| `POST /v1/rerank` | 重排序（需 Rerank 模型） | 第 9.6 节 |
| 上述接口 `stream: true` | SSE 流式返回 | 前端逐字显示 |

---

## 8.4 启动参数精解

![张量并行：把模型按层内维度切开，放到同一台机器的多张卡上](./assets/ch08-tensor-parallel.png)

本节是全章分量最重的部分。vLLM 的 CLI 参数上百个，新手最容易犯两个错：一是**照抄别人博客的数**，二是**把参数当独立旋钮开**。参数之间是有耦合的（比如 `gpu-memory-utilization` 直接决定 `max-num-seqs` 能开多大）。

下面按四类梳理最关键的一组。每个参数严格遵守四要素：**默认值 → 影响什么 → 怎么调 → 调错的后果**。

### 8.4.1 显存类

| 参数 | 默认值 | 影响什么 | 怎么调 | 调错的后果 |
|---|---|---|---|---|
| `--gpu-memory-utilization` （vLLM v0.20+） | `0.90` | 引擎**最多能占用单卡显存的比例**，其余留给碎片、激活值和系统。KV Cache 池就从这个比例里划分。 | 显存紧张就调低（如 `0.85`）；想压出更多 KV 空间就调高（如 `0.95`）。调高前确认没有其他进程抢显存。 | **设太高**（如 `0.99`）→ 激活值或碎片没空间，运行中 OOM 崩溃，且是「跑着跑着才挂」最隐蔽；**设太低** → KV 池太小，并发上不去，吞吐浪费。 |
| `--swap-space` （vLLM v0.20+） | `4`（GB） | 当 KV Cache 显存不够时，可换出到** CPU 内存**的空间大小。是「抢占」的兜底手段。 | 长上下文、大并发场景可调大（如 `16`）。需机器有空闲内存。 | **设太小** → 高并发下无法换出，请求被拒或反复抢占，延迟抖动；**设太大** → 换出到 CPU 走 PCIe，极慢，一旦真触发 swap，单个请求延迟爆表。swap 是保命阀，不是性能手段。 |
| `--kv-cache-dtype` （vLLM v0.20+） | `auto`（即权重同精度，BF16/FP16） | KV Cache 的存储精度。可设 `fp8_e5m2` / `fp8_e4m3` 等，用精度换显存。 | 显存吃紧、长上下文、且业务对 KV 精度不敏感时设 `fp8_e5m2`。 | **盲目设 FP8** → 长上下文或高精度任务下输出质量肉眼可见下降；**该省不省** → KV 占满显存，并发被压死。KV 量化对效果的影响见第 9.2 节。 |

> **作者经验**：`gpu-memory-utilization` 不要迷信「越大越好」。0.90 是官方给的稳妥默认值，背后留了 10% 给碎片和激活值。我见过太多人设到 0.98 然后「凌晨三点服务随机 OOM」——碎片是运行时动态产生的，静态估计永远估不准。稳定优先的场景，0.90–0.92 是甜区。

### 8.4.2 并发类

| 参数 | 默认值 | 影响什么 | 怎么调 | 调错的后果 |
|---|---|---|---|---|
| `--max-num-seqs` （vLLM v0.20+） | `256` | **同一时刻在引擎里并发处理的请求数上限**。直接决定排队与吞吐。 | 短请求高并发可调大；显存 / 延迟扛不住就调小。它受 KV 池大小硬约束。 | **设太大** → KV 池被撑爆 OOM，或单请求 TPOT 飙升（大家抢算力）；**设太小** → GPU 喂不饱，吞吐浪费，请求排队。 |
| `--max-num-batched-tokens` （vLLM v0.20+，V1 新增关键参数） | `2048`（V1 下以文档为准） | **一个调度步里所有请求 input+output token 的总量上限**。控制每批「吃多少 token」。 | 长 prompt 多 → 调大（如 `8192`/`16384`）；纯短请求 → 可调小降延迟。 | **设太大** → 单步 prefill 耗时变长，正在 decode 的请求被饿，TPOT 抖动；**设太小** → 长 prompt 被迫分很多 chunk，prefill 效率低、TTFT 变高。这是吞吐-延迟三角的核心杠杆，第 10.3 节专门扫。 |
| `--max-model-len` （vLLM v0.20+） | 模型自身 `max_position_embeddings`（如 32768） | 引擎**允许的最大序列长度**（输入+输出）。KV 池按它预估。 | 只用短文本就调小（如 `4096`），省下 KV 显存给并发；确有长上下文需求才设大。 | **设太大** → 即使你不用长文本，引擎也按最大长度预留 KV 结构，显存被虚占，并发上不去；**设太小** → 超长请求直接被拒。这个值应略大于「业务最长请求」，不是越大越好。 |

> **关键耦合提醒**：`max-num-seqs × 平均序列长度 × 每层 KV 大小` 必须塞进「`gpu-memory-utilization` 划出的 KV 池」。三个并发类参数里任意一个变大，都会挤压另外两个的上限。**调参时把它们当一组，而不是三个独立开关**——这就是本书反复强调的「参数耦合」，也是第 0.7 节所说「工程复杂度」最典型的体现。

### 8.4.3 并行类

| 参数 | 默认值 | 影响什么 | 怎么调 | 调错的后果 |
|---|---|---|---|---|
| `--tensor-parallel-size`（简称 `--tp`） （vLLM v0.20+） | `1` | **张量并行度**：把一层权重切到几张卡。模型放不进单卡时用。 | 模型权重 > 单卡显存就调大（7B→1，70B→8 等）。要求卡间高带宽（NVLink）。 | **跨节点设 TP** → 卡间走低速网络，通信成本 > 切分收益，反而更慢；**TP 设太大** → 通信开销边际递增，收益递减。详见第 9.1 节。 |
| `--pipeline-parallel-size`（简称 `--pp`） （vLLM v0.20+） | `1` | **流水并行度**：把不同层切到不同卡，切段转发。 | 模型层数多、TP 受限时配合用（如 TP4+PP2）。 | **PP 切太小**（如每层一卡）→ bubble（流水线气泡）严重，GPU 空转；**PP 与 TP 组合不当** → 既没喂饱也没省显存。第 6 章有决策表。 |
| `--data-parallel-size`（简称 `--dp`） （vLLM v0.20+，V1 强化） | `1` | **数据并行**：同一权重起多个副本，各吃不同请求，前端聚合。多机常用。 | 需要横向扩展吞吐、且单副本已占满显存时设 >1。需配合路由（避免前缀缓存命中率被稀释，见第 4.6 节）。 | **DP 配轮询路由** → 前缀缓存命中率被稀释为 1/N，缓存收益归零；**DP 副本数超过请求量** → 资源空转浪费。 |

> **反模式**：「模型放不进单卡，那就 TP 拉满到卡数」。实际上 TP 的通信是卡间 all-reduce，对带宽极其敏感。单机 NVLink 下 TP8 尚可，一旦 TP 跨节点（走 IB / 以太网），通信开销会吃掉大半收益。**放不进单卡时，先想 TP 能不能在单机内解决；不能才上 PP 或多机 DP。**

### 8.4.4 其他常用参数

| 参数 | 默认值 | 影响什么 | 怎么调 |
|---|---|---|---|
| `--enable-prefix-caching` （vLLM v0.20+） | V1 下默认开启 | 跨请求前缀 KV 复用（第 4 章）。共享 system prompt / 多轮对话场景收益大。 | 共享前缀多的负载保持开启；全随机 prompt 可关掉省一点管理开销。 |
| `--enable-chunked-prefill` （vLLM v0.20+） | 默认开启 | 长 prompt 分块与 decode 混合组批，避免 prefill 卡住 decode。 | 长 prompt 多时保持开启；极短请求为主时可评估关闭降低调度复杂度。 |
| `--dtype` （vLLM v0.20+） | `auto` | 推理计算精度（BF16 / FP16 / FP32 / half）。 | 默认 `auto` 跟随权重；追求极致吞吐且硬件支持可显式 `half`。 |
| `--max-logprobs` （vLLM v0.20+） | `20` | 返回 logprobs 的 top-k 数量。 | 需要置信度 / beam 类分析时调大，但会增加开销。 |
| `--served-model-name` （vLLM v0.20+） | 模型路径 / 名 | 对外暴露的模型名。 | 用与版本解耦的稳定别名（见 6.5.3）。 |

### 8.4.5 新手调参顺序建议

给一个可执行的顺序，避免一上来就乱调：

1. **先让它能跑**：默认参数起服务，确认模型加载、接口通。
2. **定 `--max-model-len`**：按业务最长请求设，别用模型默认值（详见 6.4.2）。
3. **调 `--gpu-memory-utilization`**：0.90 起，稳定优先不超 0.92。
4. **看并发上不去 → 调 `--max-num-seqs` / `--max-num-batched-tokens`**：这两个受 KV 池硬约束，调不动先回头看第 3 步。
5. **多卡才动并行类**：单卡能跑就别碰 TP/PP。

> **作者经验**：新人最容易在「吞吐上不去」时去调各种奇怪参数，其实 80% 的情况是 `max-num-seqs` 被 KV 池卡住了。先用 `vllm serve` 的日志看 KV cache 实际利用率（日志里会打 `GPU KV cache size`），它接近 100% 而并发还上不去，才说明瓶颈在并发参数，而不是显存。

### 8.4.6 一个真实调参故事（讲清耦合）

讲一个我反复见到的现场，帮你看清参数为什么不能单点调：

某团队用 8B 模型、单卡 24GB，默认参数起服务，并发一高就 OOM。他们的第一反应是「把 `max-num-seqs` 调小到 32」——结果 OOM 是不报了，但吞吐掉了快一半，因为 GPU 没喂饱。其实根因是 **`gpu-memory-utilization` 默认 0.90 留给 KV 池的空间，在长请求下不够装 256 个并发的 KV**。

正确的修法是一组联动：把 `max-model-len` 从默认 32768 降到业务真实的 8192（KV 池需求直接降到 1/4），KV 池空出来，再把 `max-num-seqs` 适当提到 128——既没 OOM，吞吐也回来了。这个例子里，**单独动任何一个参数都解决不了问题，必须 `max-model-len → KV 池 → max-num-seqs` 一起看**。这就是参数耦合，第 10 章要系统展开。

---

## 8.5 模型获取与离线部署

### 8.5.1 从 HuggingFace / ModelScope 获取

```bash
# 方式一：HuggingFace（境外，适合有网环境）
# vLLM 启动时若本地无权重会自动下载，或用 huggingface-cli 预拉
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir ./models/Qwen3.5-0.8B

# 方式二：ModelScope（国内推荐，速度快）
pip install modelscope
modelscope download Qwen/Qwen3.5-0.8B --local_dir ./models/Qwen3.5-0.8B
```

拿到权重后，启动指向本地目录即可，**不再走网络**：

```bash
vllm serve ./models/Qwen3.5-0.8B --gpu-memory-utilization 0.90
```

| 维度 | HuggingFace | ModelScope |
|---|---|---|
| 国内下载速度 | 通常慢 / 不通 | 快 |
| 模型覆盖 | 最全（含大量社区权重） | 国内主流模型齐全 |
| 命令行 | `huggingface-cli` | `modelscope download` |

> **作者经验**：境外环境若必须用 HF，配 `HF_ENDPOINT=https://hf-mirror.com` 这类镜像能显著提速。无论哪个源，下载完第一件事是校验完整性（见下节），别等到 `vllm serve` 加载一半才报损坏。

### 8.5.2 权重目录结构要求

vLLM 兼容 HuggingFace 格式的权重目录。一个能直接被 `LLM(model="<dir>")` 或 `vllm serve <dir>` 加载的目录，至少包含：

```
Qwen3.5-0.8B/
├── config.json          # 模型结构定义（必填，引擎读它判断架构/层数/隐藏维）
├── tokenizer.json       # 分词器（必填）
├── tokenizer_config.json
├── model.safetensors    # 权重（或 pytorch_model-*.bin / 分片 *.safetensors）
├── generation_config.json
└── ...
```

> **作者经验**：权重下到一半断网，目录里会留下 `.safetensors` 但不完整，vLLM 会报「文件损坏 / 校验失败」。判断本地权重是否完整的最快方法：看 `config.json` 里的 `num_hidden_layers` 等字段是否齐全，再核对 `model.safetensors.index.json`（若有）里声明的分片是否都在。不要凭「目录存在」就认为能加载。更稳妥的是下载后跑一次 `python -c "from transformers import AutoModel.from_pretrained('<dir>')"` 做轻量加载校验。

### 8.5.3 内网离线部署完整流程

把上面两步串起来，就是一条可复现的内网落地路径：

1. **联网机下载**：用 ModelScope / HuggingFace 把权重拉到本地目录，校验完整性。
2. **打包搬运**：权重目录 `tar` 压缩，连同 Docker 镜像（`docker save vllm/vllm-openai:tag -o vllm.tar`）一起拷到内网机。纯 pip 环境则改用 `pip download` 离线包（见 6.1.3）。
3. **内网加载**：`docker load -i vllm.tar`，然后：

```bash
# 内网离线启动：--model 指向本地目录，--served-model-name 给稳定别名
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/models:/models \
  vllm/vllm-openai:latest \
  --model /models/Qwen3.5-0.8B \
  --gpu-memory-utilization 0.90 \
  --served-model-name qwen35-08b   # 给服务起个稳定名字，避免调用方写死路径
```

4. **验证**：`curl http://<内网IP>:8000/v1/models` 返回模型即成功。调用方用 6.3.2 的 OpenAI SDK，`base_url` 换成内网地址，零改动迁移。

> **作者经验**：内网部署最容易被忽略的是「服务名」。很多团队把 `served-model-name` 设成路径或版本号，结果模型一升级，所有调用方代码都要改。建议用一个**与版本解耦的稳定别名**（如 `qwen35-08b`），升级时改指向即可，调用方代码不动。这是生产环境的小但关键的约定。

---

## 本章小结

1. **安装优先选 Docker**：免去「在我机器上能跑」的环境差异。pip 适合本地开发，源码编译只在改引擎本身时才需要，新手不必碰。
2. **版本匹配是硬约束**：驱动支持的 CUDA 上限 → PyTorch wheel 的 CUDA 版本 → vLLM 依赖，三者绑定。报错先 `nvidia-smi` 看 CUDA 上限，再选 wheel，别无脑升驱动。
3. **离线推理用 `LLM` 类，对外服务用 `vllm serve`**：前者适合批量任务，后者给 OpenAI 兼容接口，迁移成本仅一行 `base_url`。
4. **启动参数要成组调**：`gpu-memory-utilization` 决定 KV 池，进而约束 `max-num-seqs` 等并发参数；它们彼此耦合，照抄单点参数容易翻车。
5. **`gpu-memory-utilization` 不是越大越好**：0.90 是留了碎片的甜区，盲目拉到 0.98 会换来「跑着跑着 OOM」的午夜事故。
6. **内网部署要解耦服务名**：用稳定的 `served-model-name` 别名，权重升级时调用方代码零改动
