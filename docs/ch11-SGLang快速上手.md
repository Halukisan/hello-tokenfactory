# 第 11 章　SGLang 快速上手与低延迟实战

> **本章要回答的问题**
> 1. 已经会用 vLLM，切到 SGLang 到底要改多少东西？
> 2. SGLang 有哪三种使用方式，分别适配什么场景？
> 3. 前端 DSL（gen / select / fork / join）到底特别在哪，为什么能"保证"前缀共享？
> 4. vLLM 的启动参数怎么一一映射到 SGLang，迁移时哪些不用动、哪些必须重写？
> 5. 最常用的一批启动参数，默认值是什么、改大了改小了分别会怎样？

难度 ★☆☆　|　硬件要求：🖥️　|　⏭️ 无（全章建议通读；9.3 是本章特色但非必跳）

---

## 11.1 安装与环境

SGLang 的安装路径和 vLLM 几乎平行：pip 装、`lmsysorg/sglang` 官方镜像跑。如果你机器上已经能跑 vLLM，那么 SGLang 大概率也满足环境前提——它同样依赖 CUDA、同样吃 Triton、同样需要较新的 PyTorch。

### 11.1.1 pip 安装

推荐用官方提供的「带 CUDA 轮子」的极简命令，避免在你本机编译 Triton kernel：

```bash
# 干净虚拟环境里执行，Python 3.10+
pip install "sglang[all]>=0.5.18"   # 适用 v0.5.18+，更高版本请以官方文档为准

# 验证安装
python -c "import sglang; print(sglang.__version__)"
```

`[all]` 会顺带装好 `sgl-kernel`、`xgrammar`、`torch` 等依赖。如果你只想用 OpenAI 兼容 Server、不需要 DSL 与结构化生成，可只装 `pip install sglang`，省一点体积。

### 11.1.2 Docker 安装（生产首选）

和 vLLM 一样，生产环境强烈建议用镜像，避免宿主机 CUDA 驱动与容器内核版本错配：

```bash
docker run --gpus all --shm-size 32g \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-0.8B \
    --tp 1 --port 30000
```

> 注意：SGLang 镜像里默认已带匹配好的 CUDA / PyTorch / xgrammar，因此 `--gpus all` 之外的宿主机依赖只有 NVIDIA Container Toolkit。这一点与 vLLM 的 `vllm/vllm-openai` 镜像体验一致。

### 11.1.3 与 vLLM 的环境差异（重点）

站在"已会 vLLM"的视角，下面这几条差异最值得记：

| 维度 | vLLM | SGLang | 对你的影响 |
|---|---|---|---|
| 核心依赖 | vLLM 自带 `vllm-flash-attn` 等 | 依赖 `sgl-kernel` 与 `xgrammar` | 同机共存时两个包的 Triton 环境可能打架，建议分虚拟环境 |
| 结构化生成后端 | xgrammar / guidance | 原生 xgrammar（压缩 FSM） | 概念通用，迁移成本低（详见 9.2） |
| 显存占用感知 | `gpu-memory-utilization` | `mem-fraction-static` | **语义不同**（见 9.5），直接套用会 OOM |
| 默认前缀缓存 | V1 默认开启 automatic prefix cache | RadixAttention 默认开启 | 行为接近，但命中粒度与路由语义不同（见 9.1 / 9.5） |

**一条实战提醒**：不要把 SGLang 和 vLLM 装进同一个虚拟环境比对压测。两者对 Triton、flash-attn 的版本诉求不完全一致，混装后常见症状是「import 某方就 Segmentation fault」。需要时各起一个 conda / venv 环境即可。

### 11.1.4 版本兼容性注意事项

- **CUDA / 驱动**：SGLang v0.5.x 需要 CUDA 12.x 与较新驱动（建议 535+，H 卡推荐 550+）。低于此会卡在 kernel 编译或 `sgl-kernel` 加载失败。
- **PyTorch**：v0.5.18+ 配套 PyTorch 2.11+。不要拿一年前的 2.1 老环境硬跑，Triton 算子不兼容。
- **模型支持节奏**：SGLang 以"前沿模型 Day-0 支持"著称（详见 9.4），但**新模型刚合入时 API 细节可能漂移**。生产锁定版本，不要用 `:latest` 裸奔。

> **什么场景值得用 / 不值得用**
> - 值得：已经在用 vLLM、想试 SGLang 的 RadixAttention 前缀缓存或 DSL 多分支推理；需要 Day-0 跑刚发布的新模型。
> - 不值得：生产强依赖某个 vLLM 独占能力（如某些特定 LoRA 调度、特定量化后端），且暂未验证 SGLang 等价支持时，先并行跑、别急着替换。

---

## 11.2 三种使用方式

![前缀命中：多个请求共享同一段 KV，只算没命中的后缀](./assets/ch11-prefix-cache-hit.png)

SGLang 提供三种使用姿势，从"零改动"到"全控制"递进。选哪种取决于你愿不愿意动业务代码。

### 11.2.1 方式一：OpenAI 兼容 Server（迁移成本最低）

这是给"已会 vLLM 的读者"最顺的一条路——`python -m sglang.launch_server` 起的端口，和 vLLM 的 `vllm serve` 一样，兼容 `/v1/chat/completions`、`/v1/completions`、`/v1/models`、流式接口。

```bash
# 启动（SGLang v0.5.18+ 同时支持 sglang serve 简写）
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-0.8B \
  --tp 1 \
  --port 30000 \
  --host 0.0.0.0
```

```python
# 客户端：把 base_url 从 vLLM 改成 SGLang 即可，其余一行不动
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen3.5-0.8B",
    messages=[{"role": "user", "content": "用一句话解释 KV Cache。"}],
    stream=False,
)
print(resp.choices[0].message.content)
```

**关键判断**：如果你的业务已经走 OpenAI SDK，那么从 vLLM 切到 SGLang **代码零改动**，只换 `base_url` 和启动命令。这一节的迁移成本 = 0。

代码见 `src/ch11-sglang-quickstart/01_openai_server.py`。

### 11.2.2 方式二：`sgl.Engine` 进程内引擎（离线批处理 / RL rollout）

当你不想起一个 HTTP 服务、只想在 Python 进程里把一批 prompt 灌进去拿结果，用 `sgl.Engine`：

```python
import sglang as sgl

# 进程内引擎：权重随进程加载，省去网络往返
engine = sgl.Engine(model_path="Qwen/Qwen3.5-0.8B", tp=1)

# 离线批量生成
outputs = engine.generate(
    [
        {"role": "user", "content": "把这句话翻译成英文：你好世界"},
        {"role": "user", "content": "给'推理引擎'下个定义"},
    ],
    sampling_params={"temperature": 0, "max_new_tokens": 256},
)
for o in outputs:
    print(o["text"])
```

这种用法最典型的场景有三个：

1. **离线批处理**：几万条数据跑一遍抽取/打标，不需要在线延迟，只要吞吐。直接 `engine.generate(list)` 比每条约一次 HTTP 快得多（省去序列化与排队）。
2. **RL rollout**：强化学习训练循环里，每步都要拿策略模型采样一批轨迹。进程内引擎 + 权重热更新（见 9.6）能跳过"重新加载权重"的昂贵环节。
3. **压测前的本地冒烟**：先确认引擎能跑通，再去起 server 压线上形态。

> **什么场景值得用 / 不值得用**
> - 值得：离线打标、评测基准（如 MMLU / GSM8K 批量跑）、RL rollout、轻量本地脚本。
> - 不值得：需要多进程/多机横向扩展、需要鉴权与限流、需要被多个业务方共享时——这些该用 9.2.1 的 Server，而不是在每处业务代码里塞一个 Engine。

### 11.2.3 方式三：前端 DSL（本章特色）

前两种方式对 vLLM 用户都不陌生。真正 SGLang 独有的，是 11.3 节要讲的前端 DSL：把"prompt 怎么拼、怎么分支、怎么抽取"写成一段程序，由引擎负责调度和前缀共享。先放一个最小味道：

```python
import sglang as sgl

@sgl.function
def qa(s, question):
    s += sgl.system("你是严谨的技术助手。")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))

runtime = sgl.Runtime(model_path="Qwen/Qwen3.5-0.8B", tp=1)
print(qa.run(question="什么是 RadixAttention？", runtime=runtime).get("answer"))
```

它看起来只是"把 prompt 模板写成代码"，但配上 `fork` / `join` 之后，就能表达 vLLM 纯 prompt 拼不出结构的东西（见 9.3）。

---

## 11.3 SGLang 前端 DSL

![SGLang 前端 DSL：gen / select / fork / join 把流程写成一张可分支的图](./assets/ch11-sglang-dsl.png)

> 本节是本章特色，也是 SGLang 相对 vLLM 区分度的来源。如果时间紧，至少读完 9.3.2「为什么 DSL 能保证前缀共享」——它解释了 SGLang 的核心卖点。

### 11.3.1 四个原语

SGLang DSL 的骨架是四个原语，把它们当"控制流 + 生成"的组合来理解：

| 原语 | 作用 | 一句话直觉 |
|---|---|---|
| `sgl.gen(name, max_tokens=…)` | 让模型生成一段文本，存到 `name` | "模型来补这一段" |
| `sgl.select(options, name=…)` | 从给定候选里选一个（无需采样） | "分类/打分，答案只能在这几个里" |
| `sgl.fork(n)` | 把当前状态复制成 n 个分支 | "同一个前缀，后面走 n 条路" |
| `sgl.join(states)` | 把若干分支汇回 | "n 条路的结果收口" |

**示例 1：`gen` —— 普通生成**

```python
@sgl.function
def gen_demo(s, topic):
    s += sgl.user(f"用三句话介绍：{topic}")
    s += sgl.assistant(sgl.gen("out", max_tokens=200))
```

**示例 2：`select` —— 分类打分（候选受约束，天然就是结构化输出）**

```python
@sgl.function
def sentiment(s, text):
    s += sgl.user(f"判断情感倾向：{text}")
    # select 直接把输出锁死在三个选项里，不会吐出别的字
    s += sgl.assistant(sgl.select(["正面", "负面", "中性"], name="label"))
```

`select` 的底层就是 12.2 节要讲的压缩 FSM：候选集被编译成状态机，非法 token 在解码时被屏蔽。所以它既是"分类器写法"，也是"零重试的结构化输出"。

**示例 3：`fork` + `join` —— 多路分支推理（DSL 的杀手锏）**

```python
@sgl.function
def multi_style(s, question):
    s += sgl.system("下面请就同一问题给出三种风格的回答。")
    s += sgl.user(question)
    # 关键：fork 之前的所有 token（system + user）构成一个共享前缀
    branches = s.fork(3)
    for i, st in enumerate(branches):
        st += sgl.user(f"风格 {i+1} 的回答：")
        st += sgl.assistant(sgl.gen(f"ans_{i}", max_tokens=128))
    s.join(branches)
```

`fork` 之后的三段生成，前缀（system + user question）完全相同。RadixAttention 会**只 prefill 一次这个共享前缀**，三个分支复用同一份 KV。这正是 vLLM 纯靠 prompt 拼不出、靠"运气命中自动前缀缓存"也不稳定的地方（详见 9.3.2）。

**示例 4：`gen` + `select` 组合 —— 结构化抽取**

```python
@sgl.function
def extract(s, doc):
    s += sgl.system("从文档中抽取实体，逐个判定类型。")
    s += sgl.user(doc)
    for _ in range(3):                      # 抽三个实体
        s += sgl.assistant(sgl.gen("entity", max_tokens=16))
        s += sgl.assistant(sgl.select(["人物", "机构", "地点"], name="type"))
```

### 11.3.2 为什么 DSL 能保证前缀共享（而非依赖运气）

这是读者最容易误解、也最关键的一点。先把结论摆出来：

> **DSL 本身并不比自动前缀缓存"更会命中"。RadixAttention 的命中判定，和 vLLM 的 automatic prefix cache 一样，本质都是"两个请求的 token 前缀是否逐 token 相同"。DSL 真正的价值，是把"前缀是否相同"从运气变成了你可以主动设计的东西。**

展开说三层：

**第一层：缓存命中只看 token 序列。** 无论是 SGLang 还是 vLLM，前缀缓存命中的充要条件都是——新请求的初始 token 串，和树/表里已存的某段前缀逐 token 完全一致。模型、tokenizer、拼接方式稍有不同，命中就破。

**第二层：手写 prompt 容易"意外打破"前缀。** 在 vLLM 里你用 f-string 拼 prompt，稍不留神就埋了不确定因素：时间戳、随机 request_id、按请求 shuffle 的 few-shot 顺序、长度不定的中间结果回填。这些都会让"本该相同"的前缀在第一个 token 就分叉，自动前缀缓存命中率掉到 0。第 4 章（4.6 节）把这类坑叫"前缀缓存的四个实战陷阱"，vLLM 用户踩得最多。

**第三层：DSL 把"共享前缀"写成显式结构。** `fork` 的语义就是"先构造好一个确定的前缀，再分叉"。因为前缀是程序里写死、对所有分支相同的 token 序列，RadixAttention 必然能匹配上——这不是概率，是构造使然。换句话说：

- vLLM 自动前缀缓存：前缀相不相同，**取决于你拼 prompt 有没有手滑**（运气 + 纪律）。
- SGLang DSL + fork：前缀相不相同，**由代码控制流保证**（必然）。

**诚实的边界**：如果你不写 `fork`、只是把 DSL 当模板引擎用，那它的前缀共享能力和 vLLM 自动前缀缓存没有本质差别——都靠 token 相同。DSL 的红利只在"你确实在利用它的结构（分支、复用、受约束生成）"时才兑现。不要神化它：它不是给缓存加了魔法，而是给了你一把能稳定造出"可复用前缀"的尺子。

代码见 `src/ch11-sglang-quickstart/03_dsl.py`（含 gen/select/fork/join 四个完整示例）。

> **什么场景值得用 / 不值得用**
> - 值得：多路分支推理（同前缀多风格/多候选）、分类打分、结构化抽取、Agent 里"先共享上下文再分头探索"。
> - 不值得：单一线性的 chat 请求、纯 RAG 问答且 prompt 已由你严格控制前缀——这类用 OpenAI Server（9.2.1）更省事，DSL 带来的收益有限，反而增加代码复杂度。

---

## 11.4 vLLM → SGLang 参数对照

本节做一张"能拿着做迁移"的对照表。约定：左列 vLLM 为 V1 引擎（`vllm serve`）参数；右列 SGLang 为 `python -m sglang.launch_server` 参数。

> **重要**：SGLang 的很多参数名/默认值随版本漂移较大。下表以 v0.5.18+ 为基线，**凡标 ⚠ 的参数，语义与 vLLM 不完全等价，迁移时必须重读官方文档**，切勿直接照抄数值。

### 11.4.1 启动参数逐项对照表

| 类别 | vLLM (V1) | SGLang (v0.5.18+) | 备注 |
|---|---|---|---|
| 模型路径 | `--model` | `--model-path` | 名字不同，语义同 |
| 服务地址 | `--host` / `--port` | `--host` / `--port` | 同 |
| 服务模型名 | `--served-model-name` | `--served-model-name` | 同 |
| 张量并行 | `--tensor-parallel-size` | `--tp` | ⚠ 短名，数值语义同 |
| 数据并行 | `--data-parallel-size` | `--dp` | ⚠ 短名；SGLang 还配合 `--enable-dp-attention`（见 9.5） |
| 流水并行 | `--pipeline-parallel-size` | 通常并入 `--tp`/专家并行 | ⚠ SGLang 多卡主要靠 TP + EP，PP 用法较少 |
| 上下文长度 | `--max-model-len` | `--context-length` | ⚠ 名字不同 |
| 最大批 token | `--max-num-batched-tokens` | `--max-total-tokens` | ⚠ 语义相近：KV 池/批内 token 上限 |
| 最大并发序列 | `--max-num-seqs` | `--max-running-requests` | ⚠ 名字不同 |
| 显存占用 | `--gpu-memory-utilization` | `--mem-fraction-static` | ⚠⚠ **语义不同**：见 9.5 |
| KV 数据类型 | `--kv-cache-dtype` | `--kv-cache-dtype` | 同（如 fp8、fp8_e5m2） |
| 权重 dtype | `--dtype` | `--dtype` | 同 |
| 量化 | `--quantization` | `--quantization` | 同；取值集合略有差异 |
| 分块预填充 | `--enable-chunked-prefill` + 间接由 `--max-num-batched-tokens` 控制 | `--chunked-prefill-size` | ⚠ SGLang 直接给 chunk 大小 |
| 前缀缓存 | `--enable-prefix-caching`（V1 默认开） | `--enable-prefix-caching`（默认开）/ `--disable-prefix-caching` | 行为接近，粒度/路由语义不同 |
| 投机解码 | `--speculative-config '{"method":...}'` | `--speculative-algorithm` + `--speculative-draft` | ⚠ 配置结构不同，见 9.3 |
| 结构化输出 | 依赖 xgrammar/guidance 后端 | 原生 xgrammar（默认启用） | 概念通用，见 9.2 |
| 日志/缓存报告 | `--enable-prefix-caching` 配合 metrics | `--enable-cache-report` | ⚠ SGLang 专用缓存命中报告开关 |

> 附录 B（参数速查表）会给出"可打印完整版 + 场景化推荐配置模板"，本节只放迁移必看的差异项。

### 11.4.2 代码迁移成本小结

把迁移拆成两层看，结论很不一样：

- **OpenAI SDK 层：零改动。** 客户端只换 `base_url` 和启动命令（9.2.1）。这是绝大多数业务系统的现状，迁移成本 ≈ 改两行配置。
- **Engine / Runtime API 层：需重写。** 如果你在 vLLM 里用了 `LLM(...)` + `SamplingParams` + `llm.generate(...)` 的进程内写法，切到 SGLang 要改成 `sgl.Engine(...)` + `engine.generate(...)`，而且 `SamplingParams` 的字段名（`temperature` / `max_tokens` 等）在 SGLang 里叫 `sampling_params={"max_new_tokens": ...}`（⚠ 注意 `max_tokens` vs `max_new_tokens` 的命名差异）。此外若用了 vLLM 的 `RequestOutput` 字段，SGLang 返回结构是 `dict`，字段名需逐一核对。
- **DSL 层：SGLang 独占。** vLLM 没有等价物，只有当你确实要用 fork/join 这类结构时才写，否则别为了"用 DSL"而用 DSL。

> **什么场景值得用 / 不值得用**
> - 值得：已有 OpenAI 兼容服务，想快速验证 SGLang 前缀缓存收益——先只换 Server，业务零动。
> - 不值得：业务深度耦合 vLLM 的 `LLM` 类内部 API（自定义调度、特定 `RequestOutput` 解析）且短期不打算重写时，强行迁移收益小于重构成本。

---

## 11.5 常见启动参数精解

挑六个最常用、也最容易调错的参数，每个给「默认值 → 影响什么 → 怎么调 → 调错的后果」四要素。版本基线 v0.5.18+，**具体数值请以官方文档为准**。

### 11.5.1 `--tp`（Tensor Parallel，张量并行）

- **默认值**：`1`（单卡）。
- **影响什么**：把模型权重按层内张量切到 N 张卡，解决"单卡放不下"和"单卡算得慢"。
- **怎么调**：模型显存装不下就加；`--tp 8` 表示 8 卡切一份权重。要求卡间高带宽（NVLink 最优），跨机 TP 代价极高。
- **调错的后果**：TP 跨节点（如 4 机各 2 卡做 tp=8）→ 节点间通信拖死吞吐，比单机 TP 慢数倍；TP 设得比卡数大 → 启动直接报错。

### 11.5.2 `--dp`（Data Parallel，数据并行）

- **默认值**：`1`。
- **影响什么**：起 N 份完整模型副本，各自独立处理不同请求，总吞吐随副本数近似线性提升。SGLang 里常配 `--enable-dp-attention`（见 9.5）做注意力协同。
- **怎么调**：吞吐不够且显存有余裕时，`--dp N` 扩副本。多卡机器上常见组合 `--tp 4 --dp 2`（8 卡 = 2 份各 4 卡副本）。
- **调错的后果**：`--tp * --dp` 超过物理卡数 → OOM 或启动失败；和 `--enable-dp-attention` 不配套时命中率稀释（见 9.5）。

### 11.5.3 `--mem-fraction-static`（静态显存占比）⚠

- **默认值**：`0.88`（撰写基线 v0.5.18+，以官方文档为准）。
- **影响什么**：留给"权重 + KV Cache"的静态显存比例，余下给激活值等动态开销。**语义与 vLLM 的 `gpu-memory-utilization` 不同**：后者是"HBM 总占用上限"，前者是"静态预留比例"，激活值另算。
- **怎么调**：默认 0.88 通常安全。长上下文、KV 占大头时适当上调（如 0.9）；遇到激活值 OOM / CUDA 报错时下调（如 0.8）。
- **调错的后果**：直接套用 vLLM 的习惯填 `0.95` → 激活值没空间，decode 阶段 OOM；填太低 → KV 池过小，并发上不去。

### 11.5.4 `--chunked-prefill-size`（分块预填充大小）

- **默认值**：`8192`（撰写基线 v0.5.18+，以官方文档为准）。
- **影响什么**：一次 prefill 最多处理多少 token，超出则切块与 decode 混合组批。控制"长 prompt 会不会卡住 decode"。
- **怎么调**：长 prompt 多、对 TPOT 抖动敏感 → 调小（如 4096）；短 prompt 高吞吐 → 调大（如 16384）提升 prefill 效率。
- **调错的后果**：设太大 → 长 prefill 霸占 GPU，decode 被饿、TPOT 尖刺；设太小 → prefill 步数变多、TTFT 变长、GPU 利用率下降。

### 11.5.5 `--max-running-requests`（最大并发请求数）

- **默认值**：`0`（0 表示自动推导，由显存/KV 容量推算上限；撰写基线以官方文档为准）。
- **影响什么**：同时驻留引擎的最大请求数，直接决定排队与并发上限。
- **怎么调**：想压极限吞吐可调大；要保延迟（防队头阻塞）可调小。设具体值前先用默认跑基线，再按 KV 余量加。
- **调错的后果**：设过大 → KV 耗尽触发抢占/换出，长尾 P99 爆炸；设过小 → GPU 喂不饱，吞吐亏。

### 11.5.6 `--enable-prefix-caching`（前缀缓存，默认开启）

- **默认值**：**开启**（SGLang 默认即 RadixAttention 前缀缓存生效；可用 `--disable-prefix-caching` 关掉）。
- **影响什么**：是否复用跨请求相同前缀的 KV Cache。这是 SGLang 相对"朴素 vLLM 老版本"最大的红利来源。
- **怎么调**：**通常不用动**，保持默认开。调试缓存行为时可用 `--enable-cache-report` 打印命中统计（见 9.1）。
- **调错的后果**：手滑加 `--disable-prefix-caching` → 多轮对话/RAG/共享 system prompt 场景的重复 prefill 全部浪费，TTFT 与吞吐明显劣化；或者 prompt 本身带随机前缀（时间戳等）却开着 → 命中率本就 0，开着也白开，需先修 prompt（见 9.3.2 / 4.6 节）。

> **什么场景值得用 / 不值得用**
> - 值得：默认配置先跑通基线，再针对"长上下文/高并发/多分支"场景微调上述参数。
> - 不值得：一上来就照抄某个博客的"最优参数"。这些参数彼此耦合（kv 容量 ↔ 并发 ↔ 延迟），别人的最优不是你的最优——先测基线，再单点变量（方法见第 10.1 节）。

---

## 本章小结

1. **迁移成本分层**：OpenAI SDK 层零改动（只换 `base_url` 和启动命令）；vLLM `LLM` 进程内 API 需重写为 `sgl.Engine`；DSL 是 SGLang 独占能力，按需采用。
2. **三种使用方式**：OpenAI 兼容 Server（最顺）、`sgl.Engine` 进程内引擎（离线批处理 / RL rollout）、前端 DSL（结构化程序）。按"愿不愿动代码"选。
3. **DSL 的核心不是魔法**：它和 vLLM 自动前缀缓存命中条件相同（token 前缀一致），差异在于把"前缀是否相同"从运气变成可用 `fork` 显式控制的结构。
4. **参数名大量不同但语义可映射**：`--tensor-parallel-size→--tp`、`--max-num-seqs→--max-running-requests`、`--gpu-memory-utilization→--mem-fraction-static` 等，迁移照 9.4 表逐条核对。
5. **`--mem-fraction-static` 语义不同于 vLLM**：它是静态预留比例而非 HBM 总占用上限，直接套用 vLLM 数值会 OOM。
6. **前缀缓存默认开启**：多数场景保持默认即可；缓存收益的前提是 prompt 真的有可复用前缀，否则先修 prompt 而非调缓存。
