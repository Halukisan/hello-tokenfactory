# 第 4 章　KV Cache 与注意力工程化

## 4.1 KV Cache 的数学原理

![KV Cache 沿上下文长度不断增长：多一个请求，就多一份随长度线性膨胀的显存账单](./assets/ch04-kv-cache-growth.png)

这一节的目标只有一个：让你拿到任何一个模型的 config.json，都能手算出它的 KV Cache 占多少显存。

### 4.1.1 自回归解码的重复计算问题

自回归生成的过程是：每生成一个新 token，都要把之前所有 token 的注意力算一遍。朴素实现下，第 `t` 步要计算 attention 时，需要所有前 `t-1` 个 token 的 Key 和 Value。如果每次都重新算，代价是 `O(t²)`——序列越长，浪费越夸张。

KV Cache 的解法很直接：把每一层、每个历史 token 算出来的 K 和 V 存下来，只算新 token 的 Q，然后用新 Q 去和缓存的 K/V 做注意力。这样每步的注意力复杂度从 `O(t²)` 降到 `O(t)`。

注意这里有个和第 1 章（1.3 节）呼应的关键点：decode 每步生成 1 个 token，都要把整份 KV Cache 从 HBM 读进计算单元一次。这正说明 decode 是 memory-bandwidth-bound——瓶颈不在于「算得多快」，而在于「KV 数据从显存搬到计算单元的带宽够不够」。

但代价是显存：这些 K 和 V 必须一直留在显存里，直到这个请求结束。KV Cache 不是「优化技巧」，它是自回归解码为了换 `O(t)` 而**必须付出的存储成本**。

> **解决什么瓶颈**：把 decode 每步注意力从 `O(t²)` 降到 `O(t)`，否则长序列根本跑不动。
> **代价是什么**：KV Cache 必须常驻显存，且随序列长度线性增长——这是后面所有显存管理问题的根源。
> **什么情况下不该用**：几乎没有。这是自回归的基本盘。

### 4.1.2 显存公式

单个请求、序列长度为 `seq_len` 时，KV Cache 占用的显存字节数为：

```
KV_bytes = 2 × layers × kv_heads × head_dim × seq_len × dtype_bytes
```

**说明**：公式里用的是 `kv_heads`（config 的 `num_key_value_heads`）而不是 `num_attention_heads`。`2` 是 K 和 V 各一份，`layers` 取 `num_hidden_layers`，`dtype_bytes` 为 BF16=2、FP16=2、FP8=1。在 MHA 里 `kv_heads` 与 Q 头数相等，但在 GQA/MQA 里远小于 Q 头数——这正是 GQA 能省 KV 的原因。

### 4.1.3 手算 Qwen3-8B

以本书的演示主力模型 **Qwen3-8B**（config.json 已核实）为例，它的关键参数：

- `num_hidden_layers = 36`
- `num_attention_heads = 32`（Q 头）
- `num_key_value_heads = 8`（KV 头，GQA）
- `head_dim = 128`
- `torch_dtype = bfloat16` → `dtype_bytes = 2`

**每个 token 的 KV Cache**：

```
2 × 36 × 8 × 128 × 2 = 147,456 字节 ≈ 144 KiB / token
```

**单条请求、不同上下文长度下的 KV Cache 总量**（BF16）：

| 上下文 | `seq_len` | 单请求 KV 总量 | 备注 |
|---|---|---|---|
| 短对话 | 4,096 | 2 × 36 × 8 × 128 × 4096 × 2 ≈ 576 MiB | 占 8B 权重（~16 GiB）的约 3.6% |
| 中等 | 32,768 (32K) | ≈ 4.6 GiB | 已经超过单卡可轻松容纳的零头 |
| 长上下文 | 131,072 (128K) | ≈ 18.4 GiB | **反超 16 GiB 的权重本身** |

最后一行是重点：**在 128K 上下文下，单条请求的 KV Cache（18.4 GiB）已经超过了模型权重（16 GiB）。** 这就是第 0 章说的「长上下文下 KV Cache 反超权重成为显存主体」的由来。

### 4.1.4 手算最大并发：一个可复现的例题

以 **A800/H20 80GB** 跑 **Qwen3-8B** 为例：权重约 **16 GiB**；假设 `gpu-memory-utilization=0.9`，扣权重与激活/碎片后给 KV 约 **55 GiB**；单请求每 token KV = 144 KiB。

| 上下文长度 | 单请求 KV 总量 | 同卡理论最大并发（55 GiB ÷ 单请求） |
|---|---|---|
| 4,096 (4K) | ≈ 576 MiB | ≈ 55 GiB / 0.56 GiB ≈ **98 条** |
| 32,768 (32K) | ≈ 4.6 GiB | ≈ 55 GiB / 4.6 GiB ≈ **12 条** |
| 131,072 (128K) | ≈ 18.4 GiB | ≈ 55 GiB / 18.4 GiB ≈ **3 条** |

这张表就是调 `max-num-seqs`（vLLM，适用于 v0.20.x–v0.21.x）/ `--max-running-requests`（SGLang，适用于 v0.5.18+）的底层依据：**你设的并发上限，永远不能让「并发 × 单请求 KV」超过 KV 预算，否则就是 OOM 或抢占**。同一张卡，4K 上下文能扛约 98 条并发，128K 只能扛约 3 条，差了 30 多倍。

## 4.2 注意力变体：KV Cache 压缩的第一性原理

![注意力变体的演进：MHA → MQA → GQA → MLA，本质是让多个头共享同一份 KV](./assets/ch04-attention-variants.png)

4.1 节我们看到，KV Cache 大小由 `kv_heads × head_dim × layers` 决定（序列长度和 dtype 是外部变量）。要压缩 KV，工程上只有两条路：**减少 KV 头数**（MHA→MQA→GQA），或**对 KV 做低秩压缩**（MLA）。

### 4.2.1 MHA → MQA → GQA：减少头数

**MHA（Multi-Head Attention，原始多头）**：每个 Q 头配一个独立的 K/V 头，`kv_heads = num_attention_heads`。效果最好，但 KV 最大。

**MQA（Multi-Query Attention）**：所有 Q 头**共享同一份** K/V，`kv_heads = 1`。KV 降到 MHA 的 `1/num_heads`。例如 32 头模型直接降到 1/32。代价是质量下降明显——单个 KV 头成了信息瓶颈。

**GQA（Grouped-Query Attention）**：折中方案。把 Q 头分成若干组，每组共享一个 K/V 头，比如 32 个 Q 头配 8 个 KV 头（`kv_heads=8`）。KV 降到 MHA 的 1/4，而质量几乎与 MHA 持平——这是当前 8B~70B 稠密模型的事实标准。

关于 GQA 有一个容易被忽略的工程细节：**分组数不是任意的**。Q 头数必须能被 `kv_heads` 整除，否则分组不均。Qwen3-8B 是 32/8=每 4 个 Q 头共享 1 个 KV 头。`kv_heads` 越小越省，但太小（趋近 MQA）质量会掉——1/4 到 1/8 是工程上反复验证过的「甜区」。

> **为什么 GQA 这么香**：它把「KV 头数」从「必须 = Q 头数」这个硬约束里解放出来。Q 头多保证表达能力，KV 头少控制显存，两者解耦。代价极小，所以几乎成为默认选择。

### 4.2.2 MLA（Multi-head Latent Attention，DeepSeek）：低秩压缩

GQA 再怎么压，KV 头还是完整维度 `head_dim` 的向量。DeepSeek 系列（V2/V3/R1）用的 **MLA** 走了一条不同的路：**对 KV 做低秩压缩，缓存的不是完整 K/V，而是一个压缩后的潜变量。**

**核心思路**（具体维度以 DeepSeek-V3 技术报告为准）：

- 把原本每个 token 需要缓存的 K 和 V，先通过降维矩阵 `W^{DKV}` 压成一个低维潜变量 `c_KV = W^{DKV} · h`，`c_KV ∈ R^{d_c}`。
- 真正做注意力时，再用升维矩阵 `W^{UK}`、`W^{UV}` 把 `c_KV` 还原成这一层需要的 K 和 V（`K = W^{UK} · c_KV`，`V = W^{UV} · c_KV`）。注意 **K 和 V 共享同一个 `c_KV`**——这是 MLA 比「分别压缩 K 和 V」更省的根本原因：缓存一份，重建两份。
- 为了兼容旋转位置编码（RoPE），DeepSeek 把 RoPE 部分「解耦」出来：位置相关的那部分 key 单独缓存一个小向量 `k^R ∈ R^{d_R}`，不参与低秩压缩。

以 **DeepSeek-V3** 为例（据 DeepSeek-V3 技术报告）：`hidden_size=7168`，`num_attention_heads=128`，`head_dim=128`。

- 若用 MHA，`kv_heads=128`，每 token KV = `2 × 128 × 128 × 2 字节 = 65,536 字节 ≈ 64 KiB`。
- 用 MLA，缓存的是 `c_KV`（维度 `d_c=512`）+ 解耦 RoPE 的 `k^R`（维度 `d_R=64`），共 `512+64=576` 个数，×2 字节 = **1,152 字节 ≈ 1.125 KiB / token**；压缩比 ≈ `65536 / 1152 ≈ 1/57`。

> **MLA 解决什么瓶颈**：在保持（甚至超过）MHA 质量的前提下，把 KV Cache 压到极致，使超大模型和超长上下文的在线服务成为可能。
> **代价是什么**：必须在**训练阶段**就设计低秩结构，无法事后对现成 MHA 模型零成本套用；推理侧需要专门的 MLA 注意力 kernel（vLLM / SGLang 都为此做了专门后端）。
> **什么情况下不该用**：如果你用的是现成的 MHA/GQA 稠密模型（如 Qwen3、Llama-3），MLA 与你无关——你只能走 GQA + KV 量化（4.7 节）的路。

### 4.2.3 对照表

把四种变体放进同一张账上：

| 变体 | KV 头数 | 相对 MHA 的 KV 倍数 | 省了什么 | 代价 / 风险 | 典型代表 |
|---|---|---|---|---|---|
| **MHA** | = Q 头（如 32） | 1× | 基准，质量上限 | KV 最大 | 早期模型、部分小模型 |
| **MQA** | 1 | 1/32 | KV 极小 | 质量下降明显，单头成瓶颈 | PaLM（早期） |
| **GQA** | 8（分组） | 1/4 | KV 小且质量近 MHA | 极少见显著劣化 | Qwen3、Llama-3、Gemma |
| **MLA** | 低秩潜变量 | ~1/57（相对 128 头 MHA） | 极致压缩且质量不降 | 需训练期设计 + 专用 kernel | DeepSeek-V2/V3/R1 |

> GQA 是「当下最划算的默认值」——它用几乎可忽略的代价换来了 4× 的 KV 节省，所以你会看到 2024 年后的开源主力模型几乎全员 GQA。

## 4.3 PagedAttention（vLLM）

4.1 节算清楚了 KV Cache 是笔多大的账。4.2 节讲了怎么从模型结构层面压缩它。但还有一个最朴素的问题没解决：**这些 KV 在显存里怎么摆？**

### 4.3.1 朴素管理的浪费

第一代（HuggingFace `transformers`）朴素推理的做法：按 `max_length` **一次性预留一整块连续显存**。问题有三个：

1. **内部碎片**：实际只生成 200 token，却按 2048 预留，浪费 ~90%。
2. **外部碎片**：不同请求预留不同长度，跑着跑着显存被切得七零八落，明明总量够却分配不出连续块。
3. **无法共享前缀**：两个请求共享同一个 system prompt，KV 算两遍、存两份。

第 0 章给过数字：朴素管理下 KV Cache 利用率常只有 **20%–40%**。

PagedAttention 的解法正是把粒度从「整条序列」降到「16 token 的块」：**块够小，拼接就能凑出任意长度，外部碎片趋近于零；按需分配，内部碎片最多浪费不到一个块（≤15/16 ≈ 94% 利用率）**。

### 4.3.2 从操作系统分页借来的思想

PagedAttention 的妙处在于：**它几乎原样抄了操作系统的虚拟内存分页**。

- 把 KV Cache 切成固定大小的**块**（block，vLLM 默认 16 个 token/块；适用于 v0.20.x–v0.21.x）。
- 用一个**块表**（block table）记录「第 i 个逻辑块 → 第 j 个物理块」的映射，逻辑上连续的序列，物理上可以散落在任意不连续的块里。
- **按需分配**：请求每生成满一个块，才去向显存池要下一块，不预留。
- **写时复制（Copy-on-Write, CoW）**：两个请求共享前缀时，物理块只存一份；只有当某个请求要修改时，才复制出独立块。

一个块表的直观示意：

```
请求 A（逻辑块 0,1,2）           请求 B（逻辑块 0,1,2,3）
逻辑块 0 ──→ 物理块 #5          逻辑块 0 ──→ 物理块 #5   ← 与 A 共享
逻辑块 1 ──→ 物理块 #9          逻辑块 1 ──→ 物理块 #9   ← 与 A 共享
逻辑块 2 ──→ 物理块 #12         逻辑块 2 ──→ 物理块 #12  ← 与 A 共享
                               逻辑块 3 ──→ 物理块 #20   ← B 独有（分叉后）
```

### 4.3.3 解决了什么，代价是什么

**解决**：KV Cache 利用率从 20%–40% 提升到 **90%–95%**。利用率每提升一点，就能多塞请求，batch 变大，吞吐随之上升。

**代价**：**一层间接寻址开销**。注意力计算时，GPU 不能像连续内存那样直接寻址 KV，得先查块表拼出逻辑连续的 KV 视图。这要求专门的 PagedAttention kernel，并且每次注意力都要多一步「按块 gather / 拼接」的索引操作。

> **什么情况下不该用分页 / 该警惕**：极短序列（每个请求只占一两个块）、超小 batch 的离线场景，分页的间接开销占比会上升，且碎片问题本就不严重。但现实是：vLLM 默认就开 PagedAttention，你几乎不需要关。

### 4.3.4 动手：mini PagedAttention 块管理模拟

下面给出一个约 200 行的 Python 模拟器，把「块分配 / 块表 / CoW 共享 / 可视化」跑起来。完整可运行代码在 `src/ch04-kvcache/paged_attention_sim.py`，这里给核心片段：

```python
# paged_attention_sim.py（核心片段，完整版见 src/ch04-kvcache/）
# 验证基线：Python 3.10+，无需 GPU，纯内存模拟
BLOCK_SIZE = 16          # 每个块容纳的 token 数（vLLM 默认 16，可改）
TOTAL_BLOCKS = 64        # 模拟显存池总块数

class PagedKVManager:
    def __init__(self, total_blocks, block_size=BLOCK_SIZE):
        self.block_size = block_size
        self.free_blocks = list(range(total_blocks))  # 空闲物理块列表（栈）
        self.phys_refcount = [0] * total_blocks       # 物理块引用计数（CoW 共享用）
        self.block_tables = {}  # req_id -> [物理块Id,...] 逻辑→物理映射

    def allocate_block(self):
        """按需分配一个物理块，没有就返回 None（模拟 OOM）"""
        if not self.free_blocks:
            return None
        b = self.free_blocks.pop()
        self.phys_refcount[b] = 1
        return b

    def append_token(self, req_id, token_id):
        """向某请求追加一个 token：当前逻辑块满了就分配新块"""
        tbl = self.block_tables[req_id]
        last_block = tbl[-1]
        used_in_block = self._count_in_block(req_id) % self.block_size
        if used_in_block == 0 and len(tbl) > 0:
            # 当前块满，分配新块
            nb = self.allocate_block()
            if nb is None:
                raise RuntimeError("OOM: 显存池耗尽")
            tbl.append(nb)
        # （真实引擎在这里把 KV 写入物理块；模拟器只记账）
        self._inc_count(req_id)

    def fork_shared_prefix(self, parent_id, child_id):
        """CoW：子请求继承父请求的全部物理块，引用计数 +1"""
        self.block_tables[child_id] = list(self.block_tables[parent_id])
        for b in self.block_tables[child_id]:
            self.phys_refcount[b] += 1

    def release(self, req_id):
        """请求结束，释放其块表里的所有物理块（引用减到 0 才真回收）"""
        for b in self.block_tables.pop(req_id):
            self.phys_refcount[b] -= 1
            if self.phys_refcount[b] == 0:
                self.free_blocks.append(b)
```

运行 `python paged_attention_sim.py` 会打印显存池从空到满、CoW 共享、释放回收的全过程。

## 4.4 RadixAttention（SGLang）

PagedAttention 解决了「单条请求内部 KV 怎么摆」。但它有个盲区：**请求与请求之间的重复，它管不了**。RadixAttention 就是来补这个盲区的。

### 4.4.1 核心数据结构：基数树

SGLang 用一棵**基数树（Radix Tree）**管理所有请求的 KV Cache。树的每条边是一段 token 序列，每个节点挂着这段序列对应的**块索引**。

新请求进来时，引擎把它的 token 序列在树上做**最长前缀匹配**：命中的部分直接复用已缓存的 KV（跳过 prefill），只有没命中的后缀需要真正计算。举一个追踪示例：

```
初始：树为空
请求1: [系统设定][你是助手][提问A]   → 树新增一条链，分配物理块 #1 #2 #3
请求2: [系统设定][你是助手][提问B]   → match 命中 [系统设定][你是助手]，
                                        直接复用物理块 #1 #2，仅 prefill [提问B] 并分配 #4
请求3: [系统设定][你是翻译]           → match 在 [系统设定] 处分叉 →
                                        split 把 [系统设定][你是助手] 拆成
                                        [系统设定] 共享 + [你是助手] 分支，
                                        新插入 [你是翻译] 节点并分配 #5
```

mermaid 示意（节点标注其指向的物理块）：

```mermaid
graph TD
    R((根)) --> N1["[系统设定] → 块#1"]
    N1 --> N2["[你是助手] → 块#2"]
    N2 --> N3A["[提问A] → 块#3"]
    N2 --> N3B["[提问B] → 块#4"]
    N1 --> N4["[你是翻译] → 块#5"]
```

### 4.4.2 LRU + 引用计数锁定

淘汰策略是 **LRU（最近最少使用）**，但有个硬约束：**引用计数锁（`lock_ref`）**。一个节点如果正被某个活跃请求依赖，`lock_ref > 0`，即便它是 LRU 也不能动——这避免了「正在用的 KV 被误淘汰」的灾难。

> **RadixAttention 解决什么瓶颈**：消除了**跨请求的重复 prefill**——同一份 system prompt、同一批 RAG 文档块、多轮对话里完全相同的历史轮次，只算一次、只存一份。
> **代价是什么**：① 树结构的维护开销（每次请求都要 match/split/insert，CPU 侧成本）；② LRU 淘汰的「命中率波动」——工作集超过缓存容量时会抖动（见 4.6 节陷阱三）。
> **什么情况下不该用 / 收益为 0**：如果你的负载里**每条 prompt 都独一无二**，命中率趋近 0，维护开销纯属浪费。

### 4.4.3 命中率能到多少

命中率**完全取决于你的负载有多相似**，不存在一个固定数字：

- **高度重复负载**（固定 system prompt、RAG 共享文档块、多轮对话）：第 0 章 0.5.2 节提到，生产环境前缀缓存命中率在 **50%–75%** 区间（据公开报告）。
- **长尾 / 大工作集**：工作集超过缓存容量时，命中率会随淘汰抖动而显著低于「理想值」。

> 不要听信「开 RadixAttention 吞吐翻几倍」的笼统说法。**它的收益是一个关于负载相似度的函数，不是关于引擎的常数。**

### 4.4.4 动手：观测 `cached_tokens`，构造命中/未命中对照

SGLang 提供了 `--enable-cache-report`（默认开启前缀缓存；适用于 v0.5.18+）来观测命中情况。下面给核心片段，完整代码在 `src/ch04-kvcache/cache_hit_experiment.py`：

```python
# cache_hit_experiment.py（核心片段，完整版见 src/ch04-kvcache/）
# 验证基线：SGLang >= v0.5.18，单卡可跑；需先启动 server
# 启动：python -m sglang.launch_server --model-path Qwen/Qwen3-8B \
#        --enable-prefix-caching --enable-cache-report
import openai, time

client = openai.Client(base_url="http://localhost:30000/v1", api_key="EMPTY")
SHARED_SYS = "你是一个严谨的财务分析助手，请基于以下准则回答："  # 稳定的共享前缀

def ask(prompt_suffix, tag):
    resp = client.chat.completions.create(
        model="Qwen3-8B",
        messages=[{"role": "system", "content": SHARED_SYS + prompt_suffix}],
        max_tokens=64,
        extra_body={"return_cached_tokens": True},  # 让响应带回命中信息
    )
    # SGLang 在 usage 中回报 cached_tokens（具体字段名以版本为准）
    usage = resp.usage
    cached = getattr(usage, "cached_tokens", None) or getattr(usage, "prompt_tokens_details", {}).get("cached_tokens", 0)
    print(f"[{tag}] prompt_tokens={usage.prompt_tokens} cached_tokens={cached}")

# 负载 A：共享同一前缀（应命中）
for i in range(5):
    ask(f"问题{i}: 分析一下公司现金流。", "共享前缀")
# 负载 B：每条完全随机（应不命中）
import random, string
for i in range(5):
    noise = "".join(random.choices(string.ascii_letters, k=200))
    ask(noise, "随机前缀")
```

**预期结果**：负载 A 的第二次起 `cached_tokens` 显著大于 0，负载 B 的 `cached_tokens` 始终为 0。

## 4.5 正本清源：分页与前缀树不是竞争关系

![分页（左）管的是「块怎么放」，前缀树（右）管的是「前缀怎么复用」——它们不是竞争关系](./assets/ch04-paged-vs-radix.png)

### 4.5.1 一个流传甚广的误解

很多文章把 PagedAttention 和 RadixAttention 写成「两种路线之争」，仿佛「用了 vLLM 就没有前缀缓存，用了 SGLang 就没有分页」。这是**错的**。

事实是：**RadixAttention 是建立在 PagedAttention 之上的**。SGLang 的 KV Cache 底层同样是分页管理的，只是在这层之上**额外**架了一棵基数树来做跨请求的前缀复用。树里每个节点挂的不是 KV 数据，而是「这段 token 对应的物理块索引」。反过来也一样：**vLLM 也有 prefix caching**，通过 `--enable-prefix-caching` 开启（Automatic Prefix Cache，APC），实现同样是「对请求前缀做哈希/匹配，复用已缓存的 KV 块」。换句话说：

> **分页管「单条请求内部 KV 怎么存」，前缀树管「请求之间 KV 怎么复用」。它们是上下层，不是替代品。**

### 4.5.2 真正的差异在哪里

既然两层技术两边都有，那 vLLM 和 SGLang 在 KV 管理上的**真实差异**是什么？三个维度：

| 维度 | vLLM | SGLang |
|---|---|---|
| **默认策略** | 前缀缓存需显式开启（不同版本默认不同，建议显式设 `--enable-prefix-caching`）；分页默认开 | 前缀缓存**默认开启**；分页默认开 |
| **淘汰算法** | prefix cache 的淘汰策略随版本演进，可通过配置调整；以官方文档为准 | 基数树 LRU + 引用计数锁定（`lock_ref`），树结构天然支持前缀共享与分裂 |
| **工程成熟度 / 生态** | prefix caching 在 V1 引擎中持续打磨，与连续批处理、投机解码深度集成 | RadixAttention 从设计之初就是一等公民，配合前端 DSL 能保证前缀「必然共享」而非「碰运气」 |

> 选 vLLM 还是 SGLang，**不该**用「谁有前缀缓存」来分——两边都有。

**两个引擎都是「分页打底 + 前缀复用在上」的两层架构**。

## 4.6 前缀缓存的四个实战陷阱

理论讲完，这一节全是生产血泪。前缀缓存「看起来免费」，但踩错一个坑，命中率直接归零。

### 4.6.1 陷阱一：时间戳 / request_id 注入 system prompt（最常见，最致命）

**现象**：你兴致冲冲开了前缀缓存，发现 `cached_tokens` 永远是 0，最后发现是**在 system prompt 里拼了 `request_id`、时间戳、或者「当前是第 N 次调用」**。

**原理**：前缀缓存要求「从第一个 token 开始，直到分叉点之前的整段序列逐 token 完全一致」。一旦 system prompt 开头就带了每请求不同的内容（时间戳 `2026-09-06 14:32:01`、uuid、客户端 IP），那么**所有请求的前缀从第一 token 起就不一样**，最长公共前缀长度为 0。

**解法**：
- 把稳定的、跨请求相同的内容放在 prompt **最前面**，且保证它**完全不变**。
- 把每请求变化的内容（时间戳、request_id、用户专属上下文）放到**最后**，或者放到 user 轮次里。

> **反模式**：`system = f"你是助手。本次请求ID={uuid4()}，时间={now()}。" + 固定指令` —— 前缀从第一 token 就变，缓存 100% 失效。正确写法：固定指令在前，变量在后；或变量完全不放 system。

用一个对照直观感受：

```python
# ❌ 错误：变量放在 system 最前面，前缀从第一 token 起就不一致
system_bad = f"请求ID={uuid4()} 时间={datetime.now()} 你是严谨的财务助手，请按准则回答。"

# ✅ 正确：稳定前缀放最前，变量内容后置或放进 user 轮次
system_good = "你是严谨的财务助手，请按准则回答。"   # 完全不变
user_turn   = f"请求ID={uuid4()} 时间={datetime.now()} 请分析某公司现金流。"
```

**把「会变的东西」往 prompt 尾部挪，是前缀缓存收益最大化的第一原则。**

### 4.6.2 陷阱二：多副本轮询把命中率稀释成 1/N

**现象**：单实例测命中率 70%，上了 N 个副本 + 轮询负载均衡，整体命中率掉到约 `1/N`。

**解法**：**缓存感知路由（cache-aware routing）**。路由层不再无脑轮询，而是对请求前缀做哈希，把「前缀相同」的请求稳定地路由到同一副本（一致性哈希）。SGLang 提供了 **SGLang Router**（适用于 v0.5.18+）。

假设单副本前缀命中率为 `h`（比如 70%），副本数 `N=4`。随机轮询下，任意一条重复请求落到「正确副本」的概率是 `1/N`，于是**集群整体命中率 ≈ h / N**。代入：`70% / 4 ≈ 17.5%`——你加了 4 倍机器，命中率反而从 70% 掉到不到 18%。

> 这个陷阱揭示了一个反直觉的事实——**水平扩展反而可能让前缀缓存失效**。扩展时务必配套缓存感知路由，否则是在用命中率换吞吐。

### 4.6.3 陷阱三：淘汰抖动（eviction thrashing）

**现象**：缓存容量偏小、工作集偏大时，命中率剧烈波动，长尾请求尤其惨。

**原理**：LRU 淘汰的前提是「缓存容量 ≥ 工作集」。一旦工作集超过可用缓存，节点刚插入就可能被下一次淘汰挤走，复用距离大于缓存容量，命中率塌方。

**解法**：
- 扩大 KV Cache 容量，让缓存 ≥ 工作集。
- 评估是否上 KV 量化（4.7 节）把单位 KV 压小，变相扩大容量。
- 在容量不足的硬约束下，接受部分命中率损失，用监控盯住 P99。

### 4.6.4 陷阱四：vLLM Automatic Prefix Cache 的开关与默认行为差异

**现象**：在 vLLM 里开了服务，前缀缓存似乎「有时有有时没有」。

**原理**：vLLM 的 prefix caching（Automatic Prefix Cache, APC）的**默认开启状态在不同版本间有变化**。

**解法**：
- **显式开启**：在 vLLM 启动命令里明确加 `--enable-prefix-caching`（适用于 v0.20.x–v0.21.x）。
- **确认生效**：通过 Prometheus 指标或日志确认 prefix cache 查询/命中计数在增长。

> **本节一句话总结**：前缀缓存不是「开了就赢」。变量内容别放前缀、副本要缓存感知路由、容量要 cover 工作集、vLLM 要显式开——四条全中，命中率才可能到你负载理论上限。

## 4.7 KV Cache 量化与卸载 

4.1 节算出 128K 上下文下单请求 KV 就可能 18 GiB。当 GQA/MLA 的结构性压缩还不够时，还有两条路：**量化**（用更少 bit 表示每个 KV 元素）和**卸载**（把 KV 挪到 CPU 内存 / SSD / 网络远端）。

### 4.7.1 FP8 KV Cache

把 KV 从 BF16/FP16（2 字节）存成 FP8（1 字节，如 E4M3 / E5M2 格式），**KV 显存直接减半**，代价是注意力计算的精度损失。

- **vLLM**：通过 `--kv-cache-dtype fp8_e5m2` 或 `fp8_e4m3`（适用于 v0.20.x–v0.21.x）。
- **SGLang**：通过 `--kv-cache-dtype fp8`（适用于 v0.5.18+）。

> **代价**：FP8 对注意力的精度敏感，长上下文（注意力分数动态范围大）下可能比短上下文更易出现质量劣化。**什么情况下不该用**：对生成质量极度敏感、且上下文很长、且又不缺显存的场景——此时半点精度都不该省。

### 4.7.2 TurboQuant 2-bit KV Cache

vLLM V1（适用于 v0.21.x+，以官方文档为准）引入了 **TurboQuant**，对 KV Cache 做**每组的 2-bit 量化**（含重建项），把 KV 进一步压到约 1/8（相对 BF16）。

> **代价**：2-bit 是激进压缩，精度损失明显大于 FP8，必须通过业务指标（而非只看吞吐）实测是否可接受。把它当作「显存实在塞不下时的逃生舱」。

### 4.7.3 CPU / SSD 层级卸载

当 GPU 显存放不下全部 KV 时，把不常用的 KV 块卸载（offload）到 CPU 内存甚至 SSD，需要时再搬回 GPU：

- **CPU 卸载**：KV 块存在主机内存，decode 每步按需取回。代价是 PCIe 带宽瓶颈——每步多一次主机↔设备传输，TPOT 上升。
- **SSD / 分布式 KV**：如 **Mooncake** 把 KV 做成集群级共享存储，跨节点复用；**HiSparse** 等思路结合稀疏化减少搬运量。

> **解决什么瓶颈**：让「KV 总量 > GPU 显存」的场景仍然可服务，极端长上下文或超大并发不再直接 OOM。
> **代价是什么**：每一层卸载都引入一次数据搬运延迟，直接抬高 TPOT / 降低有效带宽。层级越深（GPU→CPU→SSD→网络），延迟越大。
> **什么情况下不该用**：命中率低、或 decode 延迟敏感的在线交互场景。**层级卸载的盈亏平衡点，取决于「复用收益」是否大于「搬运成本」**，必须实测。

## 4.8 前沿：新架构对 KV 管理的冲击（2025–2026）

前面的整条叙事都建立在一个假设上：**模型每层都是标准注意力，KV Cache 由 `kv_heads × head_dim × layers` 决定，且随序列长度线性增长**。

### 4.8.1 冲击一：混合注意力——KV 不再是「每层都有」

**代表**：Qwen3-Next（Gated DeltaNet 线性注意力 + 1/4 层全注意力交错）、Kimi Linear（KDA，Kimi Delta Attention）。

**思路**：既然 decode 的瓶颈是「每步把整份 KV 搬一遍」，那就让**大部分层干脆不用 KV Cache**——换成线性注意力 / Delta Rule 类机制。这类层的「状态」是一个固定大小的矩阵（不随序列长度增长），只有少数保留的全注意力层才维护真正的 KV。

**对显存公式的冲击**：公式里的 `layers` 不再等于 `num_hidden_layers`，而是「全注意力层数」（比如 1/4）；线性层那一项变成 `固定状态大小 × 线性层数`，与 `seq_len` 无关。结果是：**长上下文下 KV Cache 从「显存主体」退回「零头」**——Kimi Linear 在 1M 上下文下 KV Cache 约减少 75%。

**工程上的新麻烦**：引擎要同时管理两种状态：线性层的 **conv state + delta rule state** 和全注意力层的传统 KV 块，vLLM 为此引入了 **Hybrid KV Cache Manager**。

### 4.8.2 冲击二：稀疏注意力——KV 可以「选择性读取」

**代表**：DeepSeek-V3.2 的 DSA（DeepSeek Sparse Attention）、NSA（Native Sparse Attention）。

**思路**：标准注意力里每个新 token 都要「看」全部历史 token，但大部分 query 真正需要的只有少数关键 token。DSA 用一个轻量 **lightning indexer** 给每个 query 选出 top-k（如 2048）个 token，注意力只在这些被选中的 KV 上算——**复杂度从 `O(L²)` 降到 `O(L·k)`**。

**关键澄清——稀疏注意力省的和 GQA/量化省的不是同一笔账**：

| 技术 | 省的是什么 | 显存里 KV 还在吗 |
|---|---|---|
| GQA / MLA（4.2 节） | **存储**：每 token 存的 KV 变小 | 变小了，但仍全量存 |
| KV 量化（4.7 节） | **存储**：每个元素的字节变少 | 变小了，但仍全量存 |
| 稀疏注意力（DSA/NSA） | **带宽**：每步注意力只读被选中的 KV | 基本全量存，但每步只读一小部分 |

也就是说，DSA 主要攻击的是「decode 每步搬运 KV」的带宽账，而非显存容量账。DeepSeek 公布的数据是：DSA 使其 128K 上下文的 API 成本下降约 83%（prefill 与 decode 双侧受益）。

### 4.8.3 冲击三：多 token 预测（MTP）——「一层特殊头」改写显存账与调度

**代表**：DeepSeek-V3/R1 的 MTP 层、Qwen3-Next 的 MTP head。

**思路**：在主干之外多训一个（或几个链式的）「预测头」，训练时强迫它在每个位置预测「下下个 token」；推理时这些头当作免费的草稿模型，为投机解码（第 7 章）提供候选——**模型自带的 draft，接受率远高于外挂小模型**。部署时若开启 MTP 投机解码，**草稿层自己也要一份 KV Cache**（通常只覆盖最近几层 / 几个 token，量小但不是零）。

### 4.8.4 给工程师的三条判断

1. **「KV Cache 是显存主体」从铁律变成「全注意力层视角下的局部真理」**。拿到 2025 年后的新模型，第一步看 config 判断架构形态（full / hybrid / sparse / MTP），再对「全注意力层」算账。
2. **本章的工程能力没有贬值，而是换了落点**：分页管理依然管「状态怎么摆」；前缀复用依然管「跨请求怎么省」，只是匹配对象从「纯 KV 块」扩展到「KV 块 + 线性状态快照」；量化对全注意力层的 KV 依然有效。**学的是原则，不是某个架构的参数。**
3. **引擎选型时把「新架构支持度」加进评估表**：同一个混合架构模型，不同引擎、不同版本的支持成熟度差异极大。

> 以上内容以 2026 年 9 月的公开资料为准（Qwen3-Next、Kimi Linear/K3、DeepSeek-V3.2/V4 等），模型与引擎的具体参数、命名以官方文档为准。

## 4.9 本章小结

1. **KV Cache 是自回归为了把注意力从 `O(t²)` 降到 `O(t)` 而必须付出的存储债**，它随序列长度线性增长，长上下文下会反超权重成为显存主体。
2. **显存公式是第一性工具**：`2 × layers × kv_heads × head_dim × seq_len × dtype_bytes`，其中 `kv_heads`（不是 Q 头数）和 `dtype` 是压缩的两个主要旋钮。
3. **注意力变体的本质是压缩 KV**：MHA→MQA→GQA 靠减少 KV 头数，MLA 靠低秩压缩（DeepSeek，约 1/57）；GQA 是当前最划算的默认选择。
4. **PagedAttention 管单请求内部 KV 怎么存**（利用率 20%–40%→90%–95%），代价是一层间接寻址；**RadixAttention 管请求之间 KV 怎么复用**，靠 LRU + 引用计数锁定。两者是上下层、互补而非竞争。
5. **前缀缓存有四个实战陷阱**：变量内容污染 system prompt 前缀、多副本轮询稀释命中率到 1/N、淘汰抖动、vLLM APC 需显式开关——任意一个都能让命中率归零。
