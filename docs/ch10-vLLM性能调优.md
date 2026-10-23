# 第 10 章　vLLM 性能调优

## 本章主线：先方法，后参数

第 0 章那句判断值得复述：**工程复杂度本身是一等公民挑战**。一百多个参数里十来个有决定性影响，且互相耦合——改一个，另一个的最优值就变了。先立住主线，否则每个参数讨论都会沦为「参数字典」：

> **调优的正确顺序 = 定 SLO → 建基线 → 单点变量 → 回归验证。**
> 这不是四个建议，而是四个**不可交换的步骤**。跳步就会得到「别人的参数抄来没用」的结果。

```mermaid
flowchart LR
    A[定 SLO<br/>延迟/吞吐目标] --> B[建基线<br/>可复现快照]
    B --> C[单点变量<br/>一次只动一个]
    C --> D[回归验证<br/>对照台账看 P99]
    D -->|未达标| C
    D -->|达标| E[固化配置]
```

定 SLO 的判据是第 1.2 节的 Goodput；回归验证看 P99 而非均值——均值漂亮、P99 崩了，是线上事故的标准开场。

三种「伪调优」都跳了某一步：*抄作业型*不问负载就搬「黄金配置」（**到了第三代后期（PD 分离、集群化），抄参数基本失效**）；*炼丹型*同时改五六个参数，说不清谁贡献；*指标盲人型*只盯 `tok/s`，越过 Goodput 拐点还以为在进步。

---

## 10.1 建立调优基线 ★★☆ 🖥️

![调优从建立基线开始：先拧对第一个旋钮，再谈其他](./assets/ch10-tuning-dials.png)

### 为什么第一步不是改参数

反模式变体不下十种，根源只有一个：**没有可复现基线就开始改参数**。改 `max-num-batched-tokens`，吞吐 1500 → 1700 像「有效」，但那 200 可能只是泊松抖动；抄来的「黄金配置」在你硬件上更慢，往往是负载形态不同。**基线给你一个可靠的「零号数据点」**，之后每个改动都能在上面加减归因。

### 用官方 benchmark 工具跑基线

两套入口：`vllm bench serve` 对**已启动的 OpenAI 兼容服务**发压，最接近真实部署；`vllm bench generate` 离线直跑，排除服务层。生产调优一律用 `bench serve`（键名随版本略有差异）：

```bash
# 假设 vLLM 服务已在 http://localhost:8000 启动（启动命令见 10.2 节）
vllm bench serve \
  --model Qwen/Qwen3-8B-Instruct \
  --base-url http://localhost:8000 \
  --endpoint /v1/chat/completions \
  --num-prompts 1000 \            # 发多少条请求，太小统计噪声大
  --request-rate 16 \             # 目标到达率（req/s）；inf 表示尽力压满
  --shared-conversation-id 0 \    # 多轮对话前缀复用场景，基线一般先关
  --random-input-len 512 \        # 输入长度分布，贴近你真实流量
  --random-output-len 256 \       # 输出长度分布
  --seed 42 \                     # 固定随机种子，保证可复现
  --metric-percentiles 50,90,99 \ # 必须抓 P99，只盯均值会翻车
  --save-result \                 # 把结果落盘
  --result-filename baseline.json # 基线结果文件，进台账
```

- `--request-rate` 不要用默认 `inf` 起步。`inf` 是「压到饱」，给出**吞吐上限**，不是你能稳定服务的吞吐。早期先用**你能接受的到达率**（如峰值的 80%）看 P99，之后再逐步抬它找拐点（拐点即吞吐不再涨、延迟开始飙的点）。
- `--random-input-len` / `--random-output-len` 必须**贴近真实流量形态**；`--seed` 也必须固定，不同 seed 的 P99 能有可观差距。

**压测工具自身的坑**：客户端发不出足够 QPS 时你测的是客户端（判据：`gpu_util` 还很闲却吞吐上不去）；CUDA Graph 捕获、kernel 编译让前几条延迟暴高，采样前必须先预热。

> 一句话：**先确认你测的是服务端，再谈调优服务端。**

### 记录参数快照：比记结果更重要

只记「吞吐 1700」没用——三个月后你不知道当时的 `--gpu-memory-utilization` 是多少。快照必须含**环境 + 启动参数 + 结果**：

```bash
# baseline_serve.sh —— 基线服务启动（应用于 vLLM v0.20.x–v0.21.x）
vllm serve Qwen/Qwen3-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \   # 基线先用默认 0.90
  --max-model-len 4096 \            # 先用贴近真实流量的上限
  --max-num-seqs 256 \              # V1 默认值随版本变化，以文档为准
  --max-num-batched-tokens 8192 \   # V1 默认值随版本变化，以文档为准
  --enable-prefix-caching \         # 长上下文/多轮场景建议开
  --port 8000
```

vLLM 一升级默认值就会变——把命令当数据存，是跨版本对比的命脉。

### 调优台账：你的调优「实验记录本」

台账模板如下（配套仓库 `src/ch10-vllm-tuning/tuning_log_template.csv`）：

| 实验编号 | 日期 | 改动参数 | 改动值 | 目标 SLO | TTFT P50 | TTFT P99 | TPOT P50 | TPOT P99 | 吞吐(req/s) | 吞吐(tok/s) | Goodput | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 000-baseline | | （无改动） | — | — | | | | | | | | 零号数据点，禁止动 |
| 001 | | `--max-num-batched-tokens` | 4096 | | | | | | | | | 单点变量 |
| 002 | | `--gpu-memory-utilization` | 0.95 | | | | | | | | | 观察是否 OOM |

铁律：① **每次只允许一行改动**；② **000-baseline 永远冻结**；③ **Goodput 必须填**——吞吐涨了但 P99 超时，它就是「虚假收益」（定义见第 1.2 节）；④ 备注记 GPU 型号、驱动、版本、到达率。

**台账怎么读**：001 行把 `--max-num-batched-tokens` 从 8192 改到 16384，吞吐 1500 → 2400，TTFT P99 从 260ms 到 410ms——SLO 是 300ms 就是负优化，500ms 才可固化；002 行把 `--gpu-memory-utilization` 从 0.90 到 0.95 吞吐不动，风险在峰值 OOM，别记成「没用」。

> **反模式小结（8.1）**：① 没有基线就改参数；② 同时改多个变量然后归因到一个；③ 只看均值不看 P99；④ 用 `inf` 压测结果当成「我能稳定服务的吞吐」。这四条占线上推理事故的很大一部分。

---

## 10.2 显存与并发参数调优 ★★☆ 🖥️🖥️

这一节只讲三个决定「能不能跑稳、能跑多大」的旋钮：它们定的是**容量边界**。

### `gpu-memory-utilization`：不是越大越好

这个参数（v0.20.x–v0.21.x，默认约 `0.90`）决定 vLLM 最多吃掉整卡多少显存的权重与 KV Cache。「要榨干硬件就设 0.99」是新手第一大坑——显存还有这些隐性占用：

- **激活值**：每次前向的瞬时中间结果，长序列、大 batch 时突增。
- **CUDA Graph 预留**：开启后预占 graph 缓冲；投机草稿模型权重也额外占显存（见第 7 章）。
- **通信缓冲与固定开销**：多卡 TP/PP 的 NCCL 缓冲、CUDA context、cuDNN 工作区。
- **碎片与抢占 swap**：触发抢占要靠 `swap-space` 兜底，启动期碎片仍需 headroom。

设到 0.95 以上，等于把这些「突发 + 固定」开销的空间压到几乎没有，结果**在峰值的一次激活值尖峰上直接 OOM，服务崩。**

**正确做法**：起步 0.90，长上下文可谨慎到 0.92–0.93；多卡 TP 可适度抬高，但每次只抬 0.02 一档并跑高负载压测。KV 容量不够（`gpu_cache_usage` 长期贴近 1.0 且排队）时，**先确认是不是 `max-model-len` 开太大**。

> **调了会怎样**：抬高 → KV Cache 池变大 → 能攒更多并发、更少排队。
> **调错会怎样**：抬到 0.95+ → 峰值激活值尖峰触发启动期或运行期 OOM，服务不可用。

### KV Cache 可用量 → 最大并发数：换算关系

先把第 1.4 / 第 4.1 节的 KV 显存公式搬过来：

```
每个 token 的 KV Cache 字节数 =
    2 × num_layers × num_kv_heads × head_dim × bytes_per_elem
```

- `2`：K 和 V 两份；`bytes_per_elem`：BF16 = 2 字节，FP8 KV = 1 字节（见 8.2 与第 4.8 节）
- GQA / MQA 里 `num_kv_heads` 远小于 `num_attention_heads`，这正是它们省显存的原因

再估算「KV Cache 池能放多少 token」：

```
KV 池总字节 ≈ 整卡显存 × gpu_memory_utilization × KV_占比
KV 容量(token 数) ≈ KV 池总字节 / 每个 token 的 KV 字节数
```

`KV_占比` 是「去掉权重、激活、碎片后真正留给 KV 的比例」，**不是** `gpu_memory_utilization` 本身。8B 模型在 24GB 卡上权重约占 16GB，真实 KV 占比常只有 20%–35%，**以实测为准**。

```
最大并发数(估算) ≈ KV 容量(token 数) / 平均序列长度
                  = KV 容量 / (平均输入长度 + 平均输出长度)
```

这个值**还要被 `max-num-seqs` 截断**——它是硬上限。并发上不去、请求大量排队时，先看 `gpu_cache_usage` 是否长期 ≈ 1.0；若是，瓶颈在 KV 容量，该降 `max-model-len`、开 FP8 KV。

**动手算一遍（构造示例）**。Qwen3-8B（32 层、GQA）假设 BF16 下每 token 的 KV 约 0.5MB：权重占 16GB；`gpu-memory-utilization = 0.90` → 可用 21.6GB，减权重剩 5.6GB；再减激活与碎片 headroom（保守按一半算），留给 KV 约 2.8GB；KV 容量 ≈ 2.8GB / 0.5MB ≈ 5600 token；平均序列长度 768（输入 512 + 输出 256）时，理论并发 ≈ 5600 / 768 ≈ 7。

只有 7 个并发——这就是「短文本高并发也要靠 KV 量化或更大显存」的由来：开 FP8 KV 翻倍到约 14，序列长度压到 384 再翻倍到约 28。

### `max-num-seqs` 与 OOM 的边界：一个必须澄清的误区

大纲点名要讲这个边界。先纠正一个 widespread 误解：

> **`max-num-seqs` 本身几乎不直接导致启动期 OOM。**

原因在 PagedAttention（第 4.4 节）：KV Cache **按需、按块**分配，不是一次性预留，它只是**并发软上限**。真正决定启动期 OOM 的是 `gpu-memory-utilization`（决定 KV 池大小）与 `max-model-len`（开 128K 而卡只有 24GB，权重+位置编码先把显存吃穿）。

所以 `max-num-seqs` 调很大的后果**不是 OOM，是「排队」和「抢占抖动」**：设太小（如 32）请求在外排队；设太大（如 1024）则 KV 池有限，总 token 逼近容量时触发**抢占（preemption）**，表现为 TPOT 尖刺和 P99 拉长，**不是崩，是抖**。正确调法是算出理论并发再设其 70%–90%。

> **反模式小结（8.2）**：① `gpu-memory-utilization` 拉到 0.99 等 OOM；② 并发上不去就无脑加 `max-num-seqs`，结果换来抢占抖动；③ 忘记 `max-model-len` 才是长上下文 OOM 的真凶。

---

## 10.3 `max-num-batched-tokens` 的权衡 ★★★ 🖥️🖥️

这是本章核心之一，也是整个 vLLM 调优里**最值得花时间扫描**的参数。

### 它在控制什么

`--max-num-batched-tokens`（v0.20.x–v0.21.x，默认约 `8192`）限制**一个调度步里 prefill 与 decode 合在一起的 token 总数上限**，与 `max-num-seqs` 正交：前者管「一步处理多少 token」，后者管「同时飞几个请求」。Chunked Prefill（见 3.2 / 8.5）会把长 prefill 切成 chunk 塞进不同步的预算，所以它实际控制**每一步的拼接规模**。

### 吞吐 ↔ TTFT ↔ TPOT 的三角

**1. 对吞吐**：调大 → 每步塞进更多 token → 算术强度上升 → 吞吐上升，这是主收益。

**2. 对 TTFT**：**反直觉**的一点。调大后新请求要等当前步的预算用掉才能插进去，预算里塞满长 prefill 时 TTFT 反而升高。**大预算有利于吞吐，但牺牲新请求的排队时效。**

**3. 对 TPOT**：单步计算量变大，TPOT 轻微抬升，但敏感度低于 TTFT。

所以这是个**不可能三角**：调大偏吞吐（如 16384、32768），调小偏延迟（如 2048、4096）；默认 8192 是折中，但不等于最优。

```
          吞吐 ↑
            / \
           /   \
  TTFT ↓ ——     —— TPOT ↓
   （调小）        （调小）
```

**从算术强度看为什么调大能提吞吐**（呼应第 1.3 节）：token 少时，计算量摊到固定的「权重读取」成本上很薄，GPU 卡在访存墙（memory-bound）；调大后**算术强度上升**，正是 0.5.3 节「1.7 倍来自系统层」的红利。但每步 token 太多，单步耗时线性增长、边际收益递减，**这就是 32768 处断崖的原因**。

### 长文本 vs 短文本：最优值差异巨大

「抄参数失效」的典型现场：**短文本高并发**（输入 256、输出 256）经验上 4096–8192 更好，重点放 `max-num-seqs`；**长文本低并发**（输入 8K、输出 2K）调到 16384–32768 能让长 prefill 用更少 chunk 完成，TTFT 反而下降。

**结论先行**：没有「通用最优值」，做法是「单点变量 + 扫描」。

### 动手：参数扫描，画出吞吐-延迟帕累托前沿

固定其他参数（回到 000-baseline），只扫 `[2048, 4096, 8192, 16384, 32768]`；每个值跑同一组 `vllm bench serve`，记录 TTFT P99 与吞吐(tok/s)。配套仓库 `src/ch10-vllm-tuning/sweep_batched_tokens.py` 做了自动化，核心逻辑（节选）：

```python
# sweep_batched_tokens.py（节选自 src/ch10-vllm-tuning/，适用于 vLLM v0.20.x–v0.21.x）
# 用法见该目录 README.md；需自行保证显存足够重启服务
CANDIDATES = [2048, 4096, 8192, 16384, 32768]
for mbt in CANDIDATES:
    # 1) 用当前 mbt 重启 vllm serve（脚本内通过子进程调用，参数见文件）
    start_server(max_num_batched_tokens=mbt)
    # 2) 跑同一组固定 seed 的 bench
    result = run_bench(request_rate=ARATE, seed=42)
    # 3) 记录：mbt, TTFT_P99, TPOT_P99, throughput_tok_s, goodput
    log.append({
        "max_num_batched_tokens": mbt,
        "ttft_p99_ms": result.ttft_p99,
        "tpot_p99_ms": result.tpot_p99,
        "throughput_tok_s": result.output_throughput,
        "goodput": result.goodput,
    })
write_csv("sweep_result.csv", log)
```

`sweep_result.csv` 每行是一个候选值的表现。画成图（x 轴 TTFT P99，y 轴吞吐），**右上角那条「不被任何点 dominate」的折线就是帕累托前沿**。下面是**示意性**结果表（数字为构造示例，**以你自己的实测为准**）：

| `max-num-batched-tokens` | TTFT P99 (ms) | 吞吐 (tok/s) | 是否在前沿 |
|---|---|---|---|
| 2048 | 120 | 900 | 否（被 4096 dominate） |
| 4096 | 180 | 1500 | 是 |
| 8192 | 260 | 2050 | 是 |
| 16384 | 410 | 2400 | 是 |
| 32768 | 720 | 2450 | 否（吞吐几乎不涨，延迟暴增） |

- **8192 → 16384**：吞吐还在涨、TTFT 可接受 → 值得往右走。
- **16384 → 32768**：吞吐几乎不动（2450 vs 2400），TTFT 从 410 飙到 720 → 已**越过拐点**，落入「虚假收益区」。
- **最终选型**看 SLO：TTFT P99 < 300ms 时 8192 是唯一合规点；放宽到 500ms 且要最大吞吐，选 16384。

> **调了会怎样**：调大→吞吐升、TTFT/TPOT 升；调小→TTFT 降、吞吐降。
> **调错会怎样**：调太大越过拐点→延迟爆炸而吞吐不再涨，Goodput 崩塌；调太小→吞吐天花板被锁死。

**实操注意**：每组至少跑 3 次取中位数；`--request-rate` 锁死同一值并隔离冷启动。

> **反模式**：① 永远用默认 8192 不扫；② 扫了只看吞吐不看 TTFT，把 32768 当最优。

## 10.4 Chunked Prefill 与长上下文调优 ★★★ 🖥️🖥️

长上下文（32K、128K 乃至 1M）把前面所有权衡重新标定：KV Cache 显存线性增长、注意力计算量二次增长（第 0.7 节挑战二）。

### 长 prompt 场景的调优思路

没有 Chunked Prefill（见 5.2 节）时，一个 32K 的 prefill **独占一步**，期间所有 decode 被饿死，TPOT 出现长尖刺。Chunked Prefill 把它切成 chunk 与 decode 混合组批，代价是单步计算量变大、TPOT 轻微抬升，但消除了「长 prefill 卡死全场」。几个联动动作：

1. **`max-num-batched-tokens` 要调大**（呼应 8.3）。长文本常需 16384–32768，预算太小会把 prefill 切成极多 chunk。
2. **`max-model-len` 按真实上限设，别拍满**。开 128K 而实际只有 8K，位置编码缓冲和块表上限白占显存；设成「真实 P99 长度 × 1.2」通常够。
3. **KV Cache 量化**（`kv-cache-dtype`，精度验证见第 4.8 节）：KV 池占用砍半、并发翻倍，代价是精度。
4. **抢占代价更痛**。长序列被换出的重算成本是短序列的指数级，`max-num-seqs` 要更保守。

**长上下文的显存账（构造示例）**。换成 32K 上下文：平均序列长度从 768 变成 32768，KV 容量仍是约 5600 token——**并发直接从约 7 跌到 0.17**。不是算力不够，是 KV 池被单个长序列吃穿。破解靠：① FP8 KV；② GQA/MLA 从结构上砍 `num_kv_heads`（见第 4.2 节）；③ `max-model-len` 收紧到真实 P99。

**chunk 大小的权衡**。太小 → 长 prefill 被切成几十块，每块都付「调度 + 前向 + 写 KV」的固定开销，TTFT 升高；太大 → 单块压垮一步，decode 被饿死，TPOT 尖刺。粒度由 `max-num-batched-tokens` 与内部策略共同决定。

### 前缀缓存在长上下文下的价值

`--enable-prefix-caching` 在长上下文 + 多轮/RAG/agentic 负载下价值被放大：一个 8K 的 system prompt 被上百个请求共享时，**只有第一个请求付 prefill 全款**。但第 4.7 节的陷阱更致命：

- **system prompt 注入时间戳 / request_id** → 每个请求前缀都不同 → 命中率归零。
- **多副本轮询** → 命中率被稀释为 1/N（见 5.6 节缓存感知路由）。

> **反模式小结（8.4）**：① `max-model-len` 拍满 128K 挤占 KV 池；② 长 prompt 却用小 `max-num-batched-tokens` 被切碎；③ system prompt 注入变量导致前缀缓存失效；④ 长序列抢占风暴。

---

## 10.5 编译与 kernel 优化 ★★★ ⏭️

> 本节标 ⏭️，第一遍通读可跳过，不影响主线理解。回归调优或追求极限性能时再读。

### CUDA Graph 捕获的条件与限制

CUDA Graph（通过 `--enforce-eager` 的**反义**开启；V1 默认尽量捕获）把完整 decode 循环捕获成一张图，**几乎消除逐 step 的 kernel launch 开销**。但变长输入、极端动态控制流、部分自定义算子难以捕获，需「多张 graph + padding」覆盖，捕获耗时与显存随之上升——又回到 8.2 的 headroom 问题。

**何时必须 `--enforce-eager` 回退**：① 启动时报捕获失败或 capture 阶段 OOM——先降 `gpu-memory-utilization` 试；② 开 graph 后输出异常或卡死；③ 负载形状极度动态，重捕获开销反而更大；④ 调试期。

**原则**：它是「默认开启、出事再关」的优化，收益远不如 8.3 的扫描和第 7 章投机解码。

### `torch.compile` 的收益与编译耗时

V1 通过 `--torch-compile-levels`（或等价配置）做算子融合，减少 kernel 数量，但**首次编译有明显耗时**（冷启动变慢，可能触发显存峰值）。把它当作「上线前定一次、然后冻结」的开关，用 A/B 回台账看 P99 与启动耗时再定。

### Attention backend 选择（FlashAttention / FlashInfer / MLA 后端）

V1 引擎通过 `--attention-backend`（v0.20.x–v0.21.x）选择注意力实现：

| backend | 特点 | 适用 |
|---|---|---|
| **FlashAttention**（`FLASH_ATTN`） | 通用、稳定，大多数模型的默认安全选项 | 默认起步 |
| **FlashInfer**（`FLASHINFER`） | 对变长、分页 KV 的融合更优，长上下文常更好 | 长上下文、高并发变长 |
| **MLA 后端**（`MLA_ATTN` / 相关键） | 针对 DeepSeek MLA 的低秩 KV 做专门路径 | DeepSeek 等 MLA 模型 |

选型同样是「回到台账」：同一负载下换 backend 跑 bench 看 TPOT P99 与吞吐。

---

## 10.6 调优方法论总结 ★★☆

![吞吐与延迟的帕累托前沿：最优解在拐点上，不在任何一个极端](./assets/ch10-pareto-frontier.png)

这是全章收尾，也是被引用最多的三张表——**读者可以把它们打印贴在墙上**。

### 表一：参数敏感度排序（优先调哪个）

按影响力从高到低排序，调优时**从上往下**走。

| 排名 | 参数 | 影响力 | 主要影响 | 调优优先级动作 |
|---|---|---|---|---|
| 1 | `max-num-batched-tokens` | ★★★★★ | 吞吐、TTFT、TPOT 三角 | 先扫描出帕累托前沿（8.3） |
| 2 | `gpu-memory-utilization` | ★★★★ | KV 容量、是否 OOM | 默认 0.90 起，长上下文谨慎抬 |
| 3 | `max-num-seqs` | ★★★★ | 并发上限、抢占抖动 | 与 KV 容量公式联动设 70%–90% |
| 4 | `max-model-len` | ★★★ | 长上下文 OOM、KV 池占比 | 按真实 P99 长度 ×1.2 设 |
| 5 | 投机解码配置 | ★★★★（高陷阱） | decode 吞吐、TPOT | 先 N-gram 探路，再看接受率（第 7 章） |
| 6 | `--enable-prefix-caching` | ★★★（依赖负载） | 长上下文/多轮 prefill 成本 | 多轮/RAG/agentic 必开 |
| 7 | attention backend | ★★（依赖形状） | 长上下文 TPOT | 长上下文试 FlashInfer/MLA |
| 8 | `torch.compile` 级别 | ★★ | 特定 shape 吞吐、启动耗时 | 上线前定一次冻结 |
| 9 | `kv-cache-dtype`（FP8） | ★★★（长上下文） | KV 池占用、精度 | 长上下文首选，需验精度 |

> 读法：别从第 9 项开始调。绝大多数「吞吐翻倍」的收益，来自第 1–3 项；第 5 项是「高风险高回报」，放后面单独验证。

### 表二：场景化推荐配置（四种，附完整启动命令）

四套配置基于 vLLM v0.20.x–v0.21.x，模型 Qwen3-8B、硬件单卡 24GB（如 4090/L20）：**数字为示例起点，以你的实测为准。**

**场景 A：短文本高并发（chatbot，输入≈256、输出≈256）**

```bash
vllm serve Qwen/Qwen3-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096 \          # 短文本，别开大
  --max-num-seqs 512 \            # 高并发，提上来
  --max-num-batched-tokens 4096 \ # 短序列偏小的预算更稳
  --enable-prefix-caching         # 多轮对话开
```

**场景 B：长文本低并发（长文档 / RAG，输入≈8K、输出≈2K）**

```bash
vllm serve Qwen/Qwen3-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \  # 长上下文谨慎抬高
  --max-model-len 16384 \         # 按真实上限，勿拍满
  --max-num-seqs 128 \            # 低并发，避免长序列抢占
  --max-num-batched-tokens 16384 \# 长 prefill 需要大预算
  --kv-cache-dtype fp8 \          # 长上下文 KV 量化收益大
  --enable-prefix-caching         # RAG 共享文档必开
```

**场景 C：混合负载（长短不一、到达率波动）**

```bash
vllm serve Qwen/Qwen3-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \          # 折中上限
  --max-num-seqs 256 \            # 折中并发
  --max-num-batched-tokens 8192 \ # 默认折中，先用基线再扫
  --enable-prefix-caching
# 注意：混合负载最该做的是 8.3 的参数扫描，找到你流量分布下的前沿点
```

**场景 D：Agentic（超长上下文、多轮工具调用、大量重复前缀）**

```bash
vllm serve Qwen/Qwen3-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.93 \  # 前缀复用高，可略抬
  --max-model-len 32768 \         # agentic 上下文长
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \# 长 prefill 预算
  --kv-cache-dtype fp8 \          # 长上下文 KV 量化
  --enable-prefix-caching         # 重复前缀收益最大，必开
  --speculative-config '{"method":"ngram","num_speculative_tokens":5}' # 先 N-gram 探路
# 若有对应 EAGLE draft 权重，可换 EAGLE 并实测接受率
```

**共同逻辑**：四套都从表一推出来，没写死「最优值」——**只复制结构，不复制数字**。

### 表三：症状 → 可能原因 → 调哪个参数（贴墙版）

| 症状 | 可能原因 | 先查 / 调哪个参数 | 佐证指标 |
|---|---|---|---|
| **吞吐上不去** | 并发天花板低；KV 容量不足导致排队；`max-num-batched-tokens` 太小 | 先看 `max-num-seqs`、再扫 `max-num-batched-tokens`、查 `gpu_cache_usage` | `gpu_cache_usage`≈1.0、请求排队长度高 |
| **TTFT 突然变高** | `max-num-batched-tokens` 过大，新请求排队；长 prefill 未分块；前缀缓存失效 | 调小 `max-num-batched-tokens`；查 `--enable-prefix-caching` 是否命中 | TTFT P99 飙升、prefill 队列堆积 |
| **TPOT 抖动** | 抢占风暴（并发逼近 KV 容量）；chunked prefill 与 decode 争预算 | 降 `max-num-seqs` 到 KV 公式的 70%–90%；调 `max-num-batched-tokens` | TPOT P99 远大于 P50、抢占次数指标升高 |
| **频繁 OOM** | `gpu-memory-utilization` 过高；`max-model-len` 开太大；激活值尖峰 | 降 `gpu-memory-utilization` 到 0.90；检查 `max-model-len` | 启动或峰值期 OOM 日志 |
| **并发上不去** | `max-num-seqs` 设太小；或在外层（网关/客户端）被限流 | 提 `max-num-seqs`；排查网关连接数 | 请求持续排队、在飞序列数触顶 |
| **GPU 利用率忽高忽低** | 负载到达率波动；prefill/decode 阶段交替空转（未分离）；batch 形状剧烈变化 | 稳定到达率压测；考虑 PD 分离（第 13 章）；检查输入长度分布 | `gpu_util` 时间序列锯齿 |
| **长请求饿死短请求** | 无优先级调度，长请求占满 KV 与步预算；SLO 未区分 | 降 `max-num-seqs` 缓冲；启用 SLO-aware 调度思路；拆长短队列 | 短请求 TPOT/TTFT 被长请求带高 |

> 用法：线上出问题，先对症状表定位「可能原因」，**回台账做单点变量验证**，不要一次性改三处。

---

## 本章小结

1. **调优的本质是方法，不是参数**。定 SLO → 建基线 → 单点变量 → 回归验证，四步不可交换；跳步得到的是噪声。

2. **`gpu-memory-utilization` 不是越大越好**。0.90 起步，留出激活值尖峰、碎片、CUDA Graph、通信缓冲的 headroom，拉到 0.95+ 往往换来 OOM。

3. **`max-num-seqs` 不背 OOM 的锅**——决定启动期 OOM 的是 `gpu-memory-utilization` 与 `max-model-len`；它调太大换来的是抢占抖动，应设在 KV 公式理论并发的 70%–90%。

4. **`max-num-batched-tokens` 是吞吐-延迟三角的核心旋钮**：调大升吞吐、升 TTFT/TPOT，必须靠扫描画帕累托前沿、在拐点前停下。投机解码同理，是 decode 吞吐杠杆最大也最容易用错的优化（第 7 章），要当成单独实验：先 N-gram 探路、看接受率再常开。

5. **10.6 三张表是可打印的作战地图**：敏感度排序决定先调哪个，场景化配置给四种起点，症状表是排障第一反应；「精确数字」都要以你的实测为准。
