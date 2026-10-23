# 第 12 章　SGLang 核心能力

## 12.1 RadixAttention 前缀缓存实战

![RadixAttention：用一棵基数树管理所有请求的前缀，公共前缀直接复用](./assets/ch12-radix-tree.png)

第 0 章 0.5.2 节给了概念：基数树存所有 KV 块、命中前缀直接复用；第 4 章讲了它与 PagedAttention 的关系和四个实战陷阱。本节讲观测、诊断与收益。

### 12.1.1 命中率怎么观测

命中率可以这样观测：

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen3.5-0.8B --tp 1 \
  --enable-prefix-caching \
  --enable-cache-report        # 在日志/指标中输出缓存命中统计
```

开启后引擎会周期性打印（或暴露到 Prometheus），压测时直接看收益；结果里也能拿到 `cached_tokens`。

> 诚实提示：`cached_tokens` 的具体字段名和取值口径随版本变动。请以你所用版本的官方文档与引擎日志为准，上面的开关是"观测思路"而非逐字段承诺。

### 12.1.2 命中率健康区间与诊断方法

把命中率当仪表盘看，分三档：

| 区间 | 含义 | 动作 |
|---|---|---|
| **50%–75%+** | 健康（第 0 章 0.5.2 节给的生产经验区间） | 保持现状，这是 RadixAttention 的主场 |
| **20%–50%** | 偏低，有优化空间 | 查 prompt 是否注入了随机前缀（时间戳/ID）、多副本是否轮询稀释 |
| **≈0%** | 缓存几乎失效 | 要么 prompt 全唯一（RAG 文档各异、无共享 system），要么踩了 4.7 节的陷阱 |

常见诊断（第 4 章 4.7 节）：**随机前缀注入**——system prompt 带 `{datetime.now()}` 或 `request_id`，第一个 token 就分叉、命中率归零，把易变内容挪到前缀后；**多副本轮询稀释**——`--dp N` 或轮询 LB 把相同前缀打散到 N 副本，命中率≈1/N（引出 10.4）；**缓存淘汰抖动**——长尾 KV 被 LRU 赶出又重算，P99 会随命中率周期波动。

### 12.1.3 动手实验：三种负载对比缓存收益

```text
负载 A 多轮对话：  同一 session 连续追问，前缀逐轮累积 → 命中率应很高
负载 B RAG：       共享同一段检索文档块 + 不同问题 → 命中文档前缀
负载 C 全随机：    每个请求 prompt 完全不同、无共享 → 命中率≈0
```

实验脚本 `src/ch12-sglang-features/01_prefix_cache_bench.py` 跑这三类负载、打印 `cached_tokens` 与 TTFT。预期（定性，数字以实测为准）：A、B 相对 C 的 TTFT 显著下降、吞吐上升；C 与"关掉前缀缓存"差异极小——**前缀缓存不是万能加速，收益完全绑定"请求相似度"**。

> **什么场景值得用 / 不值得用**
> - 值得：多轮对话、共享 system prompt 的 Agent、RAG（共享文档块）、同模板批量抽取。
> - 不值得：每个 prompt 都独一无二的生成任务（创意写作、随机种子驱动）——它帮不上忙，别指望它"自动变快"，优化预算应投到量化、投机解码、并行。

---

## 12.2 结构化输出与压缩 FSM

![结构化输出：用有限状态机把非法 token 直接屏蔽，输出天然合法](./assets/ch12-structured-fsm.png)

Agent / 工具调用时代，"模型必须吐出可被程序解析的 JSON"是硬需求；早期"生成完再校验、错了重试"代价极高。现代做法是把约束编译成**有限状态机（FSM）**，解码每步屏蔽非法 token 的 logits——非法输出物理上不可能产生。

### 12.2.1 压缩有限状态机与 jump-forward

未压缩的 FSM 每个 token 一个状态，而 JSON 里结构字符大多是确定的（如 `{`、`"`、`field_name`：），逐 token 走会造出大量"只有一条合法出边"的状态——每步一次 GPU 前向，纯浪费。**压缩 FSM（xgrammar）**把"只有唯一确定路径"的那串 token 折叠成一个跳跃：生成 `"name": "` 不必分 7 步，而是一段压缩边。

由此得到的 **jump-forward（前向跳跃）**：FSM 处于"后续 token 强制唯一确定"的状态时，**一次前向吐出整串强制 token**，把 N 次 decode 步压成 1 次。

> 要点：jump-forward 省的是**decode 步数**（每步都要把权重从 HBM 读一遍的内存墙，见第 1.3 节），不是省算力——它把"必然确定的 token"从逐 step 生成变成批量生成，是 0.5.2 节「结构化生成解码开销下降」的工程落地。

### 12.2.2 JSON / Regex / EBNF 实战

SGLang 原生用 xgrammar 做结构化后端，推荐通过 `structured` 字段声明约束：

```python
out = engine.generate(
    [{"role": "user", "content": "抽取姓名与年龄"}],
    sampling_params={
        "structured": {            # SGLang 的结构化输出入口
            "type": "json",
            "schema": {"name": "string", "age": "integer"},
        }
    },
)
```

> 上面的 `structured` 字段名/取值为思路示意，SGLang 各版本用 `regex` / `json` / `ebnf` 等不同键位，请以官方文档为准。Regex / EBNF 同理，给一条文法规则即可。

### 12.2.3 与 vLLM 结构化输出的实现差异（诚实对比）

SGLang 原生带压缩 FSM + jump-forward，vLLM 也可用 xgrammar / guidance（`guided_json` / `guided_regex` 等字段）。**判断**：两者底层都站在 xgrammar 肩上，单看"结构化输出速度"没有代际差距；**SGLang 的相对优势是把结构化约束和 DSL 控制流（fork 多候选、select 分类）揉在一起写**。已在 vLLM 上、只要 JSON → 不必迁；要"结构化 + 多分支 + 前缀复用"→ 用 SGLang DSL。

> **什么场景值得用 / 不值得用**
> - 值得：工具调用 / 函数签名解析、表单抽取、任何需要"零重试硬约束"的解析。
> - 不值得：输出高度自由、不需程序解析的文本——结构化约束反而限制模型，且编译与屏蔽本身也有成本。

---

## 12.3 大模型与 MoE 部署实战

### 12.3.1 MLA / EP / 量化组合

以 DeepSeek 系列（V2/V3/R1）为例，硬门槛 🖥️🖥️，三个关键工程点：**MLA（Multi-head Latent Attention）**低秩压缩 KV，把 KV Cache 压到极小（第 4 章 4.2 节），同样显存能塞更大并发或更长上下文；**EP（Expert Parallel）**把专家层切到多卡，瓶颈在 All-to-All 通信（第 6 章），配合 NVLink / IB 服务超大 MoE；**量化**常见"权重 FP8 + MLA 低秩 KV"，大幅降显存。

```bash
# DeepSeek-V3 多卡部署示意（数值与卡数以你的硬件为准）
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 \
  --quantization fp8 \
  --enable-prefix-caching
```

> 诚实提示：DeepSeek-V3/R1 是数百 B 参数的 MoE，8 卡不一定放得下（取决于量化与 EP 配置）。具体卡数、EP 度、量化组合请以官方部署示例和你的硬件显存为准，**不要照抄上面的 8 卡**。

### 12.3.2 混合稀疏注意力与 Day-0 支持

混合稀疏注意力下，稀疏层本不缓存全部历史，SGLang 的思路（社区称 **ShadowRadix**）是在稀疏层也维护"影子"前缀树，让可复用前缀仍能被匹配。它也以"新模型 Day-0 能跑"著称：新架构后端跟得快，但新合入 API 可能漂移、生产要锁版本，且能跑 ≠ 跑得最优。

> **什么场景值得用 / 不值得用**
> - 值得：要服务 DeepSeek 级超大 MoE、需要 MLA 低秩 KV 与大规模 EP、需要追新模型 Day-0 验证。
> - 不值得：中小稠密模型（7B–32B）且已在 vLLM 稳定运行——这类 SGLang 没有代际优势，迁移只为"尝鲜"不划算。

---

## 12.4 SGLang Router

第 4、5 章都点过：**多副本轮询会把前缀缓存命中率稀释成 1/N**。SGLang Router 是解决它的缓存感知路由层：维护各副本的前缀树摘要，把"前缀最可能命中"的请求派到对应副本；代价是路由层要感知缓存状态，副本多、相似度高时收益远大于开销。

### 12.4.1 部署示例

```bash
# 先起多个带前缀缓存的 SGLang 实例
for i in 0 1 2; do
  python -m sglang.launch_server --model-path Qwen/Qwen3-32B \
    --tp 4 --port 3000$i --enable-prefix-caching &
done

# 再起 Router，做缓存感知路由（端口/参数以官方文档为准）
python -m sglang_router.launch_router \
  --host 0.0.0.0 --port 40000 \
  --worker http://localhost:30000 http://localhost:30001 http://localhost:30002
```

客户端只连 Router 的 `http://localhost:40000/v1`。另有 `--enable-dp-attention`：在 `--dp N` 下让各 DP rank 协同注意力计算，与 Router 互补——Router 做跨副本路由，它做副本内协同、减少重复 prefill。

> **什么场景值得用 / 不值得用**
> - 值得：多副本（≥3）且请求有明显共享前缀（多轮对话、RAG、批量同模板）——命中率从 1/N 救回。
> - 不值得：单副本、或前缀本就全随机（见 10.1）——路由层白白增加复杂度和延迟，命中率救不回来。

---

## 12.5 作为 RL rollout 引擎 ⏭️ ☁️

标记 ⏭️（进阶）与 ☁️（多机/集群倾向），第一遍可跳。RL 后训练循环里，推理引擎要反复当 rollout 工人采样轨迹，三点不同恰好命中 SGLang 的设计：**权重频繁更新**——**支持权重原地更新（NCCL/权重同步通道）**，跳过重启重读权重，是 verl、OpenRLHF 等框架选它做 backend 的主因；**前缀复用**——轨迹共享 system prompt 时同样生效；**sleep mode / 显存让渡**——训练与推理交替时把推理显存交还。

协同形态是训练进程算出新权重、经通道推给引擎热替换，关键是**数值一致性**：采样路径要与训练侧对齐，否则 rollout 分布偏移会让 RL 收敛出问题。

> SGLang 在 RL rollout 上的优势是"工程协同成熟度"，不是"推理算法更快"。不涉及 RL 后训练的场景，跳过无妨。

---

## 本章小结

1. **优势是场景绑定的，不是全面"更快"**：前缀缓存、结构化+多分支、MoE、缓存感知路由各有最适负载。
2. **RadixAttention 收益 = 请求相似度**：多轮对话/RAG/同模板批量收益明显，全随机 prompt 命中率≈0、帮不上忙。
3. **压缩 FSM + jump-forward 省的是 decode 步数**（内存墙），不是算力；与 vLLM 同站 xgrammar，相对优势在和 DSL 控制流融合。
4. **投机解码（第 7 章）走 Spec V2 统一框架**，覆盖自回归起草（EAGLE/MTP/N-gram）与并行起草（DFlash/DSpark）。
5. **MoE 靠 MLA 低秩 KV + EP + 量化**，Day-0 跟得快但需锁版本；Router 把多副本命中率从 1/N 救回，RL rollout 偏爱它则因权重热更新与显存让渡。
