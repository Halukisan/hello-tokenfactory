<div align='center'>
    <img src="./logo.png" alt="Hello-TokenFactory" width="100%">
    <h1>hello-toeknfactory（🧪Beta公测版本）</h1>
</div>

> [!WARNING]
> 🧪 Beta公测版本提示：教程主体已完成，正在优化细节，欢迎大家提Issue反馈问题或建议。

## 🚀 项目介绍
算力不是稀缺品，能用的 token 才是。数据不能出内网、云 API 账单压不住——成本与隐私这两股力量，正把越来越多的推理负载赶回企业自己的机房。
hello-toeknfactory想解决的，就是"从算力到 token"这条链上的每一环。把 vLLM / SGLang 两大推理引擎当作产线，带你从理解推理原理一路走到 token 工厂。

## 📌项目受众

无论你是刚接触推理部署、第一次给自己的卡搭服务，还是已经跑着 vLLM / SGLang、想把算力变成稳定产能的架构师，都能在本书找到对应的内容。

## 📖目录

| 章节 | 关键内容 | 状态 |
| --- | --- | --- |
| [前言](./docs/前言.md) | 本书讲什么、四幕节奏、四条阅读路径、环境清单、三种拿到算力的方式 | ✅ |
| [第 0 章 为什么需要一座 token 工厂](./docs/ch00-为什么需要token工厂.md) | 推理演化史、三代引擎、瓶颈转移、token 自由的定义 | ✅ |
| [第 1 章 算力 ≠ token：从裸卡到可用服务](./docs/ch01-算力不等于token.md) | TTFT/TPOT/Goodput 指标、Prefill vs Decode 两种瓶颈、显存账本 | ✅ |
| [第 2 章 token 工厂全景：一条从 GPU 到 API 的流水线](./docs/ch02-token工厂全景.md) | 工厂剖面图（五层）、各层职责与代表技术、vLLM/SGLang/GPUStack 生态位、裸装 vs 平台 | ✅ |
| [第 3 章 推理引擎原理与生态](./docs/ch03-推理引擎原理与生态.md) | 引擎四大件、按切入口分类的生态全景、vLLM 与 SGLang 的定位差异、出厂规格怎么看 | 🚧 |
| [第 4 章 KV Cache 与注意力工程化](./docs/ch04-KV缓存与注意力工程化.md) | KV 显存公式、MHA→MLA、PagedAttention vs RadixAttention、前缀缓存陷阱 | 🚧 |
| [第 5 章 调度与批处理](./docs/ch05-调度与批处理.md) | 连续批处理、Chunked Prefill、vLLM V1 架构、缓存感知路由 | 🚧 |
| [第 6 章 并行策略全景](./docs/ch06-并行策略全景.md) | DP/TP/PP/SP、MoE 与 EP、并行决策表 | 🚧 |
| [第 7 章 投机解码](./docs/ch07-投机解码.md) | Draft-Verify、六种方案选型、接受率、双引擎配置 | 🚧 |
| [第 8 章 vLLM 快速上手与单机多卡部署](./docs/ch08-vLLM快速上手与多卡部署.md) | 安装三选一、离线/API、启动参数、单机多卡 TP/PP | 🚧 |
| [第 9 章 vLLM 核心能力](./docs/ch09-vLLM核心能力.md) | 多卡多机、量化、结构化输出、LoRA、多模态、Embedding | 🚧 |
| [第 10 章 vLLM 性能调优](./docs/ch10-vLLM性能调优.md) | 调优基线、显存/并发、`max-num-batched-tokens`、敏感度排序 | 🚧 |
| [第 11 章 SGLang 快速上手与低延迟实战](./docs/ch11-SGLang快速上手.md) | 三种使用方式、前端 DSL、参数对照、命中率实战 | 🚧 |
| [第 12 章 SGLang 核心能力](./docs/ch12-SGLang核心能力.md) | RadixAttention 实战、压缩 FSM、MoE 部署、Router、RL rollout | 🚧 |
| [第 13 章 PD 分离与大规模部署](./docs/ch13-PD分离与大规模部署.md) | PD 分离架构、KV 传输、成本账、不该做的门槛 | 🚧 |
| [第 14 章 双引擎选型与决策](./docs/ch14-双引擎选型与决策.md) | vLLM vs SGLang、横向对比、国产化适配、选型决策树 | 🚧 |
| [第 15 章 GPUStack 是什么](./docs/ch15-GPUStack是什么.md) | 定位（MaaS+GPUaaS）、架构解剖（Server/Worker/Gateway/Ray）、请求数据流、硬件版图 | 🚧 |
| [第 16 章 装起一座 GPUStack](./docs/ch16-安装GPUStack.md) | 前置检查、一键安装、纳管 Worker、单机多卡、Web 控制台、排障 | 🚧 |
| [第 17 章 部署模型：从 GPU 到 OpenAI API](./docs/ch17-部署模型到API.md) | 模型来源、四种部署模式、自动引擎选择、API Key/Endpoint/Playground、接入上层应用、计量限流 | 🚧 |
| [第 18 章 GPUStack 技术栈深潜](./docs/ch18-GPUStack技术栈深潜.md) | Binpack/Spread 调度、Ray 分布式 vLLM、Higress 网关、Prometheus/Grafana、加速集成、规模化复盘 | 🚧 |
| [第 19 章 压测与可观测](./docs/ch19-压测与可观测.md) | 负载建模、压测陷阱、吞吐-延迟拐点、P99 诊断、监控面板 | 🚧 |
| [第 20 章 生产部署架构](./docs/ch20-生产部署架构.md) | 部署形态、K8s、网关五职责、灰度、成本优化、完整架构示例 | 🚧 |
| [第 21 章 完整案例: 3分钟搭建跨州算力工厂](./docs/ch21-完整案例.md) | 3分钟搭建跨州算力工厂 | 🚧 |

## ✨核心贡献者
- [牧小熊-项目负责人](https://github.com/muxiaoxiong)(Datawhale成员)

## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull Request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*
