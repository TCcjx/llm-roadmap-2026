# 03 应用层：Prompt / RAG / Agent

[← 返回路线总览](README.md) ｜ 🟢 稳定层（方法）+ 🟡 滚动层（框架）

> **前置**：会 Python + 能调用模型 API。**不需要显卡**。｜ **预计**：4–8 周 ｜ 本页是"路径 A"的主战场

> **目标**：能设计**不靠运气**的检索与智能体流程 —— 即：每个环节都有评估、有兜底、有可解释的失败原因。

原仓库的 Prompt、RAG、Multi-Agent、协议、LangGraph 内容全部并入本页，并补齐了工程化部分。

---

## 1. Prompt 与推理诱导

不是"抄模板"，而是理解每种技术**在什么情况下有效、为什么**。

### 基础手法

| 技术 | 什么时候用 | 失效条件 |
|---|---|---|
| Zero-shot / Few-shot | 格式示范比讲规则更有效 | 示例有偏 → 输出有偏 |
| **CoT**（思维链） | 多步推理、数学、逻辑 | 简单任务反而降智；推理模型不需要 |
| **Self-Consistency**（多采样投票） | 答案可验证、成本可接受 | 开放式生成无意义 |
| **ToT / 搜索式推理** | 需要回溯的规划问题 | token 成本爆炸 |
| 角色/约束/输出格式 | 一切生产场景 | 约束写得含糊 |
| 让模型说"不知道" | 降低幻觉的关键 | 需要评测"拒答率" |

### 现代实践的三个变化

1. **推理模型改变了 CoT 的用法**：对推理模型，"让我们一步步思考"是多余的，甚至有害；需要的是**清晰的目标与约束**
2. **Prompt 要进版本管理**：与模型版本绑定，每次改动跑回归测试
3. **结构化输出优先于解析**：能用 JSON Schema / 约束解码就不要写正则去抠

**资料**

- [Prompt Engineering Guide（系统全面）](https://www.promptingguide.ai/) 🟢
- [Anthropic Prompt Engineering 文档](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) 🟢
- [Lovable Prompting 指南](https://docs.lovable.dev/prompting/prompting-one) 🟢
- [Chain-of-Thought (2022)](https://arxiv.org/abs/2201.11903) ｜ [Self-Consistency (2022)](https://arxiv.org/abs/2203.11171) ｜ [Tree of Thoughts (2023)](https://arxiv.org/abs/2305.10601)
- [OpenAI Cookbook](https://github.com/openai/openai-cookbook) 🟢 ｜ [Anthropic Cookbook](https://github.com/anthropics/anthropic-cookbook) 🟢 — 大量可直接抄的范式

### 结构化输出

- [Instructor](https://github.com/567-labs/instructor) 🟢 — 用 Pydantic 约束输出 + 自动重试
- [Outlines](https://github.com/dottxt-ai/outlines) 🟢 — 约束解码，从生成层面保证合法
- [Pydantic AI](https://ai.pydantic.dev/) 🟢 — 类型安全的 Agent 框架

---

## 2. RAG：从"检索 + 拼接"到可评估系统

**先记住一条主线**：RAG 的效果排序通常是 **切分与数据质量 > 检索召回 > rerank > prompt 编排**。大部分人把时间花在最后一层，收益却最小。

### 2.1 基线与必备组件

```
文档 → 解析(表格/图片/公式) → 切分 → 向量化 → 索引
                                          ↓
Query → 改写/扩展 → 检索(向量+BM25 混合) → Rerank → 压缩 → 拼装上下文 → 生成 → 引用
```

每个环节的常见坑：

| 环节 | 坑 | 对策 |
|---|---|---|
| 解析 | 表格、公式、PDF 双栏错乱 | 用专业解析（版面分析），别用纯文本抽取 |
| 切分 | 一刀切固定长度，切断语义 | 按结构切（标题/段落），带 overlap，保留元数据 |
| 向量化 | 文档与 query 分布不匹配 | 用非对称检索模型或加 query 指令前缀 |
| 检索 | 纯向量丢关键词 | **混合检索**（向量 + BM25），设权重 |
| Rerank | 直接用 top-k 满分结果 | 召回 50~100 再 rerank 到 5~10 |
| 生成 | 不引用、编造来源 | 强制引用 + 无法回答时拒答 |

**资料**

- [All-in-RAG（Datawhale，中文最系统的 RAG 教程）](https://datawhalechina.github.io/all-in-rag/#/chapter1/01_RAG_intro) 🟢 ｜ [源码](https://github.com/datawhalechina/all-in-rag) 🟢
- [RAG 原论文 (2020)](https://arxiv.org/abs/2005.11401) 🟢
- [RAG 的演进（2025）](https://dev523.medium.com/the-evolution-of-rag-how-retrieval-augmented-generation-is-transforming-enterprise-ai-in-2025-a0265bc1c297) 🟡
- [FlashRAG（高效 RAG 研究工具包）](https://github.com/RUC-NLPIR/FlashRAG) 🟢 — 想跑消融实验用它
- [RAGFlow（可落地的开源 RAG 引擎）](https://github.com/infiniflow/ragflow) 🟢 — 看工程实现怎么处理解析与切分
- [LlamaIndex 文档](https://docs.llamaindex.ai/) 🟢 ｜ [Haystack](https://github.com/deepset-ai/haystack) 🟢

### 2.2 查询侧优化（Query Rewriting / Expansion）

原仓库已有内容，此处整理并补全。

#### 假设性文档嵌入（HyDE）

**思路**：先让 LLM 生成一个"假答案"文档，再拿它去检索。

1. 生成假设性文档：LLM 根据 query 生成一个理想化回答，可能含事实错误
2. 嵌入假设性文档：把它向量化
3. 检索：用这个向量做相似度搜索

**为什么有效**：把"短 query 检索长文档"的不对称问题，变成"长文档检索长文档"。**代价**：多一次 LLM 调用，且假设文档跑偏时会误导检索。

- [HyDE 论文 (2022)](https://arxiv.org/abs/2212.10496) 🟢
- [Haystack HyDE 实现](https://docs.haystack.deepset.ai/docs/hypothetical-document-embeddings-hyde) 🟢
- [Medium: HyDE 解读](https://cgorale111.medium.com/hyde-hypothetical-document-embeddings-3071840e364c) 🟡
- [FalkorDB: Advanced RAG](https://www.falkordb.com/blog/advanced-rag/) 🟡

#### 基于 LLM 的查询扩展（LLM-QE）

**思路**：用**检索器反馈**来对齐"扩展模型"，让它生成能提升排名的内容。

- 采用 DPO 训练扩展模型，用两种奖励：
  - **排序奖励**：扩展文档与标准答案文档在检索器眼中的相似度 → 直接对齐检索器偏好
  - **答案奖励**：先让 LLM 依据 query + 标准文档生成"理想答案"，再评估扩展文档与该答案的语义相关性 → 抑制为刷排名而生成冗长无关内容
- [LLM-QE 论文](https://arxiv.org/html/2502.17057v1) 🟡

#### 退步提示（Step-Back Prompting）

先让模型把具体问题抽象成更上位的原理性问题，再同时检索"原理"和"具体"，用于需要背景知识的问题。

- [Step-Back 解读](https://blog.devops.dev/step-back-prompting-smarter-query-rewriting-for-higher-accuracy-rag-0eb95a9cc032) 🟡
- [Advanced RAG 综述](https://www.designveloper.com/blog/advanced-rag/) 🟡

#### 其他查询侧手法

- **查询分解**：复杂问题拆成子问题分别检索 —— [2506.21384](https://arxiv.org/html/2506.21384v1)、[2507.18910](https://arxiv.org/html/2507.18910v1)
- **查询扩展 / 多查询**：生成多个表述并行检索再融合
- **HyDE 的替代**：直接用好的 embedding 模型 + 指令前缀，很多时候更划算

### 2.3 检索与压缩侧

- **混合架构**（向量 + 稀疏）— [2509.12765](https://arxiv.org/html/2509.12765v1)
- **交叉编码器 rerank** — [2508.08742](https://arxiv.org/html/2508.08742v1)
- **动态段落选择器** — [2508.09497](https://arxiv.org/html/2508.09497v1)
- **上下文感知结构化与压缩** — [2508.19357](https://arxiv.org/html/2508.19357v1)
- **SimGRAG（图结构语义对齐）** — [ACL 2025 Findings](https://aclanthology.org/2025.findings-acl.163/)

### 2.4 自适应与自我纠错 RAG

| 方法 | 核心机制 | 资料 |
|---|---|---|
| **Self-RAG** | 模型自己决定"要不要检索""检索结果是否相关""答案是否有依据"，用反思 token 控制 | [官网](https://selfrag.github.io/) ｜ [论文](https://arxiv.org/abs/2310.11511) ｜ [教程](https://www.datacamp.com/tutorial/self-rag) |
| **CRAG** | 加一个**轻量检索评估器**作为决策节点：检索差就走 web 搜索或改写 query | [论文](https://arxiv.org/abs/2401.15884) ｜ [解读](https://medium.com/@sulbha.jindal/corrective-retrieval-augmented-generation-crag-paper-review-2bf9fe0f3b31) |
| **Adaptive-RAG** | 按问题复杂度在"不检索/单步检索/多步检索"间路由 | [论文](https://arxiv.org/abs/2403.14403) |
| **自适应检索** | 让模型学"何时检索" | [2504.05312](https://arxiv.org/abs/2504.05312) |
| **Graph RAG** | 实体关系图 + 社区摘要，解决全局性问题 | [中文文档](https://msdocs.cn/graphrag/) ｜ 详见 [02 §7](02-工程与推理部署.md#7-知识图谱原仓库此处为空已补全) |
| **Reasoning RAG** | 用推理过程驱动多轮检索 | [2506.10408](https://arxiv.org/html/2506.10408v1) |

**工程结论**：先做扎实的混合检索 + rerank，再考虑这些"花活"。CRAG 的检索评估器是**性价比最高**的一个（一个小的分类调用就能挡掉大量坏检索）。

### 2.5 RAG 的评估

见 [04 评测](04-评测-观测-安全.md)。一句话：**没有 retrieval 指标（recall@k、MRR）和 faithfulness 指标，就没有 RAG 工程**。

---

## 3. Agent 工程

### 3.1 先判断：你到底需不需要 Agent

**不需要**（用固定流程更好）：
- 步骤固定、可枚举
- 对确定性有强要求
- 延迟敏感

**需要**：
- 路径未知，需要根据中间结果决定下一步
- 工具集合会变化
- 任务可分解且有明确成功判据

> Anthropic 的观点："先用最简单可行的方案，只在必要时增加复杂度。" 大部分"Agent"其实是一个带 2~3 个分支的工作流。

**资料**

- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) 🟢 — **必读**，把 workflow 与 agent 讲得很清楚
- [Anthropic: Multi-Agent Research System](https://www.anthropic.com/engineering/multi-agent-research-system) 🟢 — 生产级多智能体的真实取舍（token 消耗、并行、评估）
- [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) 🟢 — 把 Agent 当软件工程问题，而不是 prompt 玄学
- [Eugene Yan: LLM Patterns](https://eugeneyan.com/writing/llm-patterns/) 🟢 — Evals / RAG / Guardrails 模式集合

### 3.2 核心组件

| 组件 | 关键问题 | 常用做法 |
|---|---|---|
| **工具（Tools）** | 参数 schema 设计、错误信息要给模型能看懂的话 | 窄接口、幂等、返回结构化错误；工具宁少勿多 |
| **规划** | 何时分解、何时重规划 | 显式 TODO 列表 + 定期复盘；避免无界递归 |
| **记忆** | 短期（上下文）/ 长期（外部存储） | 会话摘要 + 向量记忆 + 结构化用户画像 |
| **执行** | 工具调用可能失败、可能危险 | 沙箱（容器）、超时、最大步数、危险操作需确认 |
| **观测** | 排障靠 trace，不靠猜 | 每步入参/出参/耗时/成本全量记录 |

**记忆与规划的经典论文**

- [ReAct (2022)](https://arxiv.org/abs/2210.03629) 🟢 — 推理与行动交替，一切 Agent 的起点
- [Toolformer (2023)](https://arxiv.org/abs/2302.04761) 🟢 — 模型自学用工具
- [Reflexion (2023)](https://arxiv.org/abs/2303.11366) 🟢 — 用语言化反馈做自我改进
- [MemGPT (2023)](https://arxiv.org/abs/2310.08560) 🟢 — 把上下文当操作系统的内存层级
- [Generative Agents (2023)](https://arxiv.org/abs/2304.03442) 🟢 — 记忆流 + 反思 + 规划

**工程实现**

- [OpenHands](https://github.com/All-Hands-AI/OpenHands) 🟢 — 开源软件工程 Agent，工程细节值得读
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) 🟢 — Agent-计算机接口（ACI）设计的经典案例
- [smolagents](https://github.com/huggingface/smolagents) 🟢 — 代码即动作（Code-as-Action），几百行看懂 Agent
- [smol-course](https://github.com/huggingface/smol-course) 🟢 — 配套课程
- [Letta（原 MemGPT）](https://docs.letta.com/) 🟢 ｜ [Mem0](https://github.com/mem0ai/mem0) 🟢 — 记忆层现成方案
- [AutoGen](https://github.com/microsoft/autogen) 🟢 — 多智能体对话编排
- [CrewAI](https://docs.crewai.com/) 🟢 — 角色化多智能体，上手快
- [ContextAgent](https://github.com/context-machine-lab/ContextAgent) 🟢 — 上下文感知的 Agent 研究实现
- [hello-agents（中文入门教程）](https://github.com/datawhalechina/hello-agents) 🟢
- [多智能体系统全景指南（2025）](https://collabnix.com/multi-agent-multi-llm-systems-the-future-of-ai-architecture-complete-guide-2025/) 🟡
- [Awesome AI Agents](https://github.com/e2b-dev/awesome-ai-agents) 🟢

### 3.3 多智能体：什么时候真的有用

**有用的场景**：任务高度并行（多个独立检索）、需要不同工具集/不同权限、需要"评审者"角色分离。

**常见失败模式**：

- 智能体之间**信息传递损耗**（A 的结论 B 看不懂）
- **成本爆炸**：N 个智能体 × 多轮 = token 数量级增长
- **责任不清**：出错后无法定位是哪个环节
- **收敛困难**：互相"客气"导致达不成结论

**建议**：先用单 Agent + 好的工具 + 明确的 TODO 列表，只有在并行度高、且能独立验证子结果时才上多智能体。

---

### 3.4 2026：harness 成为核心概念（必学）

2026 年，“Agent 框架”这个词基本被 **harness** 取代。原因很现实：**换了 harness 的收益，可以盖过换一代模型的收益。**

一个 2026 年的 harness 至少包含七件事：

```
① 工具接口   窄、幂等、错误信息模型能看懂
② 上下文管理 compaction 压缩 / 外部笔记 / token 预算
③ 执行环境   沙箱、超时、最大步数、成本封顶
④ 验证循环   跑测试、编译、自检 —— 不能相信模型说“完成了”
⑤ 记忆       跨 session 持久化：什么该记、何时忘
⑥ 可观测     每步入参出参 / 耗时 / 成本
⑦ 降级路径   换模型、换 effort 档位、转人工
```

**2026 关键论文与实践**

- [What makes a harness a harness (2606.10106)](https://arxiv.org/abs/2606.10106) — 给出了包含/排除条件，写方案时可直接引用
- [Agent System and Harness Design 综述 (2606.20683)](https://arxiv.org/abs/2606.20683)
- [Agent Harness Distillation (2607.28147)](https://arxiv.org/abs/2607.28147) — harness 本身也能蒸馏/复现
- [Harness-Zero (2609.24974)](https://arxiv.org/abs/2609.24974) ｜ [JIT-Agent (2608.25593)](https://arxiv.org/abs/2608.25593) — harness 自演化
- Anthropic：[Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)
- 值得一看的实战：[Building a C compiler with a team of parallel Claudes](https://www.anthropic.com/engineering)（2026-02-05）

### 3.5 2026 的两个新机制：compaction 与 subagent forking

- **Context compaction**：长任务不再“堆上下文”，而是周期性把历史压成摘要 + 把细节写到外部文件
- **Subagent forking**：派生**独立上下文**的子 agent，子任务结束后只回传结论
- **Agent Swarm**：自主并行分解任务并调度多个子 agent（[Kimi K2.5](https://arxiv.org/abs/2602.02276)）

**代价与坑**（2026 年的新知识）：

1. 轨迹会跨多个 session，**打破单 session 的 RL 训练假设** → 见 [Mosaic: Multi-Session Agentic RL (2610.22174)](https://arxiv.org/abs/2610.22174)
2. 压缩是有损的，**“压掉的那句约束”往往就是失败原因** → 见 [AI Agents Do Not Fail Alone: The Context Fails First (2607.14275)](https://arxiv.org/abs/2607.14275)
3. 子 agent 之间信息传递损耗、成本叠加，需要预算与并发上限

### 3.6 Agent Memory：2026 的独立课题

2026 年 memory 已经从“向量库 + 历史拼接”变成独立研究方向：

- [A Survey of Agent Memory in the Second Half (2602.06052)](https://arxiv.org/abs/2602.06052)
- [From Storage to Experience：记忆机制演进综述 (2605.06716)](https://arxiv.org/abs/2605.06716)
- [Long-Term Memory Security 综述 (2604.16548)](https://arxiv.org/abs/2604.16548) — **记忆是新的攻击面**
- Anthropic：[The components of agent memory](https://www.anthropic.com/engineering)（2026-11-06，含“dreaming”式整理与纠错）

**要能回答的问题**：什么进长期记忆？什么该遗忘？记忆如何隔离不同用户？被污染的记忆如何回溯修正？

---

## 4. 标准化协议：MCP 与 A2A

### MCP（Model Context Protocol）

把"工具/数据接入"标准化，避免每个模型 × 每个工具都要写一遍适配器。

- **核心概念**：Server（提供 tools / resources / prompts）、Client（宿主应用）、Transport（stdio / HTTP+SSE）
- **工程注意**：
  - 工具描述就是 prompt，描述质量直接决定调用准确率
  - 权限最小化：文件系统/数据库 MCP server 必须限制路径与只读
  - **MCP 是提示注入的主要入口之一**，见 [04 §4](04-评测-观测-安全.md)

**资料**

- [MCP 官方文档](https://modelcontextprotocol.io/docs/getting-started/intro) 🟢
- [MCP 官方博客（协议演进看这里）](https://blog.modelcontextprotocol.io/) 🟢
- [官方 Servers 集合](https://github.com/modelcontextprotocol/servers) 🟢
- [规范源码仓库](https://github.com/modelcontextprotocol/modelcontextprotocol) 🟢
- [MCP 中文文档](https://docs.mcpcn.org/introduction) 🟢
- [MCP 生态导航](https://mcpmarket.cn/) 🟡
- [MCP 概览（Descope）](https://www.descope.com/learn/post/mcp) 🟢

**2026 协议层必知（2026-07-28 规范，核对 2026-11-13）**

- **无状态协议核心**：服务端不再依赖会话粘滞，水平扩展变简单
- **Multi Round-Trip Requests**：一次调用内可多次往返（多轮确认/交互）
- **Tasks / MCP Apps / Extensions 框架**：长任务、UI 型能力、可插拔扩展成为一等公民
- **可缓存的 list 结果 + header 路由**：降低列表类调用开销
- **授权加固**：Enterprise-Managed Authorization（零接触 OAuth，组织级集中开通，2026-06 稳定）
- **发布节奏固定化**（2026-11-11）、**弃用策略正式化**

**安全**：MCP 已成为主要攻击面之一，必读 [Auditing MCP Servers for Over-Privileged Tool Capabilities (2603.21641)](https://arxiv.org/abs/2603.21641)、[SMCP: Secure MCP (2602.01129)](https://arxiv.org/abs/2602.01129)

### A2A（Agent2Agent）

解决"智能体之间如何互相发现与委派任务"，与 MCP 互补：**MCP 是 Agent↔工具，A2A 是 Agent↔Agent**。

- [A2A 规范与 MCP 的关系](https://a2a-protocol.org/latest/#how-does-a2a-work-with-mcp) 🟢
- [Google A2A 实战 Codelab](https://codelabs.developers.google.com/intro-a2a-purchasing-concierge?hl=zh-cn#0) 🟢

---

## 5. 框架选型

| 框架 | 定位 | 什么时候选 | 状态（2026-11） |
|---|---|---|---|
| **LangGraph** | 有状态的图式编排，循环/分支/人工介入 | 需要复杂控制流、可恢复执行 | 活跃（v1.2.x） |
| **LlamaIndex** | 数据与检索为中心 | RAG 为主 | 活跃 |
| **Haystack** | 生产级 pipeline | 企业搜索、可维护性优先 | 活跃 |
| **DSPy** | **声明式编程**：你写“输入输出契约”，让框架优化 prompt/few-shot | 想摆脱手调 prompt；有评测集可自动优化 | 活跃（v3.x）—— **2026 年最容易被忽略的真东西** |
| **Pydantic AI** | 类型安全 | 强 schema 约束 | 活跃（v2.x） |
| **Agno** | 轻量 agent + 内置记忆/知识 | 想少写样板代码 | 活跃（v3.x） |
| **smolagents** | 极简、代码即动作 | 学习与轻量场景 | 活跃 |
| **CrewAI** | 角色化多智能体 | 概念验证 | 活跃 |
| **Microsoft Agent Framework** | 替代 AutoGen 的新一代 SDK | 需要微软生态 / Semantic Kernel 兼容 | 新项目选它 |
| **AutoGen** | 多智能体老牌项目 | 维护已有代码 | ⚠️ 已放缓（PyPI 最后发布 2025-09），新项目请用 Agent Framework |
| **Dify / n8n** | 低代码 | 原型、内部工具、非工程同学使用 | 活跃 |

**别做的事**：为了"用上框架"而引入框架。一个 300 行的自研状态机往往比一堆抽象更好维护。

- [LangGraph 文档](https://docs.langchain.com/oss/python/langgraph/overview) 🟢 ｜ [源码](https://github.com/langchain-ai/langgraph) 🟢
- [DSPy 文档](https://dspy.ai/) 🟢 ｜ [源码](https://github.com/stanfordnlp/dspy) 🟢
- [Agno 文档](https://docs.agno.com/) 🟢
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) 🟢

> **DSPy 为什么值得单独拎出来**：2026 年“手写 prompt”正在被“**定义签名 + 用评测集自动优化**”取代。它和你手上的评测集是乘数关系：有评测集 → DSPy 才能真正发挥作用。

---

## 6. AI 编程工具链（2026 的日常）

这是 2026 年新增的一节，也是**新手最快建立手感的地方**。2025 年的路线几乎都不提，但它已经是日常工作的主体。

### 6.1 三个核心认知

1. **coding agent 不是“自动补全”**，它是“能读写仓库、跑命令、开 PR 的执行者”。它靠 **diff + 测试**工作，所以 **Git 与测试是前提**，不是可选。
2. **harness（外层壳）决定效果**：同一个模型放在不同 tool 里，效果差很多。所以你选的不是“模型”，是“模型 + 外壳”。
3. **你的角色从“写代码”变成“审查、拆任务、写验证”**。会后端能力/工程审美的人反而更吃香。

### 6.2 主流工具（链接均验活）

| 工具 | 形态 | 说明 |
|---|---|---|
| [Claude Code](https://code.claude.com/docs/en/overview) | CLI + IDE | 2026 年事实上的标杆，也是“harness”概念的源头之一 |
| [Codex](https://github.com/openai/codex) | CLI / 云端 | OpenAI 系；官方文档见 developers.openai.com |
| [OpenCode](https://opencode.ai/) | 开源 CLI | 模型无关，**可用 DeepSeek/Qwen 等国产后端** |
| [OpenClaw](https://github.com/openclaw/openclaw) | 开源 agent | 2026 年新兴的自托管选择 |
| [gemini-cli](https://github.com/google-gemini/gemini-cli) | 开源 CLI | Google 系 |
| [Qwen Code](https://github.com/QwenLM/qwen-code) | 开源 CLI | 阿里系，国内可直接用 |
| [Cursor](https://cursor.com/) | IDE | 编辑器形态的 agent |
| [Antigravity](https://antigravity.google/) | 云端 IDE | Google 系，带 `/swarm` 并行子 agent |

**用国产模型跑这些工具**：DeepSeek 官方提供了按工具划分的接入说明（Claude Code / Codex / OpenCode / OpenClaw / Qoder / CodeBuddy 等），大多数只需改 `base_url` 与 key —— 见 [DeepSeek Agent Integrations](https://api-docs.deepseek.com/guides/agent_integrations)。

### 6.3 新手用 coding agent 的正确姿势

```
✅ 先写测试/验收标准，再让 agent 改
✅ 一次只让它做一件事（一个 PR = 一个意图）
✅ 看不懂的 diff 一律不合并
✅ 让它自己跑测试，失败就自己修（这就是 harness 的验证循环）
❌ 不要说“帮我优化一下这个项目”
❌ 不要盲目接受它“声称已完成”
```

**必读的一手材料**：

- Anthropic《Harness design for long-running application development》（2026-03-24，见 [Engineering 索引](https://www.anthropic.com/engineering)）
- Anthropic《Building a C compiler with a team of parallel Claudes》（2026-02-05）—— 多 agent 并行的真实案例
- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) 🟢

---

## 7. Agent Skills 与 code-mode（2026 两个新增实践）

### 7.1 Agent Skills：把“能力包”变成可分发资产

2025 年底提出、2026 年成为生态事实标准：用一组文件（指令 + 脚本 + 资源）封装一个能力，agent 按需加载。

- 规范与生态：[agentskills.io](https://agentskills.io/) 🟢
- 官方说明：[Agent Skills 文档](https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview) 🟢
- 治理：[Skills 已捐赠给 Agentic AI Foundation](https://www.anthropic.com/news/donating-agent-skills-to-the-agentic-ai-foundation) 🟢（2026）

**为什么重要**：它回答了“**怎么把一个人的经验变成团队可复用的东西**”——不是写更长的 system prompt，而是打包成 skill。相关综述：[Agent Skills Survey (2605.07358)](https://arxiv.org/abs/2605.07358)、[Agentic Skills (2608.29596)](https://arxiv.org/abs/2608.29596)。

### 7.2 code-mode：让模型“写代码调工具”而不是“逐个调工具”

当工具多到几十个时，逐次调用会爆炸 token。替代方案：**把工具当 API，让模型写一段代码去调用**。

- Anthropic《Code execution with MCP: Building more efficient agents》—— 见 [Engineering 索引](https://www.anthropic.com/engineering)
- 与本页 §3.4 的“工具接口设计”配合看；实现上注意**沙箱**（这是安全边界）

---

## ✅ 动手清单

- [ ] 搭一个最小 RAG：解析 PDF → 切分 → pgvector → 混合检索 → rerank → 生成 + 引用
- [ ] 在 30 条自建问题上，对比"纯向量" vs "混合检索" vs "混合+rerank" 的 recall@5
- [ ] 实现 CRAG 的检索评估器（一个分类调用），测量它挡掉了多少坏检索
- [ ] 实现 HyDE，测量它在"短 query"场景的收益与额外成本
- [ ] 用 LangGraph 写一个带循环和人工确认的 Agent（含最大步数与超时）
- [ ] 写一个 MCP server 暴露你自己的数据，并只读/限制路径
- [ ] 给 Agent 加 trace，能回放一次失败任务的完整过程
- [ ] 用 coding agent（Claude Code / OpenCode / Qwen Code）完成一个小需求：**先写测试，再让它过测试**
- [ ] 把你自己的一段经验（如“如何做代码审阅”）封成一个 **Agent Skill**，在真实任务里试用
- [ ] 用一个 Decision model 或小模型替代你的 LLM-judge，对比分数一致率与成本

## 🔍 自检问题

1. 什么情况下你会**建议客户不要用 RAG**？（提示：结构化查询、权限差异、实时性）
2. 切分长度怎么定？为什么固定 512 token 通常是错的？
3. 混合检索里 BM25 权重给多少？怎么调？（提示：用标注集扫参数）
4. 为什么 rerank 要在召回 50~100 条之后做，而不是直接召回 5 条？
5. Self-RAG 与 CRAG 的核心区别是什么？各自多花了多少 token？
6. Agent 陷入循环，你会从哪几个地方下手修？
7. MCP 带来的最大安全风险是什么？如何在架构上缓解？