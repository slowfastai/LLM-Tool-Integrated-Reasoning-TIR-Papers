# 🔧 LLM-Tool-Integrated-Reasoning-TIR-Papers

A curated collection of papers on **Tool-Integrated Reasoning (TIR)** — a rapidly evolving research direction where Large Language Models (LLMs) interact with **external tools** such as calculators, search engines, code interpreters, and web APIs to enhance reasoning, decision-making, and factual accuracy.

> 🧠 "The ability to use tools is what sets humans apart from other animals."  
> 🤖 Likewise, **the ability to use tools is what transforms an LLM into an agent.**

TIR marks a critical milestone in the evolution of LLMs: it extends models beyond static parametric knowledge, enabling them to dynamically interact with the external world via:
- 🖥️ Python interpreters  
- 🔍 Search engines  
- 🧮 Calculators  
- 🌐 Web APIs

Although this list focuses on Tool-Integrated Reasoning, we also include earlier or adjacent works on LLMs + Tools that may not explicitly involve reasoning, in order to provide a more complete historical and technical context.

📌 Note: This list focuses on tool-integrated reasoning with text-only LLMs and does not include multimodal models.



---

## 🔍 Filter by Category

🎯 Tool Type:
[Code](./papers/by-tool/code.md) | 
[Search](./papers/by-tool/search.md) | 
[Calculator](./papers/by-tool/calculator.md) | 
[Multi-tool](./papers/by-tool/multitool.md)

📘 Training Method:
[Prompt-only](./papers/by-training/prompt_only.md) |
[SFT](./papers/by-training/sft.md) |
[RL](./papers/by-training/rl.md)

---

## 📜 Paper List

| Paper | Date | Code | Tags | Summary |
|-------|------|------|------|---------|
| [WebGPT: Browser-assisted question-answering with human feedback](https://arxiv.org/abs/2112.09332) | 2021-12 | Not officially released | `search` `browser` `sft` `rlhf`| WebGPT is an early tool-augmented QA agent that trains GPT-3 to use a simulated browser for information retrieval via SFT and RLHF, enabling it to answer questions with citations in a fixed “browse → answer” pipeline.|
| [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) | 2023-02 | [Unofficial](https://github.com/lucidrains/toolformer-pytorch) | `wiki-search` `alculator` `calendar` `qa-api` `mt-api` `sft` | Toolformer enables LLMs to learn when and how to use external tools by generating self-supervised training data and fine-tuning via SFT, without human annotation. |
| [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face](https://arxiv.org/abs/2303.17580) | 2023-03 | [Official](https://github.com/microsoft/JARVIS) | `multi-tool` `prompt` | HuggingGPT uses prompt-driven planning to let LLMs act as a central controller that delegates tasks to expert models, enabling multi-model collaboration via natural language. |
| [ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings](https://arxiv.org/abs/2305.11554) | 2023-05 | [Official](https://github.com/Ber666/ToolkenGPT) | `multi-tool` `embedding` `tool-tokenization`| ToolkenGPT enables LLMs to call tools like predicting tokens by introducing learnable tool-specific embeddings (“toolkens”) into the vocabulary, without modifying model parameters. |
| [START: Self-taught Reasoner with Tools](https://arxiv.org/abs/2503.04625) | 2025-03 | Not officially released | `python-interpreter` `hint-infer` `sft`| START enables Qwen to self-learn tool use by inserting natural language hints into reasoning paths to generate TIR data, then fine-tuning via SFT to internalize tool-calling abilities.|
| [R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.05592) | 2025-03 | [Official](https://github.com/RUCAIBox/R1-Searcher) | `search` `sft` `rlvr`| R1-Searcher introduces a two-stage RLVR framework that teaches LLMs to use search tools for multi-hop QA, first encouraging tool use, then optimizing for correct answers with increasingly difficult examples. |
| [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516) | 2025-03 | [Official](https://github.com/PeterGriffinJin/Search-R1) | `search` `rlvr`| Search-R1 trains LLMs via RLVR to reason and use a search engine tool, using structured prompts and a reward function that encourages effective <search>–<answer> sequences while masking retrieved content during optimization. |
| [ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning](https://arxiv.org/abs/2503.19470) | 2025-03 | [Official](https://github.com/Agent-RL/ReCall) | `search` `rlvr`| ReSearch applies RLVR to multi-hop QA, training LLMs to reason effectively by learning when and how to use a search engine tool in a reward-driven framework. |
| [ToRL: Scaling Tool-Integrated RL](https://arxiv.org/abs/2503.23383) | 2025-03 | [Official](https://github.com/GAIR-NLP/ToRL) | `python-interpreter` `rlvr`| ToRL trains Qwen2.5-math to generate and execute Python code for math problem solving via RLVR. |
| [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958) | 2025-04 | [Official](https://github.com/qiancheng0/ToolRL) | `prm-style-reward` `rlvr`| ToolRL extends RL-based tool use beyond math by introducing fine-grained reward functions that directly compare LLM-generated responses with ground truth, resembling a PRM-style variant of RLVR for general tasks. |
| [Nemotron-Research-Tool-N1: Exploring Tool-Using Language Models with Reinforced Reasoning](https://arxiv.org/abs/2505.00024) | 2025-05 | [Official](https://github.com/NVlabs/Tool-N1) | `general-domain` `rlvr`| ToolN1 extends LLM+RL+Tool training to non-math domains by combining GRPO with a binary reward function that encourages correct response formatting and tool usage. |
| [ZeroSearch: Incentivize the Search Capability of LLMs without Searching](https://arxiv.org/abs/2505.04588) | 2025-05 | [Official](https://github.com/Alibaba-NLP/ZeroSearch) | `search` `simulated-tool` `sft` `rlvr`| ZeroSearch trains LLMs to use search tools via RLVR by simulating a search engine with an SFT model, avoiding API costs while ensuring controllable document quality during training.|
| [Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving](https://arxiv.org/abs/2505.07773) | 2025-05 | [Official](https://github.com/yyht/openrlhf_async_pipline) | `python-interpreter` `rlvr`| ZTRL trains the Qwen2.5 base model with RLVR to autonomously generate and execute Python code for math problems. |
| [Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning](https://arxiv.org/abs/2505.16410) | 2025-05 | [Official](https://github.com/dongguanting/Tool-Star) | `multi-tool` `sft` `rlvr`| Tool-Star extends Tool-Integrated Reasoning to multi-tool scenarios by constructing a synthetic multi-stage training dataset and optimizing with hierarchical rewards via SFT and self-critic RL. |
| [Learning to Reason without External Rewards](https://arxiv.org/abs/2505.19590) | 2025-05 | [Official](https://github.com/sunblaze-ucb/Intuitor) | `implicit-reward`| Intuitor proposes an implicit reward strategy for RL without human feedback or ground truth, using LLM confidence (KL divergence from uniform) as a self-assessed signal to guide learning. |
| [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/abs/2505.22648) | 2025-05 | [Official](https://github.com/Alibaba-NLP/WebAgent) | `search` `multi-tool` `sft` `rlvr` `deep-research`| WebDancer enhances LLMs' information-seeking ability via ReAct-style multi-tool reasoning, controllable TIR data generation, and a two-stage SFT + DAPO training pipeline for deep research tasks.|
| [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/abs/2505.22648) | 2025-05 | [Official](https://github.com/Alibaba-NLP/WebAgent) | `search` `multi-tool` `sft` `rlvr` `deep-research`| WebDancer enhances LLMs' information-seeking ability via ReAct-style multi-tool reasoning, controllable TIR data generation, and a two-stage SFT + DAPO training pipeline for deep research tasks.|


---

Made with ❤️ by the open-source research community.
