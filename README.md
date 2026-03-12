# Literature Review: RAG  methods to enhance LLM Reasoning

> **Scope**: Retrieval-Augmented Generation (RAG), with emphasis on graph-structured retrieval (GraphRAG) for knowledge-intensive reasoning.

> **Last updated**: 2026-03-12

---

## Table of Contents

1. [Timeline Overview](#timeline-overview)
2. [Foundational RAG (2020–2022)](#1-foundational-rag-20202022)
3. [Advanced RAG Techniques (2023–2024)](#2-advanced-rag-techniques-20232024)
4. [Knowledge Graph QA — Pre-LLM Era (2019–2022)](#3-knowledge-graph-qa--pre-llm-era-20192022)
5. [GraphRAG: KG-Enhanced LLM Reasoning (2023–2026)](#4-graphrag-kg-enhanced-llm-reasoning-20232026)
6. [Community & Global GraphRAG Systems (2024–2025)](#5-community--global-graphrag-systems-20242025)
7. [Survey Papers](#6-survey-papers)
8. [Benchmarks & Datasets](#7-benchmarks--datasets)
9. [State-of-the-Art Leaderboards](#9-state-of-the-art-leaderboards)
10. [Taxonomy & Positioning Map](#10-taxonomy--positioning-map)
11. [Reading Order Recommendation](#reading-order-recommendation)
12. [Open Research Questions](#open-research-questions-as-of-2026)

---

## Timeline Overview

```
2020  ██ RAG (Lewis), REALM (Guu)                          — Retrieve-then-generate paradigm
2021  █  FiD (Izacard)                                      — Multi-doc fusion
2022  ██ RETRO (DeepMind), Atlas (Meta)                     — Scale + pre-training
      █  EmbedKGQA, UniKGQA                                 — KG + dense retrieval
2023  ████ FiD→HyDE, IRCoT, FLARE, KAPING, StructGPT        — RAG++ and KG-prompting
      ████ ToG, RoG                                          — LLM walks KG iteratively
2024  ████ Self-RAG, RAPTOR, CRAG, Adaptive RAG              — Adaptive / hierarchical RAG
      ████ GNN-RAG, SubgraphRAG, HippoRAG                    — Subgraph retrieval
      ████ Microsoft GraphRAG, LightRAG, Fast GraphRAG        — Community / index-based GraphRAG
2025  ███ TraceRAG (ours), ToG 2.0, Edge (Lazy GraphRAG)    — Supervision + efficiency
```

---

## 1. Foundational RAG (2020–2022)

### 1.1 RAG — Retrieval-Augmented Generation
| Field | Detail |
|-------|--------|
| **Authors** | Patrick Lewis, Ethan Perez, et al. (Facebook AI) |
| **Venue** | NeurIPS 2020 |
| **Paper** | https://arxiv.org/abs/2005.11401 |
| **Code** | https://github.com/huggingface/transformers (integrated) |

**Key idea**: Combines a pre-trained seq2seq model (BART) as parametric memory with a dense vector index of Wikipedia (DPR) as non-parametric memory. Two variants: RAG-Sequence (retrieve once per sequence) and RAG-Token (retrieve per token). Marginalizes over top-K retrieved documents.

**Why it matters**: Coined the term "RAG" and established the standard retrieve-then-generate paradigm. All subsequent work in this survey builds on this formulation.

**Limitations**: Fixed retrieval corpus; no iterative refinement; retrieval and generation not jointly trained end-to-end.

---

### 1.2 REALM — Retrieval-Augmented Language Model Pre-Training
| Field | Detail |
|-------|--------|
| **Authors** | Kelvin Guu, Kenton Lee, et al. (Google Research) |
| **Venue** | ICML 2020 |
| **Paper** | https://arxiv.org/abs/2002.08909 |
| **Code** | https://github.com/google-research/language/tree/master/language/realm |

**Key idea**: Pre-trains the retriever and language model jointly end-to-end using an unsupervised masked language modeling objective. The retriever selects Wikipedia documents; the LM conditions on them to predict masked tokens.

**Why it matters**: First to show that retrieval can be learned during pre-training, not just fine-tuning. Demonstrated the knowledge retriever can learn world knowledge implicitly.

---

### 1.3 FiD — Fusion-in-Decoder
| Field | Detail |
|-------|--------|
| **Authors** | Gautier Izacard, Edouard Grave (FAIR) |
| **Venue** | EACL 2021 |
| **Paper** | https://arxiv.org/abs/2007.01282 |
| **Code** | https://github.com/facebookresearch/FiD |

**Key idea**: Encodes each retrieved passage independently with T5 encoder (cheap, parallelizable), then fuses all encoded representations in the decoder via cross-attention. Scales to 100 passages without quadratic cost.

**Why it matters**: Showed that reading more documents nearly always helps. Became a dominant open-domain QA architecture; the "encode separately, attend jointly" design influenced later graph RAG architectures.

---

### 1.4 RETRO — Retrieval-Enhanced Transformer
| Field | Detail |
|-------|--------|
| **Authors** | Sebastian Borgeaud, et al. (DeepMind) |
| **Venue** | ICML 2022 |
| **Paper** | https://arxiv.org/abs/2112.04426 |

**Key idea**: Retrieval integrated directly into Transformer pre-training at scale (7B parameters). Uses chunked cross-attention to attend over a 2-trillion token retrieval database. Matches GPT-3 performance with 25× fewer parameters on language modeling.

**Why it matters**: Demonstrated that retrieval can substitute for parametric capacity at large scale; retrieval-augmented pre-training is competitive with pure scaling.

---

### 1.5 Atlas — Few-shot Learning with Retrieval Augmented Language Models
| Field | Detail |
|-------|--------|
| **Authors** | Gautier Izacard, Patrick Lewis, et al. (Meta AI) |
| **Venue** | JMLR 2023 |
| **Paper** | https://arxiv.org/abs/2208.03299 |
| **Code** | https://github.com/facebookresearch/atlas |

**Key idea**: Jointly pre-trains retriever (Contriever) and reader (T5) at scale (11B). Shows few-shot performance competitive with much larger models on knowledge-intensive tasks.

**Why it matters**: Establishes the "joint training" recipe for large-scale RAG. Contriever (the retriever component) became a widely-used off-the-shelf dense retriever.

---

## 2. Advanced RAG Techniques (2023–2024)

### 2.1 HyDE — Hypothetical Document Embeddings
| Field | Detail |
|-------|--------|
| **Authors** | Luyu Gao, et al. (CMU) |
| **Venue** | ACL 2023 |
| **Paper** | https://arxiv.org/abs/2212.10496 |

**Key idea**: Instead of embedding the query directly, use an LLM to generate a hypothetical answer/document first, then embed that for retrieval. The hypothesis is often closer to gold documents in embedding space than the query alone.

**Why it matters**: Simple zero-shot technique that improves retrieval quality. Influential for showing LLM generation can bootstrap retrieval.

---

### 2.2 IRCoT — Interleaving Retrieval with Chain-of-Thought
| Field | Detail |
|-------|--------|
| **Authors** | Harsh Trivedi, et al. (Stony Brook / Allen AI) |
| **Venue** | ACL 2023 |x
| **Paper** | https://arxiv.org/abs/2212.10509 |
| **Code** | https://github.com/StonyBrookNLP/ircot |

**Key idea**: Alternates between chain-of-thought (CoT) reasoning steps and retrieval calls — each CoT step generates a partial reasoning trace that is used as a new query for the next retrieval round.

**Why it matters**: Pioneer of iterative / multi-hop RAG. The interleaved retrieve-reason loop influenced ToG and RoG in the graph setting.

---

### 2.3 FLARE — Forward-Looking Active REtrieval
| Field | Detail |
|-------|--------|
| **Authors** | Zhengbao Jiang, et al. (CMU) |
| **Venue** | EMNLP 2023 |
| **Paper** | https://arxiv.org/abs/2305.06983 |
| **Code** | https://github.com/jzbjyb/FLARE |

**Key idea**: Actively decides *when* to retrieve during generation by predicting the next sentence and retrieving only when the model is uncertain (low-probability tokens). Retrieval is triggered by generation, not upfront.

**Why it matters**: First to frame retrieval as an on-demand, uncertainty-driven decision during generation.

---

### 2.4 Self-RAG — Self-Reflective Retrieval-Augmented Generation
| Field | Detail |
|-------|--------|
| **Authors** | Akari Asai, et al. (UW / AI2) |
| **Venue** | ICLR 2024 (Oral) |
| **Paper** | https://arxiv.org/abs/2310.11511 |
| **Code** | https://github.com/AkariAsai/self-rag |

**Key idea**: Fine-tunes an LLM to generate special "reflection tokens" (Retrieve, ISREL, ISSUP, ISUSE) that control when to retrieve and how to critically assess retrieved passages and its own outputs. The model itself decides retrieval necessity.

**Why it matters**: Unifies adaptive retrieval and self-critique in one model. Highly influential; introduces the idea of LLM-as-its-own-critic for RAG.

---

### 2.5 RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval
| Field | Detail |
|-------|--------|
| **Authors** | Parth Sarthi, et al. (Stanford) |
| **Venue** | ICLR 2024 |
| **Paper** | https://arxiv.org/abs/2401.18059 |
| **Code** | https://github.com/parthsarthi03/raptor |

**Key idea**: Recursively clusters and summarizes text chunks into a tree structure at multiple abstraction levels. Queries can retrieve both fine-grained leaf nodes and high-level summaries, enabling multi-scale retrieval.

**Why it matters**: Hierarchical document organization addresses the "lost in the middle" problem for long-context reasoning. Directly inspired HippoRAG and MSFT GraphRAG community summarization.

---

### 2.6 CRAG — Corrective Retrieval Augmented Generation
| Field | Detail |
|-------|--------|
| **Authors** | Shi-Qi Yan, et al. |
| **Venue** | ICLR 2024 Workshop / arXiv 2024 |
| **Paper** | https://arxiv.org/abs/2401.15884 |
| **Code** | https://github.com/HuskyInSalt/CRAG |

**Key idea**: Trains a lightweight retrieval evaluator that judges relevance of retrieved documents. If documents are irrelevant, it triggers web search and applies a "knowledge refinement" decompose-filter-recompose pipeline.

**Why it matters**: Introduces the concept of retrieval *correction* — verifying and fixing bad retrievals before generation.

---

### 2.7 Adaptive RAG
| Field | Detail |
|-------|--------|
| **Authors** | Soyeong Jeong, et al. (KAIST) |
| **Venue** | NAACL 2024 |
| **Paper** | https://arxiv.org/abs/2403.14403 |
| **Code** | https://github.com/starsuzi/Adaptive-RAG |

**Key idea**: Trains a classifier to select among three strategies — (1) no retrieval, (2) single-step retrieval, (3) multi-step iterative retrieval — based on query complexity. Uses self-collected labels from existing QA systems.

**Why it matters**: Formalizes query complexity routing as a learned decision, avoiding over-retrieval for simple queries.

---

## 3. Knowledge Graph QA — Pre-LLM Era (2019–2022)

These methods set the stage for GraphRAG but rely on structured KGQA pipelines (entity linking + subgraph extraction + neural reasoning) rather than prompting LLMs.

### 3.1 EmbedKGQA
| Field | Detail |
|-------|--------|
| **Authors** | Apoorv Saxena, et al. (IIT Delhi) |
| **Venue** | ACL 2020 |
| **Paper** | https://arxiv.org/abs/2106.09996 |
| **Code** | https://github.com/malllabiisc/EmbedKGQA |

**Key idea**: Embeds multi-hop KGQA as a KG completion problem. Uses KG embeddings (ComplEx) to bridge incomplete KGs; answers are found by maximizing entity-answer similarity in embedding space.

**Why it matters**: Showed KG embeddings can handle incomplete KGs for multi-hop QA. Benchmark on MetaQA.

---

### 3.2 UniKGQA — Unified Retrieval and Reasoning for KG QA
| Field | Detail |
|-------|--------|
| **Authors** | Jinhao Jiang, et al. (Tsinghua) |
| **Venue** | ICLR 2023 |
| **Paper** | https://arxiv.org/abs/2212.00959 |
| **Code** | https://github.com/RUCAIBox/UniKGQA |

**Key idea**: Unifies subgraph retrieval and reasoning in a single model by sharing parameters across both tasks. A retrieval module selects relevant triples; a reasoning module aggregates them. Jointly trained on WebQSP and CWQ.

**Why it matters**: Precursor to SubgraphRAG — demonstrated that shared-encoder retrieve + reason outperforms pipeline methods.

---

## 4. GraphRAG: KG-Enhanced LLM Reasoning (2023–2025)

This is the core section. These methods use LLMs as the reasoning backbone but rely on explicit graph traversal or subgraph retrieval to ground multi-hop reasoning.

### 4.1 StructGPT — Structured Interface for LLMs
| Field | Detail |
|-------|--------|
| **Authors** | Jinhao Jiang, et al. (Renmin University) |
| **Venue** | EMNLP 2023 |
| **Paper** | https://arxiv.org/abs/2305.09645 |
| **Code** | https://github.com/RUCAIBox/StructGPT |

**Key idea**: Provides LLMs with a structured interface (Iterative Reading-then-Reasoning, IRR) that wraps KG/table/database access as callable functions. The LLM calls these APIs iteratively to gather structured evidence before answering.

**Why it matters**: Early work on tool-augmented LLMs for structured reasoning. Direct precursor to ToG/RoG-style KG traversal.

---

### 4.2 KAPING — Knowledge-Augmented Language Model Prompting
| Field | Detail |
|-------|--------|
| **Authors** | Jinheon Baek, et al. (KAIST) |
| **Venue** | ACL 2023 Findings |
| **Paper** | https://arxiv.org/abs/2306.04136 |

**Key idea**: Retrieves relevant KG triples for a query and directly injects them as text into the LLM prompt ("soft prompting" with KG facts). No fine-tuning needed. Evaluates whether LLMs can use KG facts better than their parametric memory.

**Why it matters**: Establishes the simplest GraphRAG baseline — KG triples as prompt context — against which later methods are compared.

---

### 4.3 ToG — Think-on-Graph
| Field | Detail |
|-------|--------|
| **Authors** | Jiashuo Sun, et al. (HKUST + Baidu) |
| **Venue** | ICLR 2024 |
| **Paper** | https://arxiv.org/abs/2307.07697 |
| **Code** | https://github.com/HKUST-KnowComp/Think-on-Graph |

**Key idea**: Uses an LLM as an agent that *walks* the KG beam-search style: at each hop, the LLM selects which relations to follow from the current entity set, expands the frontier, prunes irrelevant paths, and repeats until an answer is found. The KG is the "scratchpad."

**Why it matters**: Demonstrated that letting LLMs perform explicit multi-hop reasoning over KG topology (rather than ingesting retrieved triples) dramatically improves complex QA. Highly influential; spawned ToG 2.0 and many follow-ups.

**Limitations**: Expensive — each reasoning step calls the LLM multiple times; latency scales with hop count and beam width.

---

### 4.4 RoG — Reasoning-on-Graph
| Field | Detail |
|-------|--------|
| **Authors** | Linhao Luo, et al. (Monash University) |
| **Venue** | ICLR 2024 |
| **Paper** | https://arxiv.org/abs/2310.01061 |
| **Code** | https://github.com/RManLuo/reasoning-on-graphs |

**Key idea**: Two-stage pipeline: (1) **Planning** — fine-tune an LLM to generate *relation paths* (e.g., `born_in → located_in`) as a reasoning plan; (2) **Reasoning** — retrieve all KG paths matching the plan and generate the final answer. The plan guides faithful KG traversal.

**Why it matters**: Introduces the plan-then-retrieve paradigm for KG reasoning. Relation paths are interpretable and constrain retrieval to semantically coherent subgraphs. State-of-the-art on WebQSP and CWQ at the time.

**Limitations**: Plan generation requires fine-tuning; plan-KG alignment can fail for unseen relation patterns.

---

### 4.5 GNN-RAG
| Field | Detail |
|-------|--------|
| **Authors** | Costas Mavromatis, George Karypis (UMN / NVIDIA) |
| **Venue** | arXiv 2024 |
| **Paper** | https://arxiv.org/abs/2405.20139 |
| **Code** | https://github.com/cmavro/GNN-RAG |

**Key idea**: Uses a GNN (specifically LHNN) trained on KGQA to find candidate answer paths, then extracts the shortest paths between question entities and GNN-predicted answers as the RAG context for an LLM. Bridges learned GNN reasoning with LLM generation.

**Why it matters**: Shows GNN-predicted candidates produce better retrieval seeds than topic entity expansion alone. Combining GNN + LLM reasoning outperforms each in isolation.

---

### 4.6 SubgraphRAG — Simple is Effective
| Field | Detail |
|-------|--------|
| **Authors** | Mufei Li, Siqi Miao, Pan Li |
| **Venue** | ICLR 2025 |
| **Paper** | https://arxiv.org/abs/2410.20724 |
| **Code** | https://github.com/Graph-COM/SubgraphRAG |

**Full title**: *Simple is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation*

**Key idea**: Lightweight MLP retriever that scores each KG triple independently using [query, head, relation, tail] concatenated with DDE (Degree-based Distinguishing Embedding) structural features. Trained with binary BCE on shortest-path weak labels. At inference, returns top-K triples (K=100) as flat context for the LLM.

**Architecture details**:
- Text encoder: GTE-large-en-v1.5 (1024D → projected to 4D)
- DDE: non-learnable mean aggregation from topic entities, relation-blind
- Retrieval: per-triple scoring, no iterative graph traversal at inference

**Results** (backbone-dependent):
| Backbone | WebQSP Hits@1 | WebQSP F1 | CWQ Hits@1 | CWQ F1 |
|----------|--------------|-----------|------------|--------|
| Llama3.1-8B | ~72 | ~63 | ~57 | ~47 |
| GPT-4o-mini | 87.4 | 78.3 | ~65 | ~58 |
| GPT-4o | 86.4 | 77.6 | 68.9 | 66.0 |

**Why it matters**: Shows that a small, fast MLP retriever can match or exceed iterative LLM-walk methods (ToG, RoG) at a fraction of the inference cost. The key finding is "simple retrieval + strong LLM > complex retrieval + weak LLM."

**Limitations** (motivating our work):
1. DDE is relation-blind — same propagation weight regardless of relation type or query
2. Weak supervision from shortest-path heuristics may not reflect LLM's actual reasoning needs
3. Retrieval quality is the bottleneck when using smaller LLMs as backbone

---

### 4.7 G-Retriever — Graph Retrieval-Augmented Generation for Textual Graph QA
| Field | Detail |
|-------|--------|
| **Authors** | Xiaoxin He, et al. (NUS / Oxford) |
| **Venue** | NeurIPS 2024 |
| **Paper** | https://arxiv.org/abs/2402.07630 |
| **Code** | https://github.com/XiaoxinHe/G-Retriever |

**Key idea**: Addresses QA over *text-attributed graphs* (nodes and edges have text descriptions, e.g., scene graphs, knowledge graphs with rich descriptions). Uses a Prize-Collecting Steiner Tree (PCST) algorithm to extract a minimal connected subgraph relevant to the query, then encodes it with a graph transformer (GraphGPS) and feeds the graph embedding to an LLM via soft prompting (cross-attention).

**Why it matters**: Bridges structured graph reasoning and LLM generation for graphs that are neither pure text corpora nor pure symbolic KGs. Demonstrates that graph topology (not just retrieved text) provides useful inductive bias for LLM reasoning.

**Distinction from SubgraphRAG**: G-Retriever targets text-attributed graphs and uses differentiable graph-neural encoding; SubgraphRAG targets symbolic KGs (Freebase) with a triple-scoring MLP.

---

### 4.8 ChatKBQA — Generate-then-Retrieve for KGQA
| Field | Detail |
|-------|--------|
| **Authors** | Haoran Luo, et al. (Beijing Institute of Technology) |
| **Venue** | ACL 2024 Findings |
| **Paper** | https://arxiv.org/abs/2304.09167 |
| **Code** | https://github.com/LHRLAB/ChatKBQA |

**Key idea**: Fine-tunes an LLM (LLaMA-2) to generate SPARQL-like logical forms (S-expressions) for KGQA. Applies entity linking and relation retrieval as a post-processing step to ground the generated logical form onto the actual KG entities/relations (entity disambiguation). Avoids the brittleness of direct KG traversal by working in the logical form space.

**Why it matters**: Shows that LLM-based logical form generation outperforms pipeline SPARQL parsers and KG-walk methods on WebQSP/CWQ when combined with effective entity grounding.

---

### 4.9 DoG — Debate on Graph
| Field | Detail |
|-------|--------|
| **Authors** | Yufei He, et al. |
| **Venue** | AAAI 2025 |
| **Paper** | https://arxiv.org/abs/2409.03155 |

**Key idea**: Two LLM agents (a "proponent" and an "opponent") debate over KG-retrieved evidence to collaboratively arrive at the correct answer. Each agent retrieves and interprets different subgraphs, and a judge LLM resolves the debate. The debate format forces explicit evidence attribution and reduces hallucination.

**Results**: 91.0 Hits@1 on WebQSP with GPT-4; 58.2/56.0 on CWQ (GPT-3.5/GPT-4). Highest WebQSP Hits@1 with proprietary LLMs.

**Why it matters**: Introduces multi-agent debate as a retrieval-verification mechanism for KGQA. Complementary to retrieval improvements — the debate can correct retrieval errors via agent disagreement.

---

### 4.10 RoE — Reasoning by Exploration
| Field | Detail |
|-------|--------|
| **Authors** | Various (preprint 2025) |
| **Venue** | arXiv 2025 |
| **Paper** | https://arxiv.org/abs/2510.07484 |

**Key idea**: Fine-tunes a small LLM (LLaMA-3.1-8B) to interactively explore the KG — generating action sequences (explore entity, follow relation, verify answer) as structured reasoning traces. Uses reinforcement learning with answer-match reward to learn efficient KG traversal strategies without supervision on intermediate steps.

**Results**: 89.1 Hits@1 / 74.8 F1 on WebQSP; 66.5 / 53.2 on CWQ with LLaMA-3.1-8B. Best published small-LLM results at time of release.

**Why it matters**: Demonstrates that RL-trained KG exploration with a small LLM can approach GPT-4-level performance. The RL formulation avoids the need for curated path supervision.

---

### 4.11 RPO-RAG — Reward-guided Path Optimization for RAG
| Field | Detail |
|-------|--------|
| **Authors** | Various (preprint 2026) |
| **Venue** | arXiv 2026 |
| **Paper** | https://arxiv.org/abs/2601.19225 |

**Key idea**: Combines KG path retrieval with a reward model that scores retrieved paths by their downstream QA utility. Uses DPO/RLHF-style training to teach the LLM (LLaMA-3.1-8B) to both retrieve better paths and reason over them. The reward signal comes from QA answer correctness.

**Results**: **89.9 Hits@1 / 81.3 F1** on WebQSP; **72.3 / 64.5** on CWQ — current best published results with an open-source LLM (as of early 2026).

**Why it matters**: Shows that reward-guided training of the full retrieve+reason pipeline outperforms separate optimization of retrieval and generation. Represents the current SOTA direction for KGQA with small LLMs.

---

### 4.13 DALK — Domain-Adaptive LLM + KG
| Field | Detail |
|-------|--------|
| **Authors** | Dawei Li, et al. |
| **Venue** | arXiv 2024 |
| **Paper** | https://arxiv.org/abs/2405.04819 |

**Key idea**: Builds a domain-specific KG from medical literature, then augments LLM responses with retrieved KG triples for medical QA tasks. Adapts the standard KGQA pipeline to specialized domains (e.g., Alzheimer's QA).

**Why it matters**: Demonstrates GraphRAG applicability beyond Freebase, showing structured domain KGs can significantly improve factual accuracy in specialized domains.

---

### 4.14 ToG 2.0 — Think-on-Graph with Deep Thinking
| Field | Detail |
|-------|--------|
| **Authors** | Jiashuo Sun, et al. (follow-up) |
| **Venue** | arXiv 2024 |
| **Paper** | https://arxiv.org/abs/2407.10805 |
| **Code** | https://github.com/HKUST-KnowComp/Think-on-Graph |

**Key idea**: Extends ToG with *deep thinking* — incorporates CoT reasoning within each beam step and introduces a "think-before-you-walk" mechanism that prunes impossible paths using LLM self-consistency before expansion.

**Why it matters**: Improves ToG's accuracy and efficiency. Demonstrates that structured KG traversal + LLM self-reflection is a powerful combination.

---

### 4.15 KG-CoT — Knowledge Graph Chain-of-Thought
| Field | Detail |
|-------|--------|
| **Authors** | Multiple groups (multiple concurrent works use this name) |
| **Venue** | Various 2023–2024 |

**Key idea**: General paradigm — interleave explicit KG triple retrieval with chain-of-thought reasoning steps, similar to IRCoT but grounded in structured KG queries rather than free-text retrieval.

**Why it matters**: Represents a whole class of methods; the core idea that structured KG hops can replace unstructured text retrieval in a CoT loop.

---

## 5. Community & Global GraphRAG Systems (2024–2025)

These systems move beyond KGQA (structured Freebase/Wikidata) toward building *ad hoc* graphs from unstructured text corpora.

### 5.1 Microsoft GraphRAG
| Field | Detail |
|-------|--------|
| **Authors** | Darren Edge, Ha Trinh, et al. (Microsoft Research) |
| **Venue** | arXiv 2024 |
| **Paper** | https://arxiv.org/abs/2404.16130 |
| **Code** | https://github.com/microsoft/graphrag |
| **Blog** | https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/ |

**Key idea**: Builds a KG from unstructured text using an LLM (entity/relation extraction), then applies hierarchical Leiden community detection to create a community tree. Community summaries at each level enable *global* queries (e.g., "What are the main themes?") that flat RAG cannot answer. Local queries use entity-centric subgraph retrieval.

**Two modes**:
- **Local**: entity-centric subgraph + related text chunks → LLM answer
- **Global**: community summaries at chosen resolution → map-reduce LLM answer

**Why it matters**: First system to tackle "global" sensemaking queries over entire document corpora. Showed KG + hierarchical summarization is necessary for corpus-level analysis. Hugely influential in production RAG.

**Limitations**: Very LLM-call heavy (indexing cost is ~$10–50 per small corpus); community summaries can be noisy.

---

### 5.2 HippoRAG
| Field | Detail |
|-------|--------|
| **Authors** | Bernal Jiménez Gutiérrez, et al. (OSU) |
| **Venue** | NeurIPS 2024 |
| **Paper** | https://arxiv.org/abs/2405.14831 |
| **Code** | https://github.com/OSU-NLP-Group/HippoRAG |

**Key idea**: Inspired by the hippocampal memory model in neuroscience. Builds a "schematic" KG from documents (entity-relation extraction), then uses Personalized PageRank (PPR) seeded from query-relevant entities to spread activation through the graph and retrieve contextually connected passages.

**Two-level memory**:
- Neocortex (LLM encoder): semantic similarity for entity grounding
- Hippocampus (KG + PPR): associative multi-hop connectivity

**Why it matters**: PPR-based retrieval elegantly captures multi-hop connections without explicit path enumeration. Shows that graph-connectivity signals significantly outperform dense retrieval for associative multi-hop QA. Competitive with IRCoT-style iterative retrieval at lower cost.

---

### 5.3 LightRAG
| Field | Detail |
|-------|--------|
| **Authors** | Zirui Guo, et al. (HKU) |
| **Venue** | arXiv 2024 |
| **Paper** | https://arxiv.org/abs/2410.05779 |
| **Code** | https://github.com/HKUDS/LightRAG |

**Key idea**: Builds a dual-level KG index: (1) entity-level nodes and (2) relation-level hyperedges. Supports two retrieval modes: *low-level* (specific entity queries) and *high-level* (abstract concept queries). Uses incremental graph updates for dynamic corpora without full re-indexing.

**Why it matters**: Addresses MSFT GraphRAG's key weaknesses: (1) dynamic updates without full rebuild, (2) handles both specific and abstract queries in one index. Much cheaper to build/update than GraphRAG.

**Code quality**: Well-documented; actively maintained; popular for practitioners.

---

### 5.4 Fast GraphRAG
| Field | Detail |
|-------|--------|
| **Authors** | Circlemind AI |
| **Venue** | Open source, 2024 |
| **Code** | https://github.com/circlemind-ai/fast-graphrag |

**Key idea**: Re-implements Microsoft GraphRAG with significantly reduced LLM call overhead. Uses lightweight heuristics for community detection and summarization, replacing expensive LLM-based summarization with embedding-based clustering. Reduces indexing cost by ~10×.

**Why it matters**: Practical alternative to MSFT GraphRAG for resource-constrained settings. Demonstrates that the KG structure (not the LLM summarization) provides most of the benefit.

---

### 5.5 Nano-GraphRAG
| Field | Detail |
|-------|--------|
| **Venue** | Open source, 2024 |
| **Code** | https://github.com/gusye1234/nano-graphrag |

**Key idea**: Minimal, readable Python re-implementation of GraphRAG in ~1000 lines. Educational implementation designed for understanding and extension.

**Why it matters**: Highly useful for researchers who want to build on or modify the GraphRAG pipeline without navigating MSFT's production codebase.

---

### 5.6 PropRAG — Propagation-based Retrieval-Augmented Generation
| Field | Detail |
|-------|--------|
| **Authors** | Various |
| **Venue** | EMNLP 2025 |

**Key idea**: Propagates query relevance scores through a graph of text chunks using belief propagation, treating the retrieval corpus as a Markov Random Field. Each document node passes relevance signals to its neighbors via edges representing shared entities or semantic similarity. Enables global context awareness without building an explicit KG.

**Results**: Competitive on HotpotQA, 2WikiMultiHopQA, and MuSiQue (~74+ F1 on 2Wiki), though evaluation settings vary.

**Why it matters**: Shows that graph-structured message passing over text chunks (without explicit entity extraction) captures multi-hop connections. A middle ground between pure dense retrieval and full KG-based GraphRAG.

---

### 5.7 KG-o1 — Agentic KG Reasoning with o1-style Thinking
| Field | Detail |
|-------|--------|
| **Authors** | Various |
| **Venue** | arXiv 2024 |

**Key idea**: Applies chain-of-thought reasoning in the style of OpenAI o1 to KG question answering. The agent iteratively decomposes the question, queries the KG for sub-answers, and synthesizes a final answer. Combines o1-style long thinking traces with structured KG lookups.

**Results**: 74.45 F1 / 62.4 EM on 2WikiMultiHopQA — strong among LLM agent methods.

**Why it matters**: Demonstrates that long-horizon reasoning (o1-style) combined with KG grounding significantly outperforms standard RAG on compositional multi-hop questions.

---

### 5.8 Microsoft LazyGraphRAG / Edge (2025)
| Field | Detail |
|-------|--------|
| **Authors** | Darren Edge, et al. (Microsoft Research) |
| **Venue** | Blog post / arXiv 2025 |
| **Blog** | https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/ |

**Key idea**: "Lazy" indexing — defers expensive community summarization to query time, building only a lightweight entity extraction index upfront. Only summarizes communities that are relevant to a specific query, dramatically cutting indexing cost.

**Why it matters**: Addresses the #1 criticism of GraphRAG (indexing cost). Demonstrates that most community summaries are never queried, so pre-computation is wasteful.

---

## 6. Survey Papers

### 6.1 A Survey on RAG for LLMs
| Field | Detail |
|-------|--------|
| **Authors** | Yunfan Gao, et al. |
| **Venue** | arXiv 2023 |
| **Paper** | https://arxiv.org/abs/2312.10997 |

Comprehensive 30+ page survey covering RAG stages (indexing, retrieval, augmentation, generation), training strategies, evaluation, and open challenges. Introduces the Naive/Advanced/Modular RAG taxonomy that is now widely used.

---

### 6.2 Modular RAG
| Field | Detail |
|-------|--------|
| **Authors** | Yunfan Gao, et al. |
| **Venue** | arXiv 2024 |
| **Paper** | https://arxiv.org/abs/2407.21059 |

Extends the survey above with a "modular" design perspective — treats RAG as composable modules (routing, scheduling, fusion, etc.) rather than a fixed pipeline. Introduces a RAG flow grammar.

---

### 6.3 Graph RAG Survey: KG + LLM Synergies
| Field | Detail |
|-------|--------|
| **Authors** | Shirui Pan, et al. (Griffith / Monash) |
| **Venue** | IEEE TKDE 2024 |
| **Paper** | https://arxiv.org/abs/2306.08302 |

Survey on unifying knowledge graphs and LLMs, covering three paradigms: (1) KG-enhanced LLMs, (2) LLM-augmented KGs, (3) synergistic KG+LLM. Directly relevant taxonomy for GraphRAG positioning.

---

### 6.4 Retrieval-Augmented Generation for AI-Generated Content: A Survey
| Field | Detail |
|-------|--------|
| **Authors** | Penghao Zhao, et al. |
| **Venue** | arXiv 2024 |
| **Paper** | https://arxiv.org/abs/2402.19473 |

Broadest RAG survey, covering text, code, image, audio, and video generation. Useful for understanding how text-domain RAG findings generalize.

---

## 7. Benchmarks & Datasets

| Dataset | Domain | Hops | KG | Size | Paper |
|---------|--------|------|----|------|-------|
| **WebQSP** | Freebase KGQA | 1–2 | Freebase | 4,737 test | [Yih et al., 2016](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering/) |
| **CWQ** (ComplexWebQ) | Freebase KGQA | 2–4 | Freebase | 7,500 test | [Talmor & Berant, 2018](https://arxiv.org/abs/1803.06643) |
| **MetaQA** | WikiMovies | 1/2/3-hop | WikiMovies KG | 114K total | [Zhang et al., 2018](https://arxiv.org/abs/1709.04071) |
| **MQA-3hop** | WikiMovies 3-hop | 3 | WikiMovies KG | ~14K | Same |
| **HotpotQA** | Wikipedia | 2 | — (text) | 113K | [Yang et al., 2018](https://arxiv.org/abs/1809.09600) |
| **2WikiMultiHopQA** | Wikipedia | 2–4 | Wikipedia | 167K | [Ho et al., 2020](https://arxiv.org/abs/2012.01060) |
| **MuSiQue** | Wikipedia | 2–4 | — (text) | 20K | [Trivedi et al., 2022](https://arxiv.org/abs/2108.00573) |
| **KGQA (FreebaseQA)** | Freebase | 1 | Freebase | 53K | [Jiang et al., 2019](https://arxiv.org/abs/1903.04521) |

---

## 9. State-of-the-Art Leaderboards

> Scores collected from papers published through early 2026. **Bold** = best in category. All numbers are on the test set unless noted.

---

### 9.1 WebQSP

**Metrics**: Hits@1 (primary), F1. KG: Freebase. Task: 1–2 hop KGQA.

| Method | Hits@1 | F1 | Backbone | Type | Year |
|--------|--------|----|----------|------|------|
| KV-Mem | 46.7 | 38.6 | — | Embedding | 2016 |
| GraftNet | 66.7 | 62.4 | — | GNN | 2018 |
| EmbedKGQA | 66.6 | — | — | KG Embedding | 2020 |
| NSM | 68.7 | 62.8 | — | GNN | 2021 |
| TransferNet | 71.4 | — | — | GNN | 2021 |
| NSM+h | 74.3 | 67.4 | — | GNN | 2021 |
| UniKGQA | 77.2 | 72.2 | — | GNN | 2022 |
| ReaRev | 77.5 | 72.8 | — | GNN | 2022 |
| ToG | 76.2 | — | ChatGPT | LLM+KG walk | 2024 |
| ToG | 82.6 | — | GPT-4 | LLM+KG walk | 2024 |
| RoG | 85.7 | 70.8 | LLaMA-2-7B (FT) | Plan+retrieve | 2024 |
| GNN-RAG | 85.7 | 71.3 | LLaMA-2-7B | GNN+LLM | 2024 |
| GNN-RAG+RA | 87.0 | 73.5 | LLaMA-2-7B | GNN+LLM+text | 2024 |
| SubgraphRAG | 87.4 | 78.3 | GPT-4o-mini | MLP+LLM | 2025 |
| DoG | 88.6 | — | GPT-3.5 | Debate+KG | 2025 |
| RoE | 89.1 | 74.8 | LLaMA-3.1-8B (FT) | Explore+KG | 2025 |
| DoG | 91.0 | — | GPT-4 | Debate+KG | 2025 |
| **RPO-RAG** | **89.9** | **81.3** | LLaMA-3.1-8B (FT) | Reward+KG | 2026 |

> **Note**: Hits@1 and F1 measure different things. Hits@1 = macro recall over answer entities; F1 = token overlap. SubgraphRAG achieves best F1 among non-proprietary-LLM methods.

---

### 9.2 CWQ (ComplexWebQuestions)

**Metrics**: Hits@1, F1. KG: Freebase. Task: 2–4 hop + set operations.

| Method | Hits@1 | F1 | Backbone | Type | Year |
|--------|--------|----|----------|------|------|
| KV-Mem | 21.1 | — | — | Embedding | 2016 |
| GraftNet | 36.8 | 32.7 | — | GNN | 2018 |
| NSM | 47.6 | 42.4 | — | GNN | 2021 |
| UniKGQA | 51.2 | 49.1 | — | GNN | 2022 |
| ReaRev | 53.3 | 49.7 | — | GNN | 2022 |
| ToG | 58.9 | — | ChatGPT | LLM+KG walk | 2024 |
| RoG | 62.6 | 56.2 | LLaMA-2-7B (FT) | Plan+retrieve | 2024 |
| GNN-RAG | 66.8 | 59.4 | LLaMA-2-7B | GNN+LLM | 2024 |
| ToG | 67.6 | — | GPT-4 | LLM+KG walk | 2024 |
| GNN-RAG+RA | 68.7 | 60.4 | LLaMA-2-7B | GNN+LLM+text | 2024 |
| SubgraphRAG | 68.9 | 66.0 | GPT-4o | MLP+LLM | 2025 |
| RoE | 66.5 | 53.2 | LLaMA-3.1-8B (FT) | Explore+KG | 2025 |
| **RPO-RAG** | **72.3** | **64.5** | LLaMA-3.1-8B (FT) | Reward+KG | 2026 |

> **Note**: CWQ is significantly harder than WebQSP due to compositional (count, superlative, comparative) and multi-hop questions. SubgraphRAG with GPT-4o achieves best F1 among all published results.

---

### 9.3 MetaQA 3-hop

**Metric**: Hits@1. KG: WikiMovies. Task: 3-hop chain reasoning.

| Method | Hits@1 | Type | Year |
|--------|--------|------|------|
| KV-Mem | 48.9 | Embedding | 2016 |
| GraftNet | 77.7 | GNN | 2018 |
| EmbedKGQA | 94.8 | KG Embedding | 2020 |
| NSM | 98.9 | GNN | 2021 |
| UniKGQA | 99.9 | GNN | 2022 |
| **TransferNet** | **100.0** | GNN | 2021 |
| KB-BINDER | 99.5 | LLM+KG (Codex) | 2023 |
| KG-GPT | 94.0 | LLM+KG (GPT-3.5) | 2023 |
| StructGPT | 87.0 | LLM+KG (GPT-3.5) | 2023 |
| DoG | 96.0 | GPT-4 | 2025 |

> **Note**: MetaQA-3hop is **effectively solved** by trained GNN methods (TransferNet achieves 100% since 2021). The interesting challenge is **zero-shot / cross-domain transfer**, where LLM-based methods are measured. For our work (TraceRAG), MetaQA-3hop serves as a generalization benchmark.

---

### 9.4 HotpotQA

**Metric**: Joint F1 = Answer F1 × Supporting Fact F1 (official leaderboard). Also reported: Answer F1 alone in many LLM-era papers (not directly comparable).

#### Distractor Setting (10 candidate paragraphs provided)

| Method | Joint F1 | Type | Year |
|--------|----------|------|------|
| Baseline (original) | ~59 | MRC | 2018 |
| Smoothing R3 (ALBERT) | 76.69 | MRC | 2022 |
| FE2H (ALBERT) | 76.54 | MRC | 2022 |
| PEI | 77.84 | Dense Retrieval | 2024 |
| **Beam Retrieval** | **77.54** | Dense Retrieval | 2023 |

#### Full-Wiki Setting (open-domain, retrieval from Wikipedia)

| Method | Joint F1 | Answer F1 | Type | Year |
|--------|----------|-----------|------|------|
| MDR | — | ~75.3 | Dense Retrieval | 2020 |
| TPRR | 70.83 | — | Dense Retrieval | 2021 |
| **AISO** | **72.00** | — | Dense Retrieval | 2021 |
| Chain-of-Skills | 71.65 | — | Dense Retrieval | 2023 |
| IRCoT (GPT-3) | — | ~60.7 | Iterative RAG | 2023 |
| HippoRAG (ColBERTv2) | — | ~55.0 | Graph RAG | 2024 |
| HippoRAG + IRCoT | — | ~59.2 | Graph RAG | 2024 |
| PRISM (GPT-4o) | — | ~67.0 | Agentic RAG | 2024 |

> **Note**: The official leaderboard (hotpotqa.github.io) has not been actively updated since ~2023. LLM-era papers (2024+) typically report Answer F1 only in open-domain settings, which is **not comparable** to leaderboard Joint F1. AISO at 72.00 Joint F1 is the strongest fully reproducible result on the leaderboard.

---

### 9.5 2WikiMultiHopQA

**Metric**: F1 (Answer). Open-domain Wikipedia retrieval.

| Method | F1 | EM | Type | Year |
|--------|----|----|------|------|
| Baseline (original) | ~36 | ~27 | MRC | 2020 |
| DecomP | 70.8 | — | Decomposition | 2022 |
| IRCoT (GPT-3) | 68.0 | 57.7 | Iterative RAG | 2023 |
| HippoRAG (ColBERTv2) | 59.5 | — | Graph RAG | 2024 |
| HippoRAG + IRCoT | 62.7 | — | Graph RAG | 2024 |
| PRISM | 57.0 | 48.6 | Agentic RAG | 2024 |
| KG-o1 | 74.45 | 62.4 | LLM Agent | 2024 |
| **PropRAG** | **~74–90*** | — | Graph RAG | 2025 |

> *PropRAG numbers vary across evaluation setups; the 74–90 range reflects different retrieval conditions. IRCoT at 68.0 and KG-o1 at 74.45 are reliable open-domain baselines.
>
> **Warning**: 2WikiMultiHopQA has known annotation artifacts (bridge vs. comparison questions, entity-matching shortcuts). High F1 numbers should be interpreted with care.

---

### 9.6 MuSiQue

**Metric**: Answer F1 on MuSiQue-Ans (test). Widely considered the **hardest** open-domain multi-hop benchmark.

| Method | F1 | EM | Type | Year |
|--------|----|----|------|------|
| Baseline (original) | ~21 | ~16 | MRC | 2022 |
| DecomP | 30.9 | — | Decomposition | 2022 |
| IRCoT (GPT-3) | 36.5 | 26.5 | Iterative RAG | 2023 |
| PRISM | 41.8 | 31.2 | Agentic RAG | 2024 |
| HippoRAG (ColBERTv2) | 29.8 | — | Graph RAG | 2024 |
| HippoRAG + IRCoT | 33.3 | — | Graph RAG | 2024 |
| Beam Retrieval (dev) | **69.2** | — | Dense Retrieval | 2023 |
| PropRAG | ~52–75* | — | Graph RAG | 2025 |

> *Beam Retrieval's 69.2 uses a dedicated retrieval model on dev set. In a fully open-domain LLM-RAG setting, IRCoT at 36.5 and PRISM at 41.8 are the meaningful comparisons. Human performance is ~91.3 F1.

---

### 9.7 Summary: SOTA Frontier (Early 2026)

| Benchmark | Best Method | Score | Backbone | Notes |
|-----------|-------------|-------|----------|-------|
| WebQSP Hits@1 | DoG | 91.0 | GPT-4 | Proprietary LLM |
| WebQSP Hits@1 (open) | RPO-RAG | 89.9 | LLaMA-3.1-8B FT | Best open-source LLM |
| WebQSP F1 | RPO-RAG | 81.3 | LLaMA-3.1-8B FT | |
| CWQ Hits@1 | RPO-RAG | 72.3 | LLaMA-3.1-8B FT | |
| CWQ F1 | SubgraphRAG | 66.0 | GPT-4o | |
| MetaQA-3hop | TransferNet | 100.0 | — | Saturated since 2021 |
| HotpotQA Joint F1 (dist.) | PEI | 77.84 | — | Leaderboard stale |
| HotpotQA Answer F1 (wiki) | PRISM | ~67.0 | GPT-4o | LLM-era, not Joint F1 |
| 2WikiMultiHop F1 | KG-o1 / PropRAG | ~74–90 | varies | Artifacts in dataset |
| MuSiQue F1 | Beam Retrieval | 69.2 (dev) | — | Not full RAG pipeline |

**Key takeaways for our work (TraceRAG)**:
- On WebQSP/CWQ, there is still headroom vs. RPO-RAG using better retrieval + small LLMs
- MetaQA-3hop is only interesting as a **zero-shot transfer** target, not a training benchmark
- For LLM-era KGQA, retrieval quality with small LLMs is the bottleneck (SubgraphRAG finding)

---

## 10. Taxonomy & Positioning Map

```
                    RETRIEVAL UNIT
                 Triples ←————————→ Text chunks
                    |                    |
HIGH              ToG, RoG          IRCoT, FLARE
REASONING         GNN-RAG          Self-RAG, CRAG
COMPLEXITY        SubgraphRAG      RAPTOR, Atlas
                    |                    |
LOW               KAPING            Standard RAG
                  HyDE              FiD, REALM
                    ↑
            STRUCTURED (KG)
```

```
GRAPHRAG METHODS by KG source:

  Pre-built KG (Freebase/Wikidata)      Ad-hoc KG (from corpus)
  ┌──────────────────────────────┐       ┌────────────────────────────┐
  │ ToG, RoG, GNN-RAG            │       │ Microsoft GraphRAG          │
  │ SubgraphRAG, KAPING          │       │ LightRAG, HippoRAG          │
  │ StructGPT, UniKGQA           │       │ Fast GraphRAG, RAPTOR       │
  │ DALK (domain KG)             │       │ Nano-GraphRAG               │
  └──────────────────────────────┘       └────────────────────────────┘
```

```
SUPERVISION SIGNAL for KG retrieval methods:

  Weak labels (paths/hops)      Strong labels (LLM-cited)
  ┌────────────────────────┐    ┌────────────────────────────────┐
  │ SubgraphRAG (BCE)       │    │ TraceRAG (ours, InfoNCE)        │
  │ RoG (plan fine-tuning)  │    │ Self-RAG (reflection tokens)    │
  │ GNN-RAG (GNN labels)    │    │                                │
  └────────────────────────┘    └────────────────────────────────┘
```

---

## Reading Order Recommendation

### Track A — General RAG background
1. RAG (Lewis, NeurIPS 2020) — the paradigm
2. FiD (Izacard, EACL 2021) — multi-doc fusion
3. RAG Survey (Gao, 2023) — full taxonomy
4. Self-RAG (Asai, ICLR 2024) — adaptive/reflective retrieval

### Track B — Graph-based KG-QA (structured KGs)
1. UniKGQA (ICLR 2023) — unified retrieve + reason baseline
2. KAPING (ACL 2023) — simplest GraphRAG baseline (KG triples as prompt)
3. StructGPT (EMNLP 2023) — tool-augmented LLM for KG
4. ToG (ICLR 2024) — LLM walks KG iteratively
5. RoG (ICLR 2024) — plan-then-retrieve
6. GNN-RAG (2024) — GNN-seeded path retrieval
7. SubgraphRAG (ICLR 2025) — fast MLP retriever, strong baseline
8. RoE / RPO-RAG (2025–2026) — current SOTA frontier

### Track C — Corpus-level GraphRAG (ad-hoc KG from text)
1. Microsoft GraphRAG (2024) — community-based global QA
2. HippoRAG (NeurIPS 2024) — PPR-based associative retrieval
3. LightRAG (2024) — dynamic dual-level KG index
4. PropRAG (EMNLP 2025) — propagation without explicit KG

### Track D — For TraceRAG specifically
1. SubgraphRAG (ICLR 2025) — base architecture
2. GNN-RAG (2024) — structural retrieval comparison
3. Self-RAG (ICLR 2024) — LLM-cited supervision precedent
4. RoG (ICLR 2024) — relation path supervision comparison
5. RPO-RAG (2026) — reward-guided training (related direction)

---

## Open Research Questions (as of 2026)

1. **Retrieval supervision**: What is the right signal to train a KG retriever? Shortest paths (SubgraphRAG), LLM reasoning traces (TraceRAG), or answer-reward (RPO-RAG)?

2. **Relation-awareness in structural features**: DDE (SubgraphRAG) ignores relation types during propagation. How much does relation-conditioned propagation (RDDE) help?

3. **Small LLM + better retrieval vs. large LLM + simple retrieval**: SubgraphRAG shows retrieval quality is the bottleneck for small LLMs. Does improving retrieval close the gap with GPT-4?

4. **Generalization across KG domains**: Most methods train/eval on Freebase. How well do retrieval methods transfer to WikiMovies (MetaQA), medical KGs, or ad-hoc corpus KGs?

5. **Global vs. local queries**: MSFT GraphRAG addresses global corpus queries; structured KGQA methods address local entity-centric queries. No unified method handles both well.

6. **Interpretability of retrieval**: Path-based methods (RoG, ToG) are more interpretable. MLP-based methods (SubgraphRAG) are faster but opaque. Can we get both?

7. **Efficient indexing for corpus GraphRAG**: MSFT GraphRAG indexing is expensive ($10–50 per corpus). LazyGraphRAG and LightRAG reduce this but trade off retrieval quality.

---
