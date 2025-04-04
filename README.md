# 🚀 AI Researcher Ecosystem: Empowering Scientific Discovery with AI

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/yourusername/ai-researcher-ecosystem/pulls) 
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/github/last-commit/yourusername/ai-researcher-ecosystem?label=Updated&logo=github)](https://github.com/yourusername/ai-researcher-ecosystem/commits/main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)]()
[![Contribution Welcome](https://img.shields.io/badge/Contributions-welcome-blue)](CONTRIBUTING.md) <!-- Link to your contributing section/file -->

**Dive into the rapidly evolving world of AI-powered research automation!** 💡 This curated repository is your compass to navigate the cutting-edge tools, frameworks, and methodologies that are revolutionizing scientific discovery. We track the latest advancements, focusing on resources that empower researchers and accelerate breakthroughs.

<!-- omit in toc -->
## 👀 Introduction

AI is transforming the scientific landscape, offering powerful tools to assist researchers throughout the entire research lifecycle. This repository provides a structured overview of the AI researcher ecosystem, focusing on key components and advancements.

<p align="center">
  <img src="https://via.placeholder.com/800x400.png?text=AI+Research+Automation+Landscape" alt="AI Research Automation Landscape" width="800">
</p>

We categorize resources based on the core stages of research automation impacted by AI, drawing heavily from the surveyed literature.

<!-- omit in toc -->
## 📌 Table of Contents

- [⚙️ Core Components](#️-core-components)
    - [Research Automation Stack](#research-automation-stack)
    - [Emerging Capabilities](#emerging-capabilities)
- [🔥 Featured Projects](#-featured-projects)
    - [Open Source Tools](#open-source-tools)
    - [Commercial Platforms](#commercial-platforms)
- [📚 Knowledge Building](#-knowledge-building)
    - [Knowledge Acquisition](#knowledge-acquisition)
        - [Main Acquisition & Recommendation](#main-acquisition--recommendation)
        - [Figure Understanding](#figure-understanding)
        - [Table Understanding](#table-understanding)
    - [Knowledge Utilization](#knowledge-utilization)
- [💡 Idea Generation](#-idea-generation)
    - [Generation Strategy](#generation-strategy)
    - [Generation Method](#generation-method)
    - [Internal Evaluation (Hypothesis Validation & Fact Checking)](#internal-evaluation-hypothesis-validation--fact-checking)
- [🔬 Experiment Execution](#-experiment-execution)
    - [Experiment Design](#experiment-design)
    - [Experiment Conduction](#experiment-conduction)
    - [Internal Evaluation](#internal-evaluation-1)
- [📝 Paper Writing](#-paper-writing)
    - [Main Writing & Summarization](#main-writing--summarization)
    - [Figure Generation](#figure-generation)
    - [Table Generation](#table-generation)
    - [Slides & Poster Generation](#slides--poster-generation)
- [✅ Peer Review](#-peer-review)
- [🏆 Benchmarks](#-benchmarks)
- [🌐 Community](#-community)
    - [Upcoming Events](#upcoming-events)
    - [Discussion Forums](#discussion-forums)
- [👋 Contributing](#-contributing)
- [📜 License](#-license)
- [🎖️ Credits and Acknowledgements](#️-credits-and-acknowledgements)

# ⚙️ Core Components

<!-- omit in toc -->
## Research Automation Stack

The fundamental layers where AI is impacting the research lifecycle:

| Layer                 | Key Papers Category Alignment | Example Technologies / Concepts                                                                      | Description                                                                                                                                  |
| :-------------------- | :---------------------------- | :--------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **Knowledge Building** | Knowledge Building            | [Elicit](https://elicit.org), [Consensus](https://consensus.app), SciSpace, Semantic Scholar, KG Tools | AI tools for literature discovery, summarization, knowledge graph construction, understanding figures/tables, and synthesizing existing work. |
| **Idea Generation**   | Idea Generation               | Hypothesis generators (e.g., based on KGs, LLMs), Agent-based exploration (e.g., [ResearchAgent](https://github.com/JinheonBaek/ResearchAgent)) | AI assisting in formulating novel research questions, hypotheses, identifying gaps, and exploring new research directions.                 |
| **Experimentation**   | Experiment Execution          | Autonomous Labs (e.g., [CoScientist](https://github.com/gomesgroup/coscientist)), [AlphaFold](https://alphafold.ebi.ac.uk), AutoML, Agent-driven simulation | AI aiding in experimental design, automating experiment execution (physical or simulated), data collection, analysis, and interpretation. |
| **Paper Writing**     | Paper Writing                 | [Manubot](https://manubot.org), [Overleaf AI](https://www.overleaf.com), [PaperRobot](https://github.com/EagleW/PaperRobot), Citation/Figure generators | AI assistance in drafting sections (abstract, related work), generating figures/tables, citation management, editing, and formatting.     |
| **Peer Review**       | Peer Review                   | Reviewer assignment tools, Argument mining, Review generation aids (e.g., [ReviewRobot](https://github.com/EagleW/ReviewRobot))            | AI tools supporting reviewer assignment, review quality assessment, argument extraction, and potentially review generation/summarization.      |

<!-- omit in toc -->
## Emerging Capabilities

Exciting frontiers pushing the boundaries of AI in research:

-   **End-to-End Research Agents:** AI systems attempting to automate multiple stages of the research cycle autonomously (e.g., The AI Scientist, AutoSurvey).
-   **Human-AI Collaboration Patterns:** Novel workflows and interfaces integrating AI assistance seamlessly into researcher tasks for enhanced productivity and creativity.
-   **AI for Reproducibility & Validation:** Tools focused on verifying scientific claims, attempting automated replication of results, and ensuring overall scientific rigor.

# 🔥 Featured Projects

<!-- omit in toc -->
## Open Source Tools

| Project | Description | Stars | Link |
|---------|-------------|-------|------|
| [Galactica](https://github.com/paperswithcode/galai) | Meta's scientific language model for research tasks. | ![](https://img.shields.io/github/stars/paperswithcode/galai?style=social) | [GitHub](https://github.com/paperswithcode/galai) |
| [SciKit-LLM](https://github.com/iryna-kondr/scikit-llm) | Integration of LLMs into scientific workflows. | ![](https://img.shields.io/github/stars/iryna-kondr/scikit-llm?style=social) | [GitHub](https://github.com/iryna-kondr/scikit-llm) |
| [ResearchAgent](https://github.com/JinheonBaek/ResearchAgent) | Iterative research idea generation over scientific literature. | ![](https://img.shields.io/github/stars/JinheonBaek/ResearchAgent?style=social) | [GitHub](https://github.com/JinheonBaek/ResearchAgent) |
| [CoScientist](https://github.com/gomesgroup/coscientist) | Autonomous chemical research with LLMs. | ![](https://img.shields.io/github/stars/gomesgroup/coscientist?style=social) | [GitHub](https://github.com/gomesgroup/coscientist) |
| [PaperRobot](https://github.com/EagleW/PaperRobot) | Incremental Draft Generation of Scientific Ideas. | ![](https://img.shields.io/github/stars/EagleW/PaperRobot?style=social) | [GitHub](https://github.com/EagleW/PaperRobot) |

<!-- omit in toc -->
## Commercial Platforms

*   [Elicit](https://elicit.org/)
*   [Consensus](https://consensus.app/)
*   [SciSpace](https://scispace.com/)
*   [Wolfram Research Assistant](https://www.wolfram.com/research-assistant/)
*   [IBM Watson Discovery](https://www.ibm.com/cloud/watson-discovery)
*   [Overleaf AI](https://www.overleaf.com)

# 📚 Knowledge Building


<!-- omit in toc -->
## Knowledge Acquisition

<!-- omit in toc -->
### Main Acquisition & Recommendation
*   From Who You Know to What You Read: Augmenting Scientific Recommendations with Implicit Social Networks [[Paper]](https://doi.org/10.1145/3491102.3517470) ![](https://img.shields.io/badge/CHI-2022-blue) [[Review](https://openreview.net/forum?id=7vdeyZVPHL)]
*   Comlittee: Literature discovery with personal elected author committees [[Paper]](https://doi.org/10.1145/3544548.3581371) ![](https://img.shields.io/badge/CHI-2023-blue) [[Review](https://openreview.net/forum?id=NdSOavTTqdC)]
*   An academic recommender system on large citation data based on clustering, graph modeling and deep learning [[Paper]](https://doi.org/10.1007/s10115-024-02094-7) ![](https://img.shields.io/badge/Knowl._Inf._Syst.-2024-green)
*   Paperweaver: Enriching topical paper alerts by contextualizing recommended papers with user-collected papers [[Paper]](https://doi.org/10.1145/3613904.3642196) ![](https://img.shields.io/badge/CHI-2024-blue)

<!-- omit in toc -->
### Figure Understanding
*   A Diagram is Worth a Dozen Images [[Paper]](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_15) ![](https://img.shields.io/badge/ECCV-2016-blue)
*   FigureQA: An Annotated Figure Dataset for Visual Reasoning [[Paper]](https://arxiv.org/abs/1710.07300) ![](https://img.shields.io/badge/arXiv-2017.10-red?logo=arxiv) [[Code](https://github.com/Maluuba/FigureQA)]
*   A simple neural network module for relational reasoning [[Paper]](https://papers.nips.cc/paper_files/paper/2017/hash/e6acf4b0f69f6f6e60e9a815938aa1ff-Abstract.html) ![](https://img.shields.io/badge/NeurIPS-2017-blue) [[Review](https://papers.nips.cc/paper_files/paper/2017/file/e6acf4b0f69f6f6e60e9a815938aa1ff-Reviews.html)]
*   ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning [[Paper]](https://aclanthology.org/2022.findings-acl.177/) ![](https://img.shields.io/badge/ACL_Findings-2022-blue) [[Code](https://github.com/vis-nlp/chartqa)]
*   ChartSumm: A Comprehensive Benchmark for Automatic Chart Summarization of Long and Short Summaries [[Paper]](https://arxiv.org/abs/2304.13620) ![](https://img.shields.io/badge/arXiv-2023.04-red?logo=arxiv) [[Code](https://github.com/pranonrahman/ChartSumm)]
*   SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers [[Paper]](https://arxiv.org/pdf/2407.09413) ![](https://img.shields.io/badge/arXiv-2024.07-red?logo=arxiv) [[Code](https://github.com/google/spiqa)]
*   Multimodal ArXiv: A Dataset for Improving Scientific Comprehension of Large Vision-Language Models [[Paper]](https://aclanthology.org/2024.acl-long.775/) ![](https://img.shields.io/badge/ACL-2024-blue) [[Code](https://huggingface.co/MMInstruction/Qwen-VL-ArXivQA)] [[Project](https://mm-arxiv.github.io/)]
*   SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval [[Paper]](https://aclanthology.org/2024.findings-acl.746/) ![](https://img.shields.io/badge/ACL_Findings-2024-blue) [[Code](https://github.com/Wusiwei0410/SciMMIR)]
*   CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs [[Paper]](https://openreview.net/pdf?id=cy8mq7QYae) ![](https://img.shields.io/badge/NeurIPS-2024-blue) [[Code](https://github.com/princeton-nlp/CharXiv)]
*   ChartAdapter: Large Vision-Language Model for Chart Summarization [[Paper]](https://arxiv.org/abs/2412.20715) ![](https://img.shields.io/badge/arXiv-2024.12-red?logo=arxiv)

<!-- omit in toc -->
### Table Understanding
*   ToTTo: A Controlled Table-To-Text Generation Dataset [[Paper]](https://aclanthology.org/2020.emnlp-main.89/) ![](https://img.shields.io/badge/EMNLP-2020-blue) [[Code](https://github.com/google-research-datasets/ToTTo)]
*   Towards Table-to-Text Generation with Numerical Reasoning [[Paper]](https://aclanthology.org/2021.acl-long.115/) ![](https://img.shields.io/badge/ACL-2021-blue) [[Code](https://github.com/titech-nlp/numeric-nlg)]
*   Structure-Aware Pre-Training for Table-to-Text Generation [[Paper]](https://aclanthology.org/2021.findings-acl.200/) ![](https://img.shields.io/badge/ACL_Findings-2021-blue) [[Code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/unid2t)] <!-- Assuming this is the intended code link -->
*   SciGen: a Dataset for Reasoning-Aware Text Generation from Scientific Tables [[Paper]](https://openreview.net/forum?id=Jul-uX7EV_I) ![](https://img.shields.io/badge/EMNLP_Findings-2021-blue) [[Code](https://github.com/UKPLab/SciGen)]
*   SciXGen: A Scientific Paper Dataset for Context-Aware Text Generation [[Paper]](https://aclanthology.org/2021.findings-emnlp.128/) ![](https://img.shields.io/badge/EMNLP_Findings-2021-blue)
*   Robust (Controlled) Table-to-Text Generation with Structure-Aware Equivariance Learning [[Paper]](https://aclanthology.org/2022.naacl-main.371/) ![](https://img.shields.io/badge/NAACL-2022-blue) [[Code](https://github.com/luka-group/Lattice)]
*   Few-shot Table-to-text Generation with Prefix-Controlled Generator [[Paper]](https://aclanthology.org/2022.coling-1.565/) ![](https://img.shields.io/badge/COLING-2022-blue)
*   Table-To-Text generation and pre-training with TabT5 [[Paper]](https://aclanthology.org/2022.findings-emnlp.503/) ![](https://img.shields.io/badge/EMNLP_Findings-2022-blue)
*   LoFT: Enhancing Faithfulness and Diversity for Table-to-Text Generation via Logic Form Control [[Paper]](https://aclanthology.org/2023.eacl-main.40/) ![](https://img.shields.io/badge/EACL-2023-blue) [[Code](https://github.com/Yale-LILY/LoFT)]
*   SORTIE: Dependency-Aware Symbolic Reasoning for Logical Data-to-text Generation [[Paper]](https://aclanthology.org/2023.findings-acl.715/) ![](https://img.shields.io/badge/ACL_Findings-2023-blue) [[Code](https://github.com/wenhuchen/LogicNLG)]
*   Arithmetic-Based Pretraining Improving Numeracy of Pretrained Language Models [[Paper]](https://aclanthology.org/2023.starsem-1.42/) ![](https://img.shields.io/badge/STARSEM-2023-blue) [[Code](https://github.com/UKPLab/starsem2023-arithmetic-based-pretraining)]
*   Structure-aware Table-to-Text Generation with Prefix-tuning [[Paper]](https://dlnext.acm.org/doi/10.1145/3622896.3622919) ![](https://img.shields.io/badge/CIKM-2023-blue)
*   Unifying Structured Data as Graph for Data-to-Text Pre-Training [[Paper]](https://aclanthology.org/2024.tacl-1.12/) ![](https://img.shields.io/badge/TACL-2024-blue) [[Code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/unid2t)]
*   Table-to-Text Using Pre-trained Large Language Model and LoRA [[Paper]](https://ieeexplore.ieee.org/document/10707812) ![](https://img.shields.io/badge/IEEE_Access-2024-green)
*   Integrating Table Representations into Large Language Models for Improved Scholarly Document Comprehension [[Paper]](https://aclanthology.org/2024.sdp-1.28/) ![](https://img.shields.io/badge/SDP@ACL-2024-blue) [[Code](https://github.com/buseskorkmaz/Integrating-Table-Representations-into-LLMs)]

<!-- omit in toc -->
## Knowledge Utilization
*   Multi-document scientific summarization from a knowledge graph-centric view [[Paper]](https://aclanthology.org/2022.coling-1.543) ![](https://img.shields.io/badge/COLING-2022-blue) [[Code](https://github.com/muguruzawang/kgsum)]
*   Bio-sieve: exploring instruction tuning large language models for systematic review automation [[Paper]](https://doi.org/10.48550/arXiv.2308.06610) ![](https://img.shields.io/badge/arXiv-2023.08-red?logo=arxiv) [[Code](https://github.com/ambroser53/Bio-SIEVE)]
*   Hierarchical catalogue generation for literature review: a benchmark [[Paper]](https://doi.org/10.18653/v1/2023.findings-emnlp.453) ![](https://img.shields.io/badge/EMNLP_Findings-2023-blue)
*   Litllm: A toolkit for scientific literature review [[Paper]](https://doi.org/10.48550/arXiv.2402.01788) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv) [[Code](https://github.com/LitLLM/litllms-for-literature-review-tmlr)]
*   Automating research synthesis with domain-specific large language model fine-tuning [[Paper]](https://arxiv.org/abs/2404.08680) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv) [[Code](https://github.com/peterjhwang/slr-helper)]
*   Assisting in writing wikipedia-like articles from scratch with large language models [[Paper]](https://doi.org/10.18653/v1/2024.naacl-long.347) ![](https://img.shields.io/badge/NAACL-2024-blue) [[Code](https://github.com/stanford-oval/storm)]
*   Chime: Llm-assisted hierarchical organization of scientific studies for literature review support [[Paper]](https://doi.org/10.18653/v1/2024.findings-acl.8) ![](https://img.shields.io/badge/ACL_Findings-2024-blue) [[Code](https://github.com/allenai/chime)]
*   Instruct Large Language Models to Generate Scientific Literature Survey Step by Step [[Paper]](https://arxiv.org/pdf/2408.07884) ![](https://img.shields.io/badge/NLPCC-2024-blue)
*   Language agents achieve superhuman synthesis of scientific knowledge [[Paper]](https://doi.org/10.48550/arXiv.2409.13740) ![](https://img.shields.io/badge/arXiv-2024.09-red?logo=arxiv)
*   Hireview: Hierarchical taxonomy-driven automatic literature review generation [[Paper]](https://doi.org/10.48550/arXiv.2410.03761) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv)
*   Knowledge Navigator: LLM-guided Browsing Framework for Exploratory Search in Scientific Literature [[Paper]](https://aclanthology.org/2024.findings-emnlp.516) ![](https://img.shields.io/badge/EMNLP_Findings-2024-blue) [[Code](https://github.com/katzurik/Knowledge_Navigator)]
*   Autosurvey: Large language models can automatically write surveys [[Paper]](http://papers.nips.cc/paper_files/paper/2024/hash/d07a9fc7da2e2ec0574c38d5f504d105-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2024-blue)
*   SurveyX: Academic Survey Automation via Large Language Models [[Paper]](https://arxiv.org/abs/2502.14776) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv)
*   LimTopic: LLM-based Topic Modeling and Text Summarization for Analyzing Scientific Articles limitations [[Paper]](https://www.arxiv.org/abs/2503.10658) ![](https://img.shields.io/badge/JCDL-2024-blue) [[Code](https://github.com/IbrahimAlAzhar/LimTopic/tree/master)]
*   What's In Your Field? Mapping Scientific Research with Knowledge Graphs and Large Language Models [[Paper]](https://arxiv.org/abs/2503.09894) ![](https://img.shields.io/badge/arXiv-2025.03-red?logo=arxiv) [[Code](https://github.com/chiral-carbon/kg-for-science)]

# 💡 Idea Generation


<!-- omit in toc -->
## Generation Strategy
*   Literature based discovery: models, methods, and trends [[Paper]](https://doi.org/10.1016/j.jbi.2017.08.011) ![](https://img.shields.io/badge/J._Biomed._Inform.-2017-green)
*   Predicting the Future of AI with AI: High-quality link prediction in an exponentially growing knowledge network [[Paper]](https://doi.org/10.48550/arXiv.2210.00881) ![](https://img.shields.io/badge/arXiv-2022.10-red?logo=arxiv) [[Code](https://github.com/artificial-scientist-lab/FutureOfAIviaAI)]
*   LLM and Simulation as Bilevel Optimizers: A New Paradigm to Advance Physical Scientific Discovery [[Paper]](https://openreview.net/forum?id=hz8cFsdz7P) ![](https://img.shields.io/badge/ICML-2024-blue)
*   Chain of ideas: Revolutionizing research via novel idea development with llm agents [[Paper]](https://doi.org/10.48550/arXiv.2410.13185) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) [[Code](https://github.com/DAMO-NLP-SG/CoI-Agent)]
*   Nova: An iterative planning and search approach to enhance novelty and diversity of llm generated ideas [[Paper]](https://doi.org/10.48550/arXiv.2410.14255) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv)

<!-- omit in toc -->
## Generation Method
*   Ideas are dimes a dozen: Large language models for idea generation in innovation [[Paper]](https://christophegirard.com/wp-content/uploads/2023/09/Etude-creation-idees-comparative-ChatGPT-vs-etudiants.pdf) ![](https://img.shields.io/badge/Wharton_School_Paper-2023-brown)
*   Monte Carlo Thought Search: Large Language Model Querying for Complex Scientific Reasoning in Catalyst Design [[Paper]](https://arxiv.org/abs/2310.14420) ![](https://img.shields.io/badge/arXiv-2023.10-red?logo=arxiv) [[Code](https://github.com/pnnl/chemreasoner)]
*   Large Language Models are Zero Shot Hypothesis Proposers [[Paper]](https://arxiv.org/abs/2311.05965) ![](https://img.shields.io/badge/arXiv-2023.11-red?logo=arxiv)
*   Mathematical discoveries from program search with large language models [[Paper]](https://www.nature.com/articles/s41586-023-06924-6) ![](https://img.shields.io/badge/Nature-2023-green) [[Code](https://github.com/google-deepmind/funsearch)]
*   ChemReasoner: Heuristic Search over a Large Language Model's Knowledge Space using Quantum-Chemical Feedback [[Paper]](https://arxiv.org/abs/2402.10980) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv) [[Code](https://github.com/pnnl/chemreasoner)]
*   Acceleron: A tool to accelerate research ideation [[Paper]](https://doi.org/10.48550/arXiv.2403.04382) ![](https://img.shields.io/badge/arXiv-2024.03-red?logo=arxiv)
*   Hypothesis generation with large language models [[Paper]](https://doi.org/10.48550/arXiv.2404.04326) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv) [[Code](https://github.com/ChicagoHAI/hypothesis-generation)]
*   Researchagent: Iterative research idea generation over scientific literature with large language models [[Paper]](https://doi.org/10.48550/arXiv.2404.07738) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv) [[Code](https://github.com/JinheonBaek/ResearchAgent)]
*   Data-driven discovery with large generative models [[Paper]](https://openreview.net/forum?id=5SpjhZNXtt) ![](https://img.shields.io/badge/ICML-2024-blue)
*   Generation and human-expert evaluation of interesting research ideas using knowledge graphs and large language models [[Paper]](https://doi.org/10.48550/arXiv.2405.17044) ![](https://img.shields.io/badge/arXiv-2024.05-red?logo=arxiv) [[Code](https://github.com/artificial-scientist-lab/SciMuse)]
*   Large Language Models as Biomedical Hypothesis Generators: A Comprehensive Evaluation [[Paper]](https://openreview.net/forum?id=q36rpGlG9X#discussion) ![](https://img.shields.io/badge/ICML_Workshop-2024-blue) [[Code](https://github.com/TsinghuaC3I/LLM4BioHypoGen/)]
*   The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery [[Paper]](https://doi.org/10.48550/arXiv.2408.06292) ![](https://img.shields.io/badge/arXiv-2024.08-red?logo=arxiv) [[Code](https://github.com/SakanaAI/AI-Scientist)]
*   SciMON: Scientific Inspiration Machines Optimized for Novelty [[Paper]](https://doi.org/10.18653/v1/2024.acl-long.18) ![](https://img.shields.io/badge/ACL-2024-blue) [[Code](https://github.com/EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty)]
*   Large language models for automated open-domain scientific hypotheses discovery [[Paper]](https://doi.org/10.18653/v1/2024.findings-acl.804) ![](https://img.shields.io/badge/ACL_Findings-2024-blue) [[Code](https://github.com/SenticNet/MOOSE)]
*   Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers [[Paper]](https://doi.org/10.48550/arXiv.2409.04109) ![](https://img.shields.io/badge/arXiv-2024.09-red?logo=arxiv) [[Code](https://github.com/NoviScl/AI-Researcher)]
*   Accelerating scientific discovery with generative knowledge extraction, graph-based representation, and multimodal intelligent graph reasoning [[Paper]](https://doi.org/10.1088/2632-2153/ad7228) ![](https://img.shields.io/badge/Mach._Learn.:_Sci._Technol.-2024-green) [[Code](https://github.com/lamm-mit/GraphReasoning)]
*   Can Large Language Models Unlock Novel Scientific Research Ideas? [[Paper]](https://doi.org/10.48550/arXiv.2409.06185) ![](https://img.shields.io/badge/arXiv-2024.09-red?logo=arxiv) [[Code](https://github.com/sandeep82945/Future-Idea-Generation)]
*   Sciagents: Automating scientific discovery through multi-agent intelligent graph reasoning [[Paper]](https://doi.org/10.48550/arXiv.2409.05556) ![](https://img.shields.io/badge/arXiv-2024.09-red?logo=arxiv) [[Code](https://github.com/lamm-mit/SciAgentsDiscovery)]
*   Scideator: Human-LLM Scientific Idea Generation Grounded in Research-Paper Facet Recombination [[Paper]](https://arxiv.org/abs/2409.14634) ![](https://img.shields.io/badge/arXiv-2024.09-red?logo=arxiv)
*   IdeaSynth: Iterative Research Idea Development Through Evolving and Composing Idea Facets with Literature-Grounded Feedback [[Paper]](https://doi.org/10.48550/arXiv.2410.04025) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv)
*   MOOSE-Chem: Large Language Models for Rediscovering Unseen Chemistry Scientific Hypotheses [[Paper]](https://doi.org/10.48550/arXiv.2410.07076) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) [[Code](https://github.com/ZonglinY/MOOSE-Chem)]
*   Two heads are better than one: A multi-agent system has the potential to improve scientific idea generation [[Paper]](https://doi.org/10.48550/arXiv.2410.09403) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) [[Code](https://github.com/open-sciencelab/Virtual-Scientists)]
*   Literature meets data: A synergistic approach to hypothesis generation [[Paper]](https://doi.org/10.48550/arXiv.2410.17309) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) [[Code](https://github.com/ChicagoHAI/hypothesis-generation)]
*   SciPIP: An LLM-based Scientific Paper Idea Proposer [[Paper]](https://doi.org/10.48550/arXiv.2410.23166) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv)
*   Aigs: Generating Science from Ai-Powered Automated Falsification [[Paper]](https://arxiv.org/pdf/2411.11910) ![](https://img.shields.io/badge/arXiv-2024.11-red?logo=arxiv)
*   Exploring Scientific Hypothesis Generation with Mamba [[Paper]](https://aclanthology.org/2024.nlp4science-1.17/) ![](https://img.shields.io/badge/NLP4Science@EMNLP-2024-blue) [[Code](https://github.com/fglx-c/Exploring-Scientific-Hypothesis-Generation-with-Mamba)]
*   Improving Scientific Hypothesis Generation with Knowledge Grounded Large Language Models [[Paper]](https://arxiv.org/abs/2411.02382) ![](https://img.shields.io/badge/arXiv-2024.11-red?logo=arxiv) [[Code](https://anonymous.4open.science/r/KG-CoI-C203/README.md)]
*   DiscoveryWorld: A Virtual Environment for Developing and Evaluating Automated Scientific Discovery Agents [[Paper]](http://papers.nips.cc/paper_files/paper/2024/hash/13836f251823945316ae067350a5c366-Abstract-Datasets_and_Benchmarks_Track.html) ![](https://img.shields.io/badge/NeurIPS_D&B-2024-blue) [[Code](https://github.com/allenai/discoveryworld)]
*   Learning to Generate Research Idea with Dynamic Control [[Paper]](https://doi.org/10.48550/arXiv.2412.14626) ![](https://img.shields.io/badge/arXiv-2024.12-red?logo=arxiv) [[Code](https://github.com/du-nlp-lab/Learn2Gen)]
*   ResearchTown: Simulator of Human Research Community [[Paper]](https://doi.org/10.48550/arXiv.2412.17767) ![](https://img.shields.io/badge/arXiv-2024.12-red?logo=arxiv) [[Code](https://github.com/ulab-uiuc/research-town)]
*   Dolphin: Closed-loop Open-ended Auto-research through Thinking, Practice, and Feedback [[Paper]](https://doi.org/10.48550/arXiv.2501.03916) ![](https://img.shields.io/badge/arXiv-2025.01-red?logo=arxiv) [[Code](https://github.com/Alpha-Innovator/Dolphin)]
*   Towards an AI co-scientist [[Paper]](https://arxiv.org/pdf/2502.18864) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv)
*   Graph of AI Ideas: Leveraging Knowledge Graphs and LLMs for AI Research Idea Generation [[Paper]](https://arxiv.org/abs/2503.08549) ![](https://img.shields.io/badge/arXiv-2025.03-red?logo=arxiv)

<!-- omit in toc -->
## Internal Evaluation (Hypothesis Validation & Fact Checking)
*   Generating fact checking explanations [[Paper]](https://doi.org/10.18653/v1/2020.acl-main.656) ![](https://img.shields.io/badge/ACL-2020-blue)
*   Generative language modeling for automated theorem proving [[Paper]](https://arxiv.org/abs/2009.03393) ![](https://img.shields.io/badge/arXiv-2020.09-red?logo=arxiv)
*   Fact or fiction: Verifying scientific claims [[Paper]](https://doi.org/10.18653/v1/2020.emnlp-main.609) ![](https://img.shields.io/badge/EMNLP-2020-blue) [[Code](https://github.com/allenai/scifact)]
*   MultiVerS: Improving scientific claim verification with weak supervision and full-document context [[Paper]](https://doi.org/10.18653/v1/2022.findings-naacl.6) ![](https://img.shields.io/badge/NAACL_Findings-2022-blue) [[Code](https://github.com/dwadden/longchecker)]
*   Proofver: Natural logic theorem proving for fact verification [[Paper]](https://doi.org/10.1162/tacl_a_00503) ![](https://img.shields.io/badge/TACL-2022-blue) [[Code](https://github.com/krishnamrith12/ProoFVer)]
*   HyperTree Proof Search for Neural Theorem Proving [[Paper]](http://papers.nips.cc/paper_files/paper/2022/hash/a8901c5e85fb8e1823bbf0f755053672-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2022-blue)
*   Thor: Wielding hammers to integrate language models and automated theorem provers [[Paper]](http://papers.nips.cc/paper_files/paper/2022/hash/377c25312668e48f2e531e2f2c422483-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2022-blue)
*   Missing counter-evidence renders NLP fact-checking unrealistic for misinformation [[Paper]](https://doi.org/10.18653/v1/2022.emnlp-main.397) ![](https://img.shields.io/badge/EMNLP-2022-blue) [[Code](https://github.com/UKPLab/emnlp2022-missing-counter-evidence)]
*   The state of human-centered NLP technology for fact-checking [[Paper]](https://arxiv.org/pdf/2301.03056) ![](https://img.shields.io/badge/Inf._Process._Manag.-2023-green)
*   Draft, sketch, and prove: Guiding formal theorem provers with informal proofs [[Paper]](https://openreview.net/forum?id=SMa9EAovKMC) ![](https://img.shields.io/badge/ICLR-2023-blue) [[Code](https://github.com/albertqjiang/draft_sketch_prove)]
*   Decomposing the enigma: Subgoal-based demonstration learning for formal theorem proving [[Paper]](https://doi.org/10.48550/arXiv.2305.16366) ![](https://img.shields.io/badge/arXiv-2023.05-red?logo=arxiv) [[Code](https://github.com/HKUNLP/subgoal-theorem-prover)]
*   aedFaCT: Scientific Fact-Checking Made Easier via Semi-Automatic Discovery of Relevant Expert Opinions [[Paper]](https://doi.org/10.48550/arXiv.2305.07796) ![](https://img.shields.io/badge/arXiv-2023.05-red?logo=arxiv) [[Code](https://github.com/altuncu/aedFaCT)]
*   Fact-checking complex claims with program-guided reasoning [[Paper]](https://doi.org/10.18653/v1/2023.acl-long.386) ![](https://img.shields.io/badge/ACL-2023-blue) [[Code](https://github.com/mbzuai-nlp/ProgramFC)]
*   FactKG: Fact verification via reasoning on knowledge graphs [[Paper]](https://doi.org/10.18653/v1/2023.acl-long.895) ![](https://img.shields.io/badge/ACL-2023-blue) [[Code](https://github.com/jiho283/FactKG)]
*   Prompt to be Consistent is Better than Self-Consistent? Few-Shot and Zero-Shot Fact Verification with Pre-trained Language Models [[Paper]](https://doi.org/10.18653/v1/2023.findings-acl.278) ![](https://img.shields.io/badge/ACL_Findings-2023-blue) [[Code](https://github.com/znhy1024/ProToCo)]
*   Dt-solver: Automated theorem proving with dynamic-tree sampling guided by proof-level value function [[Paper]](https://doi.org/10.18653/v1/2023.acl-long.706) ![](https://img.shields.io/badge/ACL-2023-blue)
*   An in-context learning agent for formal theorem-proving [[Paper]](https://doi.org/10.48550/arXiv.2310.04353) ![](https://img.shields.io/badge/arXiv-2023.10-red?logo=arxiv) [[Code](https://github.com/trishullab/copra)]
*   Baldur: Whole-proof generation and repair with large language models [[Paper]](https://doi.org/10.1145/3611643.3616243) ![](https://img.shields.io/badge/ESEC/FSE-2023-blue)
*   Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method [[Paper]](https://doi.org/10.18653/v1/2023.ijcnlp-main.64) ![](https://img.shields.io/badge/IJCNLP-2023-blue) [[Code](https://github.com/jadeCurl/HiSS)]
*   Investigating zero-and few-shot generalization in fact verification [[Paper]](https://doi.org/10.18653/v1/2023.ijcnlp-main.34) ![](https://img.shields.io/badge/IJCNLP-2023-blue) [[Code](https://github.com/teacherpeterpan/Fact-Checking-Generalization)]
*   Characterizing and Verifying Scientific Claims: Qualitative Causal Structure is All You Need [[Paper]](https://doi.org/10.18653/v1/2023.emnlp-main.828) ![](https://img.shields.io/badge/EMNLP-2023-blue) [[Code](https://github.com/VulnDetector/VerQCS)]
*   Mustard: Mastering uniform synthesis of theorem and proof data [[Paper]](https://openreview.net/forum?id=8xliOUg9EW) ![](https://img.shields.io/badge/ICLR-2024-blue) [[Code](https://github.com/Eleanor-H/MUSTARD)]
*   Unsupervised Pretraining for Fact Verification by Language Model Distillation [[Paper]](https://openreview.net/forum?id=1mjsP8RYAw) ![](https://img.shields.io/badge/ICLR-2024-blue) [[Code](https://github.com/AdrianBZG/SFAVEL)]
*   Lego-prover: Neural theorem proving with growing libraries [[Paper]](https://openreview.net/forum?id=3f5PALef5B) ![](https://img.shields.io/badge/ICLR-2024-blue) [[Code](https://github.com/wiio12/LEGO-Prover)]
*   Can Large Language Models Detect Misinformation in Scientific News Reporting? [[Paper]](https://doi.org/10.48550/arXiv.2402.14268) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv)
*   What Makes Medical Claims (Un)Verifiable? Analyzing Entity and Relation Properties for Fact Verification [[Paper]](https://aclanthology.org/2024.eacl-long.124) ![](https://img.shields.io/badge/EACL-2024-blue)
*   Comparing knowledge sources for open-domain scientific claim verification [[Paper]](https://aclanthology.org/2024.eacl-long.128) ![](https://img.shields.io/badge/EACL-2024-blue) [[Code](https://www.github.com/jvladika/comparing-knowledge-sources)]
*   Towards large language models as copilots for theorem proving in lean [[Paper]](https://doi.org/10.48550/arXiv.2404.12534) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv) [[Code](https://github.com/lean-dojo/LeanCopilot)]
*   Deepseek-prover: Advancing theorem proving in llms through large-scale synthetic data [[Paper]](https://doi.org/10.48550/arXiv.2405.14333) ![](https://img.shields.io/badge/arXiv-2024.05-red?logo=arxiv)
*   MAGIC: Multi-Argument Generation with Self-Refinement for Domain Generalization in Automatic Fact-Checking [[Paper]](https://aclanthology.org/2024.lrec-main.951) ![](https://img.shields.io/badge/LREC--COLING-2024-blue) [[Code](https://github.com/NYCU-NLP-Lab/XClaimCheck)]
*   Improving health question answering with reliable and time-aware evidence retrieval [[Paper]](https://doi.org/10.18653/v1/2024.findings-naacl.295) ![](https://img.shields.io/badge/NAACL_Findings-2024-blue) [[Code](https://github.com/jvladika/Improving-Health-QA)]
*   Lean-star: Learning to interleave thinking and proving [[Paper]](https://doi.org/10.48550/arXiv.2407.10040) ![](https://img.shields.io/badge/arXiv-2024.07-red?logo=arxiv) [[Code](https://github.com/Lagooon/LeanSTaR)]
*   Grounding fallacies misrepresenting scientific publications in evidence [[Paper]](https://doi.org/10.48550/arXiv.2408.12812) ![](https://img.shields.io/badge/arXiv-2024.08-red?logo=arxiv) [[Code](https://github.com/UKPLab/naacl2025-missciplus)]
*   Understanding Fine-grained Distortions in Reports of Scientific Findings [[Paper]](https://doi.org/10.18653/v1/2024.findings-acl.369) ![](https://img.shields.io/badge/ACL_Findings-2024-blue) [[Project](https://www.uni-bamberg.de/en/nlproc/resources/sciencecommdistortion/)]
*   Enhancing natural language inference performance with knowledge graph for COVID-19 automated fact-checking in Indonesian language [[Paper]](https://doi.org/10.48550/arXiv.2409.00061) ![](https://img.shields.io/badge/arXiv-2024.09-red?logo=arxiv)
*   Augmenting the Veracity and Explanations of Complex Fact Checking via Iterative Self-Revision with LLMs [[Paper]](https://doi.org/10.48550/arXiv.2410.15135) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv)
*   ClaimVer: Explainable claim-level verification and evidence attribution of text through knowledge graphs [[Paper]](https://aclanthology.org/2024.findings-emnlp.795) ![](https://img.shields.io/badge/EMNLP_Findings-2024-blue)
*   Proving theorems recursively [[Paper]](http://papers.nips.cc/paper_files/paper/2024/hash/9de7a49945898da86e062e7029baa284-Abstract-Conference.html) ![](https://img.shields.io/badge/NeurIPS-2024-blue)
*   Automating Mathematical Proof Generation Using Large Language Model Agents and Knowledge Graphs [[Paper]](https://arxiv.org/abs/2503.11657) ![](https://img.shields.io/badge/arXiv-2025.03-red?logo=arxiv)
*   Optimizing Decomposition for Optimal Claim Verification [[Paper]](https://arxiv.org/abs/2503.15354) ![](https://img.shields.io/badge/arXiv-2025.03-red?logo=arxiv) [[Code](https://github.com/yining610/dynamic-decomposition)]

# 🔬 Experiment Execution


<!-- omit in toc -->
## Experiment Design
*   HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face [[Paper]](https://arxiv.org/abs/2303.17580) ![](https://img.shields.io/badge/arXiv-2023.03-red?logo=arxiv) [[Code](https://github.com/microsoft/JARVIS)]
*   AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation [[Paper]](https://arxiv.org/abs/2308.08155) ![](https://img.shields.io/badge/arXiv-2023.08-red?logo=arxiv) [[Code](https://github.com/microsoft/autogen)]
*   Autonomous chemical research with large language models [[Paper]](https://www.nature.com/articles/s41586-023-06792-0) ![](https://img.shields.io/badge/Nature-2023-green) [[Code](https://github.com/gomesgroup/coscientist)]
*   Navigating Complexity: Orchestrated Problem Solving with Multi-Agent LLMs [[Paper]](https://arxiv.org/abs/2402.16713) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv)
*   CRISPR-GPT: An LLM Agent for Automated Design of Gene-Editing Experiments [[Paper]](https://arxiv.org/abs/2404.18021) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv)
*   Simulating Expert Discussions with Multi-agent for Enhanced Scientific Problem Solving [[Paper]](https://aclanthology.org/2024.sdp-1.23/) ![](https://img.shields.io/badge/SDP@ACL-2024-blue)
*   An automatic end-to-end chemical synthesis development platform powered by large language models [[Paper]](https://www.nature.com/articles/s41467-024-54457-x) ![](https://img.shields.io/badge/Nat._Commun.-2024-green) [[Code](https://github.com/Ruan-Yixiang/LLM-RDF)]

<!-- omit in toc -->
## Experiment Conduction
*   Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences [[Paper]](https://www.pnas.org/doi/full/10.1073/pnas.2016239118) ![](https://img.shields.io/badge/PNAS-2020-green) [[Code](https://github.com/facebookresearch/esm)]
*   Controllable Protein Design with Language Models [[Paper]](https://arxiv.org/abs/2201.07338) ![](https://img.shields.io/badge/arXiv-2022.01-red?logo=arxiv)
*   Evolutionary-scale prediction of atomic-level protein structure with a language model [[Paper]](https://www.science.org/doi/10.1126/science.ade2574) ![](https://img.shields.io/badge/Science-2023-green)
*   Bayesian Optimization of Catalysts With In-context Learning [[Paper]](https://arxiv.org/abs/2304.05341) ![](https://img.shields.io/badge/arXiv-2023.04-red?logo=arxiv) [[Code](https://github.com/ur-whitelab/BO-LIFT)]
*   Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering [[Paper]](https://arxiv.org/abs/2305.03403) ![](https://img.shields.io/badge/arXiv-2023.05-red?logo=arxiv) [[Code](https://github.com/noahho/CAAFE)]
*   ChatGPT-powered Conversational Drug Editing Using Retrieval and Domain Feedback [[Paper]](https://arxiv.org/abs/2305.18090) ![](https://img.shields.io/badge/arXiv-2023.05-red?logo=arxiv) [[Code](https://github.com/chao1224/ChatDrug)]
*   Training Socially Aligned Language Models on Simulated Social Interactions [[Paper]](https://arxiv.org/abs/2305.16960) ![](https://img.shields.io/badge/arXiv-2023.05-red?logo=arxiv)
*   Are you in a Masquerade? Exploring the Behavior and Impact of Large Language Model Driven Social Bots in Online Social Networks [[Paper]](https://arxiv.org/abs/2307.10337) ![](https://img.shields.io/badge/arXiv-2023.07-red?logo=arxiv) [[Code](https://github.com/Litsay/Masquerade-23)]
*   AutoML-GPT: Large Language Model for AutoML [[Paper]](https://arxiv.org/abs/2309.01125) ![](https://img.shields.io/badge/arXiv-2023.09-red?logo=arxiv)
*   Data-Juicer: A One-Stop Data Processing System for Large Language Models [[Paper]](https://arxiv.org/abs/2309.02033) ![](https://img.shields.io/badge/arXiv-2023.09-red?logo=arxiv) [[Code](https://github.com/modelscope/data-juicer)]
*   Monte Carlo Thought Search: Large Language Model Querying for Complex Scientific Reasoning in Catalyst Design [[Paper]](https://arxiv.org/abs/2310.14420) ![](https://img.shields.io/badge/arXiv-2023.10-red?logo=arxiv) [[Code](https://github.com/pnnl/chemreasoner)]
*   Jellyfish: A Large Language Model for Data Preprocessing [[Paper]](https://arxiv.org/abs/2312.01678) ![](https://img.shields.io/badge/arXiv-2023.12-red?logo=arxiv) [[Code](https://huggingface.co/NECOUDBFM/Jellyfish-8B)]
*   Autonomous chemical research with large language models [[Paper]](https://www.nature.com/articles/s41586-023-06792-0) ![](https://img.shields.io/badge/Nature-2023-green) [[Code](https://github.com/gomesgroup/coscientist)]
*   DrugAssist: A Large Language Model for Molecule Optimization [[Paper]](https://arxiv.org/abs/2401.10334) ![](https://img.shields.io/badge/arXiv-2024.01-red?logo=arxiv) [[Code](https://github.com/blazerye/DrugAssist)]
*   ChemReasoner: Heuristic Search over a Large Language Model's Knowledge Space using Quantum-Chemical Feedback [[Paper]](https://arxiv.org/abs/2402.10980) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv) [[Code](https://github.com/pnnl/chemreasoner)]
*   Augmenting large language models with chemistry tools [[Paper]](https://www.nature.com/articles/s42256-024-00832-8) ![](https://img.shields.io/badge/Nat._Mach._Intell.-2024-green) [[Code](https://github.com/ur-whitelab/chemcrow-public)]
*   Efficient Evolutionary Search Over Chemical Space with Large Language Models [[Paper]](https://arxiv.org/abs/2406.16976) ![](https://img.shields.io/badge/arXiv-2024.06-red?logo=arxiv) [[Code](https://github.com/zoom-wang112358/MOLLEO)]
*   Tree Search for Language Model Agents [[Paper]](https://arxiv.org/abs/2407.01476) ![](https://img.shields.io/badge/arXiv-2024.07-red?logo=arxiv) [[Code](https://github.com/kohjingyu/search-agents)]
*   De novo generation of SARS-CoV-2 antibody CDRH3 with a pre-trained generative large language model [[Paper]](https://www.nature.com/articles/s41467-024-50903-y) ![](https://img.shields.io/badge/Nat._Commun.-2024-green) [[Code](https://github.com/TencentAILabHealthcare/PALM)]
*   DrugAgent: Automating AI-aided Drug Discovery Programming through LLM Multi-Agent Collaboration [[Paper]](https://arxiv.org/abs/2411.15692) ![](https://img.shields.io/badge/arXiv-2024.11-red?logo=arxiv) [[Code](https://anonymous.4open.science/r/drugagent-5C42/README.md)]
*   PaperBench: Evaluating AI’s Ability to Replicate AI Research [[Paper]](https://cdn.openai.com/papers/22265bac-3191-44e5-b057-7aaacd8e90cd/paperbench.pdf) ![](https://img.shields.io/badge/OpenAI_Report-2025-brown) [[Code](https://github.com/openai/preparedness)]

<!-- omit in toc -->
## Internal Evaluation
*   Automl-gpt: Automatic machine learning with gpt [[Paper]](https://doi.org/10.48550/arXiv.2305.02499) ![](https://img.shields.io/badge/arXiv-2023.05-red?logo=arxiv) [[Code](https://github.com/Significant-Gravitas/AutoGPT)]
*   AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation [[Paper]](https://arxiv.org/abs/2308.08155) ![](https://img.shields.io/badge/arXiv-2023.08-red?logo=arxiv) [[Code](https://github.com/microsoft/autogen)]
*   MentaLLaMA: Interpretable Mental Health Analysis on Social Media with Large Language Models [[Paper]](https://arxiv.org/abs/2309.13567) ![](https://img.shields.io/badge/arXiv-2023.09-red?logo=arxiv) [[Code](https://github.com/SteveKGYang/MentaLLaMA)]
*   MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation [[Paper]](https://arxiv.org/abs/2310.03302) ![](https://img.shields.io/badge/arXiv-2023.10-red?logo=arxiv) [[Code](https://github.com/snap-stanford/MLAgentBench/)]
*   MechAgents: Large language model multi-agent collaborations can solve mechanics problems, generate new data, and integrate knowledge [[Paper]](https://doi.org/10.48550/arXiv.2311.08166) ![](https://img.shields.io/badge/arXiv-2023.11-red?logo=arxiv) [[Code](https://github.com/lamm-mit/MechAgents)]
*   An autonomous laboratory for the accelerated synthesis of novel materials [[Paper]](https://www.nature.com/articles/s41586-023-06734-w) ![](https://img.shields.io/badge/Nature-2023-green) [[Code](https://github.com/mattmcdermott/novel-materials-screening)]
*   Autonomous chemical research with large language models [[Paper]](https://doi.org/10.1038/s41586-023-06792-0) ![](https://img.shields.io/badge/Nature-2023-green) [[Code](https://github.com/gomesgroup/coscientist)]
*   LLM-in-the-loop: Leveraging Large Language Model for Thematic Analysis [[Paper]](https://aclanthology.org/2023.findings-emnlp.669/) ![](https://img.shields.io/badge/EMNLP_Findings-2023-blue) [[Code](https://github.com/sjdai/LLM-thematic-analysis)]
*   Training socially aligned language models on simulated social interactions [[Paper]](https://openreview.net/forum?id=NddKiWtdUm) ![](https://img.shields.io/badge/ICLR-2024-blue)
*   SWE-bench: Can Language Models Resolve Real-world Github Issues? [[Paper]](https://openreview.net/forum?id=VTF8yNQM66) ![](https://img.shields.io/badge/ICLR-2024-blue) [[Code](https://github.com/swe-bench/SWE-bench)]
*   Can Large Language Models Serve as Data Analysts? A Multi-Agent Assisted Approach for Qualitative Data Analysis [[Paper]](https://arxiv.org/abs/2402.01386) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv)
*   Automated Statistical Model Discovery with Language Models [[Paper]](https://arxiv.org/abs/2402.17879) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv)
*   Large language model agent for hyper-parameter optimization [[Paper]](https://doi.org/10.48550/arXiv.2402.01881) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv)
*   Mlcopilot: Unleashing the power of large language models in solving machine learning tasks [[Paper]](https://aclanthology.org/2024.eacl-long.179) ![](https://img.shields.io/badge/EACL-2024-blue) [[Code](https://github.com/microsoft/CoML)]
*   Researchagent: Iterative research idea generation over scientific literature with large language models [[Paper]](https://doi.org/10.48550/arXiv.2404.07738) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv) [[Code](https://github.com/JinheonBaek/ResearchAgent)]
*   Automated social science: Language models as scientist and subjects [[Paper]](https://arxiv.org/abs/2404.11794) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv)
*   Crispr-gpt: An llm agent for automated design of gene-editing experiments [[Paper]](https://doi.org/10.48550/arXiv.2404.18021) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv)
*   Position: LLMs can't plan, but can help planning in LLM-modulo frameworks [[Paper]](https://openreview.net/forum?id=Th8JPEmH4z) ![](https://img.shields.io/badge/ICML-2024-blue)
*   Augmenting large language models with chemistry tools [[Paper]](https://doi.org/10.1038/s42256-024-00832-8) ![](https://img.shields.io/badge/Nat._Mach._Intell.-2024-green) [[Code](https://github.com/ur-whitelab/chemcrow-public)]
*   Opening a conversation on responsible environmental data science in the age of large language models [[Paper]](https://doi.org/10.1017/eds.2024.12) ![](https://img.shields.io/badge/Environ._Data_Sci.-2024-green)
*   Meta-Designing Quantum Experiments with Language Models [[Paper]](https://doi.org/10.48550/arXiv.2406.02470) ![](https://img.shields.io/badge/arXiv-2024.06-red?logo=arxiv)
*   Automatic benchmarking of large multimodal models via iterative experiment programming [[Paper]](https://arxiv.org/abs/2406.12321) ![](https://img.shields.io/badge/arXiv-2024.06-red?logo=arxiv) [[Code](https://github.com/altndrr/apex)]
*   OpenHands: An Open Platform for AI Software Developers as Generalist Agents [[Paper]](https://arxiv.org/abs/2407.16741) ![](https://img.shields.io/badge/arXiv-2024.07-red?logo=arxiv) [[Code](https://github.com/All-Hands-AI/OpenHands)]
*   The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery [[Paper]](https://doi.org/10.48550/arXiv.2408.06292) ![](https://img.shields.io/badge/arXiv-2024.08-red?logo=arxiv) [[Code](https://github.com/SakanaAI/AI-Scientist)]
*   Mlr-copilot: Autonomous machine learning research based on large language models agents [[Paper]](https://doi.org/10.48550/arXiv.2408.14033) ![](https://img.shields.io/badge/arXiv-2024.08-red?logo=arxiv) [[Code](https://github.com/du-nlp-lab/MLR-Copilot)]
*   DSBench: How Far Are Data Science Agents to Becoming Data Science Experts? [[Paper]](https://arxiv.org/abs/2409.07703) ![](https://img.shields.io/badge/arXiv-2024.09-red?logo=arxiv) [[Code](https://github.com/LiqiangJing/DSBench)]
*   AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML [[Paper]](https://arxiv.org/abs/2410.02958) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv)
*   Chain of ideas: Revolutionizing research via novel idea development with llm agents [[Paper]](https://doi.org/10.48550/arXiv.2410.13185) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) [[Code](https://github.com/DAMO-NLP-SG/CoI-Agent)]
*   ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery [[Paper]](https://arxiv.org/abs/2410.05080) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) [[Code](https://github.com/OSU-NLP-Group/ScienceAgentBench)]
*   Agent-as-a-Judge: Evaluate Agents with Agents [[Paper]](https://arxiv.org/abs/2410.10934) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) [[Code](https://github.com/metauto-ai/agent-as-a-judge)]
*   SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning [[Paper]](https://arxiv.org/abs/2410.17238) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) [[Code](https://github.com/geekan/MetaGPT)]
*   AAAR-1.0: Assessing AI's Potential to Assist Research [[Paper]](https://doi.org/10.48550/arXiv.2410.22394) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv)
*   An automatic end-to-end chemical synthesis development platform powered by large language models [[Paper]](https://doi.org/10.1038/s41467-024-54457-x) ![](https://img.shields.io/badge/Nat._Commun.-2024-green) [[Code](https://github.com/Ruan-Yixiang/LLM-RDF)]
*   MatPilot: an LLM-enabled AI Materials Scientist under the Framework of Human-Machine Collaboration [[Paper]](https://doi.org/10.48550/arXiv.2411.08063) ![](https://img.shields.io/badge/arXiv-2024.11-red?logo=arxiv)
*   AI agents in chemical research: GVIM – an intelligent research assistant system [[Paper]](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00398e) ![](https://img.shields.io/badge/Digital_Discovery-2025-green) [[Code](https://github.com/KangyongMa/GVIM)]
*   Dolphin: Closed-loop Open-ended Auto-research through Thinking, Practice, and Feedback [[Paper]](https://doi.org/10.48550/arXiv.2501.03916) ![](https://img.shields.io/badge/arXiv-2025.01-red?logo=arxiv)
*   Agent Laboratory: Using LLM Agents as Research Assistants [[Paper]](https://arxiv.org/abs/2501.04227) ![](https://img.shields.io/badge/arXiv-2025.01-red?logo=arxiv) [[Code](https://github.com/SamuelSchmidgall/AgentLaboratory)]
*   SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains? [[Paper]](https://openreview.net/forum?id=riTiq3i21b) ![](https://img.shields.io/badge/ICLR_Sub-2025-blue)
*   AIDE: AI-Driven Exploration in the Space of Code [[Paper]](https://arxiv.org/abs/2502.13138) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv) [[Code](https://github.com/WecoAI/aideml)]
*   MLGym: A New Framework and Benchmark for Advancing AI Research Agents [[Paper]](https://arxiv.org/abs/2502.14499) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv) [[Code](https://github.com/facebookresearch/MLGym)]

# 📝 Paper Writing


<!-- omit in toc -->
## Main Writing & Summarization
*   Towards Automated Related Work Summarization [[Paper]](https://aclanthology.org/C10-2049/) ![](https://img.shields.io/badge/COLING-2010-blue)
*   Neural Related Work Summarization with a Joint Context-driven Attention Mechanism [[Paper]](https://aclanthology.org/D18-1204/) ![](https://img.shields.io/badge/EMNLP-2018-blue) [[Data](https://github.com/kuadmu/2018EMNLP)]
*   PaperRobot: Incremental Draft Generation of Scientific Ideas [[Paper]](https://aclanthology.org/P19-1191/) ![](https://img.shields.io/badge/ACL-2019-blue) [[Code](https://github.com/EagleW/PaperRobot)]
*   ScisummNet: A Large Annotated Corpus and Content-Impact Models for Scientific Paper Summarization with Citation Networks [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/4727) ![](https://img.shields.io/badge/AAAI-2019-blue)
*   Automatic Generation of Citation Texts in Scholarly Papers: A Pilot Study [[Paper]](https://aclanthology.org/2020.acl-main.550/) ![](https://img.shields.io/badge/ACL-2020-blue) [[Code](https://github.com/XingXinyu96/citation_generation)]
*   Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen [[Paper]](https://aclanthology.org/2020.acl-main.100/) ![](https://img.shields.io/badge/ACL-2020-blue) [[Project](https://srhthu.github.io/expertise-style-transfer/#disclaimer)]
*   Automatic related work section generation: experiments in scientific document abstracting [[Paper]](https://link.springer.com/article/10.1007/s11192-020-03630-2) ![](https://img.shields.io/badge/Scientometrics-2020-green) [[Code](https://github.com/AhmedAbuRaed/SPSeq2Seq)]
*   Automatic Related Work Section Generation by Sentence Extraction and Reordering [[Paper]](https://ceur-ws.org/Vol-2871/paper8.pdf) ![](https://img.shields.io/badge/BIR@ECIR-2021-blue)
*   Automatic Title Generation for Text with Pre-trained Transformer Language Model [[Paper]](https://ieeexplore.ieee.org/abstract/document/9364613) ![](https://img.shields.io/badge/IEEE_Access-2021-green) [[Code](https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control)]
*   AutoCite: Multi-Modal Representation Fusion for Contextual Citation Generation [[Paper]](https://dl.acm.org/doi/10.1145/3437963.3441739) ![](https://img.shields.io/badge/WSDM-2021-blue) [[Code](https://github.com/qqingwang/AutoCite)]
*   BACO: A Background Knowledge- and Content-Based Framework for Citing Sentence Generation [[Paper]](https://aclanthology.org/2021.acl-long.116/) ![](https://img.shields.io/badge/ACL-2021-blue)
*   Intent-Controllable Citation Text Generation [[Paper]](https://www.mdpi.com/2227-7390/10/10/1763) ![](https://img.shields.io/badge/Mathematics-2022-green) [[Code](https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control)]
*   Read, revise, repeat: A system demonstration for human-in-the-loop iterative text revision [[Paper]](https://doi.org/10.48550/arXiv.2204.03685) ![](https://img.shields.io/badge/ACL_Demo-2022-blue) [[Code](https://github.com/vipulraheja/IteraTeR)]
*   Generating Scientific Definitions with Controllable Complexity [[Paper]](https://aclanthology.org/2022.acl-long.569/) ![](https://img.shields.io/badge/ACL-2022-blue) [[Code](https://github.com/talaugust/definition-complexity)]
*   Understanding Iterative Revision from Human-Written Text [[Paper]](https://aclanthology.org/2022.acl-long.250/) ![](https://img.shields.io/badge/ACL-2022-blue) [[Code](https://github.com/vipulraheja/iterater)]
*   Disencite: Graph-based disentangled representation learning for context-specific citation generation [[Paper]](https://doi.org/10.1609/aaai.v36i10.21397) ![](https://img.shields.io/badge/AAAI-2022-blue)
*   CORWA: A Citation-Oriented Related Work Annotation Dataset [[Paper]](https://aclanthology.org/2022.naacl-main.397.pdf) ![](https://img.shields.io/badge/NAACL-2022-blue) [[Code](https://github.com/jacklxc/CORWA)]
*   Controllable citation sentence generation with language models [[Paper]](https://arxiv.org/abs/2211.07066) ![](https://img.shields.io/badge/arXiv-2022.11-red?logo=arxiv) [[Code](https://github.com/nianlonggu/LMCiteGen)]
*   Improving Iterative Text Revision by Learning Where to Edit from Other Revision Tasks [[Paper]](https://aclanthology.org/2022.emnlp-main.678/) ![](https://img.shields.io/badge/EMNLP-2022-blue) [[Code](https://github.com/vipulraheja/iterater)]
*   Making Science Simple: Corpora for the Lay Summarisation of Scientific Literature [[Paper]](https://aclanthology.org/2022.emnlp-main.724/) ![](https://img.shields.io/badge/EMNLP-2022-blue) [[Code](https://github.com/TGoldsack1/Corpora_for_Lay_Summarisation)]
*   Can artificial intelligence help for scientific writing? [[Paper]](https://link.springer.com/article/10.1186/s13054-023-04380-2) ![](https://img.shields.io/badge/Crit._Care-2023-green)
*   Text revision in scientific writing assistance: An overview [[Paper]](https://doi.org/10.48550/arXiv.2303.16726) ![](https://img.shields.io/badge/arXiv-2023.03-red?logo=arxiv)
*   Using ChatGPT for language editing in scientific articles [[Paper]](https://link.springer.com/content/pdf/10.1186/s40902-023-00381-x.pdf) ![](https://img.shields.io/badge/Gynecol._Oncol._Rep.-2023-green)
*   Good Practices for Scientific Article Writing with ChatGPT and Other Artificial Intelligence Language Models [[Paper]](https://www.mdpi.com/2673-687X/3/2/9) ![](https://img.shields.io/badge/Medicina-2023-green)
*   Comparing scientific abstracts generated by ChatGPT to real abstracts with detectors and blinded human reviewers [[Paper]](https://www.nature.com/articles/s41746-023-00819-6) ![](https://img.shields.io/badge/npj_Digit._Med.-2023-green)
*   Decoding the End-to-end Writing Trajectory in Scholarly Manuscripts [[Paper]](https://arxiv.org/abs/2304.00121) ![](https://img.shields.io/badge/In2Writing@CHI-2023-blue) [[Code](https://github.com/minnesotanlp/reward-system)]
*   Exploring the boundaries of reality: investigating the phenomenon of artificial intelligence hallucination in scientific writing through ChatGPT references [[Paper]](https://pubmed.ncbi.nlm.nih.gov/37182055/) ![](https://img.shields.io/badge/Cureus-2023-green)
*   The role of ChatGPT in scientific communication: writing better scientific review articles [[Paper]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10164801/) ![](https://img.shields.io/badge/Ann._Biomed._Eng.-2023-green)
*   SciLit: A platform for joint scientific literature discovery, summarization and citation generation [[Paper]](https://doi.org/10.18653/v1/2023.acl-demo.22) ![](https://img.shields.io/badge/ACL_Demo-2023-blue)
*   Artificial intelligence in scientific writing: a friend or a foe? [[Paper]](https://pubmed.ncbi.nlm.nih.gov/37142479/) ![](https://img.shields.io/badge/J._Transl._Med.-2023-green)
*   Transformers Go for the LOLs: Generating (Humourous) Titles from Scientific Abstracts End-to-End [[Paper]](https://aclanthology.org/2023.eval4nlp-1.6/) ![](https://img.shields.io/badge/EVAL4NLP@IJCNLP-2023-blue) [[Code](https://github.com/cyr19/A2T)]
*   Enabling large language models to generate text with citations [[Paper]](https://doi.org/10.18653/v1/2023.emnlp-main.398) ![](https://img.shields.io/badge/EMNLP-2023-blue) [[Code](https://github.com/princeton-nlp/ALCE)]
*   Towards a unified framework for reference retrieval and related work generation [[Paper]](https://doi.org/10.18653/v1/2023.findings-emnlp.385) ![](https://img.shields.io/badge/EMNLP_Findings-2023-blue)
*   ‘Don’t Get Too Technical with Me’: A Discourse Structure-Based Framework for Science Journalism [[Paper]](https://aclanthology.org/2023.emnlp-main.76.pdf) ![](https://img.shields.io/badge/EMNLP-2023-blue)
*   Leveraging Large Language Models for Literature Review Tasks - A Case Study Using ChatGPT [[Paper]](https://link.springer.com/chapter/10.1007/978-3-031-48858-0_25) ![](https://img.shields.io/badge/ICDM_Workshops-2023-blue)
*   Explaining Relationships Among Research Papers [[Paper]](https://arxiv.org/abs/2402.13426) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv)
*   Shallow synthesis of knowledge in gpt-generated texts: A case study in automatic related work composition [[Paper]](https://doi.org/10.48550/arXiv.2402.12255) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv)
*   LitLLM: A Toolkit for Scientific Literature Review [[Paper]](https://arxiv.org/abs/2402.01788) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv) [[Code](https://github.com/LitLLM/litllms-for-literature-review-tmlr)]
*   Techniques for supercharging academic writing with generative AI [[Paper]](https://www.nature.com/articles/s41551-024-01185-8) ![](https://img.shields.io/badge/Nat._Biomed._Eng.-2024-green)
*   Automating Research Synthesis with Domain-Specific Large Language Model Fine-Tuning [[Paper]](https://arxiv.org/abs/2404.08680) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv) [[Code](https://github.com/peterjhwang/slr-helper)]
*   Mapping the Increasing Use of LLMs in Scientific Papers [[Paper]](https://arxiv.org/abs/2404.01268v1) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv)
*   Autonomous LLM-Driven Research—from Data to Human-Verifiable Research Papers [[Paper]](https://doi.org/10.48550/arXiv.2404.17605) ![](https://img.shields.io/badge/arXiv-2024.04-red?logo=arxiv) [[Code](https://github.com/Technion-Kishony-lab/data-to-paper)]
*   A publishing infrastructure for Artificial Intelligence (AI)-assisted academic authoring [[Paper]](https://doi.org/10.1093/jamia/ocae139) ![](https://img.shields.io/badge/JAMIA-2024-green) [[Code](https://github.com/manubot/manubot-ai-editor)]
*   Step-Back Profiling: Distilling User History for Personalized Scientific Writing [[Paper]](https://doi.org/10.48550/arXiv.2406.14275) ![](https://img.shields.io/badge/arXiv-2024.06-red?logo=arxiv) [[Code](https://github.com/gersteinlab/step-back-profiling)]
*   AutoSurvey: Large Language Models Can Automatically Write Surveys [[Paper]](https://arxiv.org/abs/2406.10252) ![](https://img.shields.io/badge/arXiv-2024.06-red?logo=arxiv) [[Code](https://github.com/AutoSurveys/AutoSurvey)]
*   The Ability of ChatGPT in Paraphrasing Texts and Reducing Plagiarism: A Descriptive Analysis [[Paper]](https://pmc.ncbi.nlm.nih.gov/articles/PMC11250043/) ![](https://img.shields.io/badge/Cureus-2024-green)
*   Pelican: Correcting Hallucination in Vision-LLMs via Claim Decomposition and Program of Thought Verification [[Paper]](https://arxiv.org/abs/2407.02352) ![](https://img.shields.io/badge/arXiv-2024.07-red?logo=arxiv)
*   Reinforced Subject-Aware Graph Neural Network for Related Work Generation [[Paper]](https://doi.org/10.1007/978-981-97-5492-2_16) ![](https://img.shields.io/badge/KSEM-2024-blue)
*   Reference hallucination score for medical artificial intelligence chatbots: development and usability study [[Paper]](https://medinform.jmir.org/2024/1/e54345) ![](https://img.shields.io/badge/JMIR_Med._Inform.-2024-green)
*   LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs [[Paper]](https://arxiv.org/abs/2408.07055) ![](https://img.shields.io/badge/arXiv-2024.08-red?logo=arxiv) [[Code](https://github.com/THUDM/LongWriter)]
*   Automated focused feedback generation for scientific writing assistance [[Paper]](https://doi.org/10.18653/v1/2024.findings-acl.580) ![](https://img.shields.io/badge/ACL_Findings-2024-blue) [[Code](https://github.com/ericchamoun/FocusedFeedbackGeneration)]
*   Instruct Large Language Models to Generate Scientific Literature Survey Step by Step [[Paper]](https://arxiv.org/pdf/2408.07884) ![](https://img.shields.io/badge/NLPCC-2024-blue)
*   Aries: A corpus of scientific paper edits made in response to peer reviews [[Paper]](https://doi.org/10.18653/v1/2024.acl-long.377) ![](https://img.shields.io/badge/ACL-2024-blue) [[Code](https://github.com/allenai/aries)]
*   Toward Structured Related Work Generation with Novelty Statements [[Paper]](https://aclanthology.org/2024.sdp-1.5/) ![](https://img.shields.io/badge/SDP@ACL-2024-blue) [[Code](https://github.com/omron-sinicx/STRoGeNS)]
*   The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery [[Paper]](https://doi.org/10.48550/arXiv.2408.06292) ![](https://img.shields.io/badge/arXiv-2024.08-red?logo=arxiv) [[Code](https://github.com/SakanaAI/AI-Scientist)]
*   LongReward: Improving Long-context Large Language Models with AI Feedback [[Paper]](https://arxiv.org/abs/2410.21252) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) [[Code](https://github.com/THUDM/LongReward)]
*   Taxonomy Tree Generation from Citation Graph [[Paper]](https://arxiv.org/abs/2410.03761) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) (*Note: Same paper link as Hireview*)
*   CycleResearcher: Improving Automated Research via Automated Review [[Paper]](https://arxiv.org/abs/2411.00816) ![](https://img.shields.io/badge/arXiv-2024.11-red?logo=arxiv) [[Code](https://github.com/zhu-minjun/Researcher)]
*   Cocoa: Co-Planning and Co-Execution with AI Agents [[Paper]](https://doi.org/10.48550/arXiv.2412.10999) ![](https://img.shields.io/badge/arXiv-2024.12-red?logo=arxiv)
*   ResearchTown: Simulator of Human Research Community [[Paper]](https://doi.org/10.48550/arXiv.2412.17767) ![](https://img.shields.io/badge/arXiv-2024.12-red?logo=arxiv) [[Code](https://github.com/ulab-uiuc/research-town)]
*   Agent Laboratory: Using LLM Agents as Research Assistants [[Paper]](https://arxiv.org/abs/2501.04227) ![](https://img.shields.io/badge/arXiv-2025.01-red?logo=arxiv) [[Code](https://github.com/SamuelSchmidgall/AgentLaboratory)]
*   LongEval: A Comprehensive Analysis of Long-Text Generation Through a Plan-based Paradigm [[Paper]](https://arxiv.org/abs/2502.19103) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv) [[Code](https://github.com/Wusiwei0410/LongEval)]
*   CorpusStudio: Surfacing Emergent Patterns in a Corpus of Prior Work while Writing [[Paper]](https://arxiv.org/abs/2503.12436) ![](https://img.shields.io/badge/ACM_CHI-2025-blue)

<!-- omit in toc -->
## Figure Generation
*   Data2Vis: Automatic Generation of Data Visualizations Using Sequence-to-Sequence Recurrent Neural Networks [[Paper]](https://ieeexplore.ieee.org/document/8744242) ![](https://img.shields.io/badge/IEEE_TVCG-2019-green) [[Code](https://github.com/victordibia/data2vis)]
*   ADVISor: Automatic Visualization Answer for Natural-Language Question on Tabular Data [[Paper]](https://ieeexplore.ieee.org/document/9438784) ![](https://img.shields.io/badge/IEEE_PacificVis-2021-blue)
*   SCICAP: Generating Captions for Scientific Figures [[Paper]](https://aclanthology.org/2021.findings-emnlp.277/) ![](https://img.shields.io/badge/EMNLP_Findings-2021-blue) [[Code](https://github.com/tingyaohsu/scicap)]
*   Sevi: Speech-to-Visualization through Neural Machine Translation [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3514221.3520150) ![](https://img.shields.io/badge/IUI_Companion-2022-blue) [[Code](https://github.com/Thanksyy/Sevi)]
*   Chat2VIS: Generating Data Visualizations via Natural Language Using ChatGPT, Codex and GPT-3 Large Language Models [[Paper]](https://ieeexplore.ieee.org/abstract/document/10121440) ![](https://img.shields.io/badge/IEEE_PacificVis_Wkshp-2023-blue)
*   AutomaTikZ: Text-Guided Synthesis of Scientific Vector Graphics with TikZ [[Paper]](https://openreview.net/forum?id=v3K5TVP8kZ) ![](https://img.shields.io/badge/ICLR-2024-blue) [[Code](https://github.com/potamides/AutomaTikZ)]
*   Plots Made Quickly: An Efficient Approach for Generating Visualizations from Natural Language Queries [[Paper]](https://aclanthology.org/2024.lrec-main.1119/) ![](https://img.shields.io/badge/LREC--COLING-2024-blue)
*   DiagrammerGPT: Generating Open-Domain, Open-Platform Diagrams via LLM Planning [[Paper]](https://openreview.net/forum?id=NV8yRJRET1#discussion) ![](https://img.shields.io/badge/ACL-2024-blue) [[Code](https://github.com/aszala/DiagrammerGPT)]
*   SciDoc2Diagrammer-MAF: Towards Generation of Scientific Diagrams from Documents guided by Multi-Aspect Feedback Refinement [[Paper]](https://aclanthology.org/2024.findings-emnlp.780/) ![](https://img.shields.io/badge/EMNLP_Findings-2024-blue) [[Code](https://github.com/Ishani-Mondal/SciDoc2DiagramGeneration)]
*   DeTikZify: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ [[Paper]](https://openreview.net/forum?id=bcVLFQCOjc) ![](https://img.shields.io/badge/EMNLP-2024-blue) [[Code](https://github.com/potamides/DeTikZify)]
*   ChartMimic: Evaluating LMM's Cross-Modal Reasoning Capability via Chart-to-Code Generation [[Paper]](https://openreview.net/forum?id=sGpCzsfd1K) ![](https://img.shields.io/badge/arXiv-2025.03-red?logo=arxiv) [[Code](https://github.com/ChartMimic/ChartMimic)]
*   ScImage: How Good Are Multimodal Large Language Models at Scientific Text-to-Image Generation? [[Paper]](https://openreview.net/forum?id=ugyqNEOjoU) ![](https://img.shields.io/badge/arXiv-2025.03-red?logo=arxiv) [[Code](https://github.com/leixin-zhang/Scimage)]

<!-- omit in toc -->
## Table Generation
*   gTBLS: Generating Tables from Text by Conditional Question Answering [[Paper]](https://arxiv.org/abs/2403.14457) ![](https://img.shields.io/badge/arXiv-2024.03-red?logo=arxiv)
*   OpenTE: Open-Structure Table Extraction From Text [[Paper]](https://ieeexplore.ieee.org/document/10448427) ![](https://img.shields.io/badge/IEEE_Access-2024-green)
*   ArxivDIGESTables: Synthesizing Scientific Literature into Tables using Language Models [[Paper]](https://aclanthology.org/2024.emnlp-main.538/) ![](https://img.shields.io/badge/EMNLP-2024-blue) [[Code](https://github.com/bnewm0609/arxivDIGESTables)]
*   Is This a Bad Table? A Closer Look at the Evaluation of Table Generation from Text [[Paper]](https://aclanthology.org/2024.emnlp-main.1239/) ![](https://img.shields.io/badge/EMNLP-2024-blue)
*   LATTE: Improving Latex Recognition for Tables and Formulae with Iterative Refinement [[Paper]](https://arxiv.org/abs/2409.14201) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv)

<!-- omit in toc -->
## Slides & Poster Generation
*   SlidesGen: Automatic Generation of Presentation Slides for a Technical Paper Using Summarization [[Paper]](https://aaai.org/papers/flairs-2009-22/) ![](https://img.shields.io/badge/FLAIRS-2009-blue)
*   PPSGen: learning to generate presentation slides for academic papers [[Paper]](https://dl.acm.org/doi/10.5555/2540128.2540430) ![](https://img.shields.io/badge/KDD-2013-blue)
*   Learning to Generate Posters of Scientific Papers [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/10000) ![](https://img.shields.io/badge/AAAI-2016-blue)
*   Phrase-Based Presentation Slides Generation for Academic Papers [[Paper]](https://aaai.org/papers/10481-aaai-31-2017/) ![](https://img.shields.io/badge/AAAI-2017-blue)
*   Towards Topic-Aware Slide Generation For Academic Papers With Unsupervised Mutual Learning [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/17564) ![](https://img.shields.io/badge/AAAI-2021-blue) [[Code](https://github.com/daviddwlee84/TopicAwarePaperSlideGeneration?tab=readme-ov-file)]
*   D2S: Document-to-Slide Generation Via Query-Based Text Summarization [[Paper]](https://aclanthology.org/2021.naacl-main.111/) ![](https://img.shields.io/badge/NAACL-2021-blue) [[Code](https://github.com/IBM/document2slides)]
*   DOC2PPT: Automatic Presentation Slides Generation from Scientific Documents [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/19943) ![](https://img.shields.io/badge/AAAI-2022-blue)
*   PosterBot: A System for Generating Posters of Scientific Papers with Neural Models [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/21738) ![](https://img.shields.io/badge/AAAI_Demo-2022-blue)
*   Presentations by the Humans and For the Humans: Harnessing LLMs for Generating Persona-Aware Slides from Documents [[Paper]](https://aclanthology.org/2024.eacl-long.163/) ![](https://img.shields.io/badge/EACL-2024-blue)
*   Enhancing Presentation Slide Generation by LLMs with a Multi-Staged End-to-End Approach [[Paper]](https://aclanthology.org/2024.inlg-main.18/) ![](https://img.shields.io/badge/INLG-2024-blue)
*   Presentations are not always linear! GNN meets LLM for Text Document-to-Presentation Transformation with Attribution [[Paper]](https://aclanthology.org/2024.findings-emnlp.936/) ![](https://img.shields.io/badge/EMNLP_Findings-2024-blue)

# ✅ Peer Review


*   Online software spots genetic errors in cancer papers [[Paper]](https://www.nature.com/articles/nature.2017.23003) ![](https://img.shields.io/badge/Nature_News-2017-green)
*   Argument Mining for Understanding Peer Reviews [[Paper]](https://aclanthology.org/N19-1219/) ![](https://img.shields.io/badge/NAACL-2019-blue)
*   DeepSentiPeer: Harnessing Sentiment in Review Texts to Recommend Peer Review Decisions [[Paper]](https://aclanthology.org/P19-1106/) ![](https://img.shields.io/badge/ACL-2019-blue) [[Code](https://github.com/rajevv/DeepSentiPeer)]
*   Exploring the Potential of GPT-2 for Generating Fake Reviews of Research Papers [[Paper]](https://ebooks.iospress.nl/doi/10.3233/FAIA200717) ![](https://img.shields.io/badge/FAIA@ECAI-2020-blue) [[Code](https://github.com/allenai/PeerRead)]
*   Aspect-based Sentiment Analysis of Scientific Reviews [[Paper]](https://dl.acm.org/doi/10.1145/3383583.3398541) ![](https://img.shields.io/badge/JCDL-2020-blue)
*   ReviewRobot: Explainable Paper Review Generation based on Knowledge Synthesis [[Paper]](https://aclanthology.org/2020.inlg-1.44/) ![](https://img.shields.io/badge/INLG-2020-blue) [[Code](https://github.com/EagleW/ReviewRobot)]
*   Uncertainty-aware machine support for paper reviewing on the interspeech 2019 submission corpus [[Paper]](https://www.isca-archive.org/interspeech_2020/stappen20_interspeech.html) ![](https://img.shields.io/badge/Interspeech-2020-blue) [[Code](https://github.com/TUD-STKS/U-Interspeech19)]
*   Multi-task Peer-Review Score Prediction [[Paper]](https://aclanthology.org/2020.sdp-1.14/) ![](https://img.shields.io/badge/SDP@EMNLP-2020-blue)
*   APE: Argument Pair Extraction from Peer Review and Rebuttal via Multi-task Learning [[Paper]](https://aclanthology.org/2020.emnlp-main.569/) ![](https://img.shields.io/badge/EMNLP-2020-blue) [[Code](https://github.com/LiyingCheng95/ArgumentPairExtraction)]
*   Can We Automate Scientific Reviewing? [[Paper]](https://doi.org/10.1613/jair.1.12862) ![](https://img.shields.io/badge/JAIR-2022-blue) [[Code](https://github.com/neulab/ReviewAdvisor)]
*   Argument Mining Driven Analysis of Peer-Reviews [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/16607) ![](https://img.shields.io/badge/AAAI-2021-blue) [[Code](https://github.com/fromm-m/aaai2021-am-peer-reviews)]
*   A Deep Neural Architecture for Decision-Aware Meta-Review Generation [[Paper]](https://ieeexplore.ieee.org/document/9651825) ![](https://img.shields.io/badge/IEEE_Access-2021-green)
*   PEERAssist: Leveraging on Paper-Review Interactions to Predict Peer Review Decisions [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-91669-5_33) ![](https://img.shields.io/badge/ICONIP-2021-blue) [[Code](https://github.com/PrabhatkrBharti/PEERAssist)]
*   HedgePeer: A Dataset for Uncertainty Detection in Peer Reviews [[Paper]](https://ieeexplore.ieee.org/document/9852963) ![](https://img.shields.io/badge/IEEE_Access-2022-green) [[Code](https://github.com/Tirthankar-Ghosal/HedgePeer-Dataset)]
*   KID-Review: Knowledge-Guided Scientific Review Generation with Oracle Pre-training [[Paper]](https://doi.org/10.1609/aaai.v36i10.21418) ![](https://img.shields.io/badge/AAAI-2022-blue) <!-- Code Link Broken -->
*   Assessing Scientific Research Papers with Knowledge Graphs [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3477495.3531879) ![](https://img.shields.io/badge/SIGIR-2022-blue) [[Code](https://github.com/kianasun/kg4rr)]
*   SciFact-Open: Towards open-domain scientific claim verification [[Paper]](https://arxiv.org/abs/2210.13777) ![](https://img.shields.io/badge/arXiv-2022.10-red?logo=arxiv) [[Code](https://github.com/kianasun/kg4rr)]
*   The Quality Assist: A Technology-Assisted Peer Review Based on Citation Functions to Predict the Paper Quality [[Paper]](https://doi.org/10.1109/ACCESS.2022.3225871) ![](https://img.shields.io/badge/IEEE_Access-2022-green)
*   Exploiting Labeled and Unlabeled Data via Transformer Fine-tuning for Peer-Review Score Prediction [[Paper]](https://doi.org/10.18653/v1/2022.findings-emnlp.164) ![](https://img.shields.io/badge/EMNLP_Findings-2022-blue) [[Code](https://github.com/panitan-m/gamma_trans)]
*   CARE: Collaborative AI-Assisted Reading Environment [[Paper]](https://arxiv.org/abs/2302.12611v1) ![](https://img.shields.io/badge/arXiv-2023.02-red?logo=arxiv) [[Code](https://github.com/UKPLab/CARE)]
*   PolitePEER: does peer review hurt? A dataset to gauge politeness intensity in the peer reviews [[Paper]](https://link.springer.com/article/10.1007/s10579-023-09662-3) ![](https://img.shields.io/badge/Scientometrics-2023-green) [[Code](https://github.com/PrabhatkrBharti/PolitePEER)]
*   Scientific Fact-Checking: A Survey of Resources and Approaches [[Paper]](https://arxiv.org/abs/2305.16859) ![](https://img.shields.io/badge/arXiv-2023.05-red?logo=arxiv)
*   Scientific Opinion Summarization: Meta-review Generation with Checklist-guided Iterative Introspection [[Paper]](https://doi.org/10.48550/arXiv.2305.14647) ![](https://img.shields.io/badge/arXiv-2023.05-red?logo=arxiv) [[Code](https://github.com/Mankeerat/orsum-meta-review-generation)]
*   Gpt4 is slightly helpful for peer-review assistance: A pilot study [[Paper]](https://doi.org/10.48550/arXiv.2307.05492) ![](https://img.shields.io/badge/arXiv-2023.06-red?logo=arxiv) [[Code](https://github.com/zrobertson466920/GPT_Auto_Review)]
*   ReviewerGPT? An Exploratory Study on Using Large Language Models for Paper Reviewing [[Paper]](https://doi.org/10.48550/arXiv.2306.00622) ![](https://img.shields.io/badge/arXiv-2023.06-red?logo=arxiv) [[Code](https://github.com/niharshah/ReviewerGPT2023)]
*   Unveiling the Sentinels: Assessing AI Performance in Cybersecurity Peer Review [[Paper]](https://doi.org/10.48550/arXiv.2309.05457) ![](https://img.shields.io/badge/arXiv-2023.09-red?logo=arxiv)
*   ChatGPT and the Future of Journal Reviews: A Feasibility Study [[Paper]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10524821/#:~:text=Contextual%20Understanding%20and%20Expertise%3A%20ChatGPT,accuracy%20of%20complex%20research%20findings) ![](https://img.shields.io/badge/Cureus-2023-green)
*   A Critical Examination of the Ethics of AI-Mediated Peer Review [[Paper]](https://arxiv.org/abs/2309.12356v1) ![](https://img.shields.io/badge/arXiv-2023.09-red?logo=arxiv)
*   Can large language models provide useful feedback on research papers? A large-scale empirical analysis [[Paper]](https://doi.org/10.48550/arXiv.2310.01783) ![](https://img.shields.io/badge/arXiv-2023.10-red?logo=arxiv) [[Code](https://github.com/Weixin-Liang/LLM-scientific-feedback)]
*   Automatic Analysis of Substantiation in Scientific Peer Reviews [[Paper]](https://arxiv.org/abs/2311.11967v1) ![](https://img.shields.io/badge/arXiv-2023.11-red?logo=arxiv) [[Code](https://github.com/YanzhuGuo/SubstanReview)]
*   Summarizing Multiple Documents with Conversational Structure for Meta-Review Generation [[Paper]](https://doi.org/10.18653/v1/2023.findings-emnlp.472) ![](https://img.shields.io/badge/EMNLP_Findings-2023-blue) [[Code](https://github.com/oaimli/PeerSum)]
*   When Reviewers Lock Horn: Finding Disagreement in Scientific Peer Reviews [[Paper]](https://doi.org/10.18653/v1/2023.emnlp-main.1038) ![](https://img.shields.io/badge/EMNLP-2023-blue) [[Code](https://github.com/sandeep82945/Contradiction-in-Peer-Review)]
*   PaperMage: A Unified Toolkit for Processing, Representing, and Manipulating Visually-Rich Scientific Documents [[Paper]](https://aclanthology.org/2023.emnlp-demo.45.pdf) ![](https://img.shields.io/badge/EMNLP_Demo-2023-blue) [[Code](https://github.com/allenai/papermage)]
*   The Intended Uses of Automated Fact-Checking Artefacts: Why, How and Who [[Paper]](https://aclanthology.org/2023.findings-emnlp.577/) ![](https://img.shields.io/badge/EMNLP_Findings-2023-blue) [[Code](https://github.com/Cartus/Automated-Fact-Checking-Resources)]
*   Exploring Jiu-Jitsu Argumentation for Writing Peer Review Rebuttals [[Paper]](https://aclanthology.org/2023.emnlp-main.894/) ![](https://img.shields.io/badge/EMNLP-2023-blue) [[Code](https://github.com/UKPLab/EMNLP2023_jiu_jitsu_argumentation_for_rebuttals)]
*   MARG: Multi-Agent Review Generation for Scientific Papers [[Paper]](https://doi.org/10.48550/arXiv.2401.04259) ![](https://img.shields.io/badge/arXiv-2024.01-red?logo=arxiv) [[Code](https://github.com/allenai/marg-reviewer)]
*   ReviewFlow: Intelligent Scaffolding to Support Academic Peer Reviewing [[Paper]](https://arxiv.org/abs/2402.03530) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv) [[Code](https://github.com/alexandermoore/ReviewFlow)]
*   Reviewer2: Optimizing Review Generation Through Prompt Generation [[Paper]](https://arxiv.org/abs/2402.10886v1) ![](https://img.shields.io/badge/arXiv-2024.02-red?logo=arxiv)
*   Emerging plagiarism in peer-review evaluation reports: a tip of the iceberg? [[Paper]](https://link.springer.com/article/10.1007/s11192-024-04960-1) ![](https://img.shields.io/badge/Scientometrics-2024-green)
*   MetaWriter: Exploring the Potential and Perils of AI Writing Support in Scientific Peer Review [[Paper]](https://doi.org/10.1145/3637371) ![](https://img.shields.io/badge/PACMHCI@CSCW-2024-blue)
*   How We Refute Claims: Automatic Fact-Checking through Flaw Identification and Explanation [[Paper]](https://dl.acm.org/doi/10.1145/3589335.3651521) ![](https://img.shields.io/badge/TheWebConf-2024-blue) [[Code](https://github.com/NYCU-NLP-Lab/FlawCheck)]
*   What Can Natural Language Processing Do for Peer Review? [[Paper]](https://arxiv.org/abs/2405.06563) ![](https://img.shields.io/badge/arXiv-2024.05-red?logo=arxiv) [[Code](https://github.com/OAfzal/nlp-for-peer-review)]
*   Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews [[Paper]](https://arxiv.org/abs/2403.07183v2) ![](https://img.shields.io/badge/arXiv-2024.06-red?logo=arxiv) [[Code](https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers)]
*   RelevAI-Reviewer: A Benchmark on AI Reviewers for Survey Paper Relevance [[Paper]](https://doi.org/10.48550/arXiv.2406.10294) ![](https://img.shields.io/badge/arXiv-2024.06-red?logo=arxiv) [[Code](https://github.com/paulohenriquecrs/RelevAI-Reviewer)]
*   Peer Review as A Multi-Turn and Long-Context Dialogue with Role-Based Interactions [[Paper]](https://doi.org/10.48550/arXiv.2406.05688) ![](https://img.shields.io/badge/arXiv-2024.06-red?logo=arxiv) [[Code](https://github.com/chengtan9907/ReviewMT)]
*   Automated Peer Reviewing in Paper SEA: Standardization, Evaluation, and Analysis [[Paper]](https://arxiv.org/abs/2407.12857v2) ![](https://img.shields.io/badge/arXiv-2024.07-red?logo=arxiv) [[Code](https://github.com/ecnu-sea/sea)]
*   AI-Driven Review Systems: Evaluating LLMs in Scalable and Bias-Aware Academic Reviews [[Paper]](https://arxiv.org/pdf/2408.10365) ![](https://img.shields.io/badge/arXiv-2024.08-red?logo=arxiv) [[Code](https://github.com/EagleW/ReviewRobot)]
*   GLIMPSE: Pragmatically Informative Multi-Document Summarization for Scholarly Reviews [[Paper]](https://doi.org/10.18653/v1/2024.acl-long.688) ![](https://img.shields.io/badge/ACL-2024-blue)
*   Automated Focused Feedback Generation for Scientific Writing Assistance [[Paper]](https://doi.org/10.18653/v1/2024.findings-acl.580) ![](https://img.shields.io/badge/ACL_Findings-2024-blue) [[Code](https://github.com/ericchamoun/FocusedFeedbackGeneration)]
*   A sentiment consolidation framework for meta-review generation [[Paper]](https://doi.org/10.18653/v1/2024.acl-long.547) ![](https://img.shields.io/badge/ACL-2024-blue) [[Code](https://github.com/oaimli/MetaReviewingLogic)]
*   Missci: Reconstructing Fallacies in Misrepresented Science [[Paper]](https://aclanthology.org/2024.acl-long.240/) ![](https://img.shields.io/badge/ACL-2024-blue) [[Code](https://github.com/UKPLab/acl2024-missci)]
*   The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery [[Paper]](https://doi.org/10.48550/arXiv.2408.06292) ![](https://img.shields.io/badge/arXiv-2024.08-red?logo=arxiv) [[Code](https://github.com/SakanaAI/AI-Scientist)]
*   The emergence of Large Language Models (LLM) as a tool in literature reviews: an LLM automated systematic review [[Paper]](https://arxiv.org/abs/2409.04600v1) ![](https://img.shields.io/badge/arXiv-2024.09-red?logo=arxiv)
*   Language agents achieve superhuman synthesis of scientific knowledge [[Paper]](https://arxiv.org/abs/2409.13740) ![](https://img.shields.io/badge/arXiv-2024.09-red?logo=arxiv)
*   AgentReview: Exploring Peer Review Dynamics with LLM Agents [[Paper]](https://aclanthology.org/2024.emnlp-main.70) ![](https://img.shields.io/badge/EMNLP-2024-blue) [[Code](https://github.com/Ahren09/AgentReview)]
*   Peer Reviews of Peer Reviews: A Randomized Controlled Trial and Other Experiments [[Paper]](https://arxiv.org/abs/2311.09497) ![](https://img.shields.io/badge/arXiv-2024.11-red?logo=arxiv)
*   CycleResearcher: Improving Automated Research via Automated Review [[Paper]](https://doi.org/10.48550/arXiv.2411.00816) ![](https://img.shields.io/badge/arXiv-2024.11-red?logo=arxiv) [[Code](https://github.com/zhu-minjun/Researcher)]
*   On the Rigour of Scientific Writing: Criteria, Analysis, and Insights [[Paper]](https://aclanthology.org/2024.findings-emnlp.380/) ![](https://img.shields.io/badge/EMNLP_Findings-2024-blue)
*   ResearchTown: Simulator of Human Research Community [[Paper]](https://doi.org/10.48550/arXiv.2412.17767) ![](https://img.shields.io/badge/arXiv-2024.12-red?logo=arxiv) [[Code](https://github.com/ulab-uiuc/research-town)]
*   Prompting LLMs to Compose Meta-Review Drafts from Peer-Review Narratives of Scholarly Manuscripts [[Paper]](https://doi.org/10.48550/arXiv.2402.15589) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv) [[Code](https://github.com/BridgeAI-Lab/LLM-as-Meta-Reviewer)]
*   Grounding Fallacies Misrepresenting Scientific Publications in Evidence [[Paper]](https://arxiv.org/abs/2408.12812) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv) [[Code](https://github.com/UKPLab/naacl2025-missciplus)]
*   Claim Verification in the Age of Large Language Models: A Survey [[Paper]](https://arxiv.org/abs/2408.14317) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv)
*   PeerArg: Argumentative Peer Review with LLMs [[Paper]](https://doi.org/10.48550/arXiv.2409.16813) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv)
*   OpenReviewer: A Specialized Large Language Model for Generating Critical Scientific Paper Reviews [[Paper]](https://doi.org/10.48550/arXiv.2412.11948) ![](https://img.shields.io/badge/arXiv-2025.03-red?logo=arxiv) [[Code](https://huggingface.co/maxidl/Llama-OpenReviewer-8B)]
*   ReviewAgents: Bridging the Gap Between Human and AI-Generated Paper Reviews [[Paper]](https://arxiv.org/abs/2503.08506) ![](https://img.shields.io/badge/arXiv-2025.03-red?logo=arxiv)
*   DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking Process [[Paper]](https://arxiv.org/abs/2503.08569) ![](https://img.shields.io/badge/arXiv-2025.03-red?logo=arxiv) [[Code](https://github.com/zhu-minjun/Researcher)]
*   Enabling Inclusive Systematic Reviews: Incorporating Preprint Articles with Large Language Model-Driven Evaluations [[Paper]](https://arxiv.org/abs/2503.13857) ![](https://img.shields.io/badge/arXiv-2025.03-red?logo=arxiv)
*   Overview of the Context24 Shared Task on Contextualizing Scientific Claims [[Paper]](https://aclanthology.org/2024.sdp-1.3/) ![](https://img.shields.io/badge/SDP@ACL-2024-blue)

# 🏆 Benchmarks

*   **OpenAI Science Benchmark** [[Leaderboard]](https://openai.com/research/science-benchmarks)
*   **AI2 Scientific Reasoning Challenge (ARC)** [[Dataset]](https://allenai.org/data/arc)
*   **MLAgentBench** [[Paper]](https://arxiv.org/abs/2310.03302) ![](https://img.shields.io/badge/arXiv-2023.10-red?logo=arxiv) [[Code](https://github.com/snap-stanford/MLAgentBench/)]
*   **SWE-bench** [[Paper]](https://openreview.net/forum?id=VTF8yNQM66) ![](https://img.shields.io/badge/ICLR-2024-blue) [[Code](https://github.com/swe-bench/SWE-bench)]
*   **ScienceAgentBench** [[Paper]](https://arxiv.org/abs/2410.05080) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv) [[Code](https://github.com/OSU-NLP-Group/ScienceAgentBench)]
*   **DSBench** [[Paper]](https://arxiv.org/abs/2409.07703) ![](https://img.shields.io/badge/arXiv-2024.09-red?logo=arxiv) [[Code](https://github.com/LiqiangJing/DSBench)]
*   **AAAR-1.0** [[Paper]](https://doi.org/10.48550/arXiv.2410.22394) ![](https://img.shields.io/badge/arXiv-2024.10-red?logo=arxiv)
*   **MLGym** [[Paper]](https://arxiv.org/abs/2502.14499) ![](https://img.shields.io/badge/arXiv-2025.02-red?logo=arxiv) [[Code](https://github.com/facebookresearch/MLGym)]
*   **PaperBench** [[Paper]](https://cdn.openai.com/papers/22265bac-3191-44e5-b057-7aaacd8e90cd/paperbench.pdf) ![](https://img.shields.io/badge/OpenAI_Report-2025-brown) [[Code](https://github.com/openai/preparedness)]

# 🌐 Community

<!-- omit in toc -->
## Upcoming Events

*   **(Past) NeurIPS 2024 AI4Science Workshop** (December 2024)
*   **(Past) AAAI Conference on Automated Scientific Discovery** (February 2025)
*   *(Suggest adding relevant upcoming events)*

<!-- omit in toc -->
## Discussion Forums

*   [AI Research Discord](https://discord.gg/airesearch) (Example - replace if needed)
*   [r/MachineLearning](https://reddit.com/r/MachineLearning)
*   *(Suggest adding forums specific to AI in science)*

# 👋 Contributing

We welcome contributions! Please follow this workflow:

1.  **Fork** the repository.
2.  **Create a new branch:** `git checkout -b feature/your-contribution`
3.  **Add or modify resources.** Please try to follow the existing format, including adding conference/journal badges.
4.  **Submit a Pull Request (PR)** with a clear description of your changes.

*For detailed guidelines, please see `CONTRIBUTING.md` (if available).*

# 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

# 🎖️ Credits and Acknowledgements

*   Maintained with ❤️ by [] and the community contributors.
*   Last major content update: **Apr 2025** (Based on initial memory storage date)
