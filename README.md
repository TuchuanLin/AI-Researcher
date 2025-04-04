# 🚀 AI Researcher Ecosystem: Empowering Scientific Discovery with AI

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/yourusername/ai-researcher-ecosystem/pulls)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Last Updated](https://img.shields.io/github/last-commit/yourusername/ai-researcher-ecosystem)](https://github.com/yourusername/ai-researcher-ecosystem/commits/main) <!-- 可选：显示上次更新时间 -->

<p align="center">
  <img src="https://via.placeholder.com/800x400.png?text=AI+Research+Automation+Landscape" alt="AI Research Automation Landscape" width="800">
</p>

**Dive into the rapidly evolving world of AI-powered research automation!** 💡 This curated repository is your compass to navigate the cutting-edge tools, frameworks, and methodologies that are revolutionizing scientific discovery. We track the latest advancements, focusing on resources that empower researchers and accelerate breakthroughs.

---

## 🔖 Table of Contents

- [Core Components](#core-components)
    - [Research Automation Stack](#research-automation-stack)
    - [Emerging Capabilities](#emerging-capabilities)
- [Featured Projects 🔥](#featured-projects-🔥)
    - [Open Source Tools](#open-source-tools)
    - [Commercial Platforms](#commercial-platforms)
- [Key Papers 📚](#key-papers-📚)
    - [Survey & Overviews](#survey--overviews)
    - [Knowledge Building](#knowledge-building)
        - [Knowledge Acquisition](#knowledge-acquisition)
        - [Knowledge Utilization](#knowledge-utilization)
    - [Idea Generation](#idea-generation)
        - [Generation Strategy](#generation-strategy)
        - [Generation Method](#generation-method)
        - [Internal Evaluation](#internal-evaluation)
    - [Experiment Execution](#experiment-execution)
        - [Experiment Design](#experiment-design)
        - [Experiment Conduction](#experiment-conduction)
        - [Internal Evaluation](#internal-evaluation)
    - [Paper Writing](#paper-writing)
    - [Peer Review](#peer-review)
- [Benchmarks 🏆](#benchmarks-🏆)
- [Community 🌐](#community-🌐)
    - [Upcoming Events](#upcoming-events)
    - [Discussion Forums](#discussion-forums)
- [Contributing 👋](#contributing-👋)
- [License 📜](#license-📜)
- [Credits and Acknowledgements](#credits-and-acknowledgements) 

---


## ⚙️ Core Components

This section breaks down the ecosystem into key functional areas and emerging trends, reflecting the stages outlined in the Key Papers section.

### Research Automation Stack

The fundamental layers where AI is impacting the research lifecycle:

| Layer                 | Key Papers Category Alignment | Example Technologies / Concepts                                                                      | Description                                                                                                                                  |
| :-------------------- | :---------------------------- | :--------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- |
| **Knowledge Building** | Knowledge Building            | [Elicit](https://elicit.org), [Consensus](https://consensus.app), SciSpace, Semantic Scholar, KG Tools | AI tools for literature discovery, summarization, knowledge graph construction, understanding figures/tables, and synthesizing existing work. |
| **Idea Generation**   | Idea Generation               | Hypothesis generators (e.g., based on KGs, LLMs), Agent-based exploration (e.g., [ResearchAgent](https://github.com/JinheonBaek/ResearchAgent)) | AI assisting in formulating novel research questions, hypotheses, identifying gaps, and exploring new research directions.                 |
| **Experimentation**   | Experiment Execution          | Autonomous Labs (e.g., [CoScientist](https://github.com/gomesgroup/coscientist)), [AlphaFold](https://alphafold.ebi.ac.uk), AutoML, Agent-driven simulation | AI aiding in experimental design, automating experiment execution (physical or simulated), data collection, analysis, and interpretation. |
| **Paper Writing**     | Paper Writing                 | [Manubot](https://manubot.org), [Overleaf AI](https://www.overleaf.com), [PaperRobot](https://github.com/EagleW/PaperRobot), Citation/Figure generators | AI assistance in drafting sections (abstract, related work), generating figures/tables, citation management, editing, and formatting.     |
| **Peer Review**       | Peer Review                   | Reviewer assignment tools, Argument mining, Review generation aids (e.g., [ReviewRobot](https://github.com/EagleW/ReviewRobot))            | AI tools supporting reviewer assignment, review quality assessment, argument extraction, and potentially review generation/summarization.      |

### Emerging Capabilities

Exciting frontiers pushing the boundaries of AI in research:

-   **End-to-End Research Agents:** AI systems attempting to automate multiple stages of the research cycle autonomously (e.g., The AI Scientist, AutoSurvey).
-   **Human-AI Collaboration Patterns:** Novel workflows and interfaces integrating AI assistance seamlessly into researcher tasks for enhanced productivity and creativity.
-   **AI for Reproducibility & Validation:** Tools focused on verifying scientific claims, attempting automated replication of results, and ensuring overall scientific rigor.


---

## 🔥 Featured Projects

### Open Source Tools

| Project | Description | Stars | Link |
|---------|-------------|-------|------|
| [Galactica](https://github.com/paperswithcode/galai) | Meta's scientific language model for research tasks. | ![](https://img.shields.io/github/stars/paperswithcode/galai) | [GitHub](https://github.com/paperswithcode/galai) |
| [SciKit-LLM](https://github.com/iryna-kondr/scikit-llm) | Integration of LLMs into scientific workflows. | ![](https://img.shields.io/github/stars/iryna-kondr/scikit-llm) | [GitHub](https://github.com/iryna-kondr/scikit-llm) |

### Commercial Platforms

- [Wolfram Research Assistant](https://www.wolfram.com/research-assistant/)
- [IBM Watson Discovery](https://www.ibm.com/cloud/watson-discovery)

---

## 📚 Key Papers

_(Note: This section provides a non-exhaustive list of influential and recent papers. Contributions welcome!)_

### Survey & Overviews

*   _(Placeholder for survey papers - contributions needed!)_
    *   *Example Format:* **A Survey on AI for Scientific Discovery** *ACM Computing Surveys* (2024) [[arXiv]](link)

### Knowledge Building _(Contributor: linzhen)_

#### Knowledge Acquisition

##### Main Acquisition & Recommendation

-   **From Who You Know to What You Read: Augmenting Scientific Recommendations with Implicit Social Networks** *CHI 2022, 2022.04* [[Paper](https://doi.org/10.1145/3491102.3517470)] [Code: N/A] [[Review](https://openreview.net/forum?id=7vdeyZVPHL)]
-   **Comlittee: Literature discovery with personal elected author committees** *CHI 2023, 2023.04* [[Paper](https://doi.org/10.1145/3544548.3581371)] [Code: N/A] [[Review](https://openreview.net/forum?id=NdSOavTTqdC)]
-   **An academic recommender system on large citation data based on clustering, graph modeling and deep learning** *Knowledge and Information Systems 2024, 2024.04* [[Paper](https://doi.org/10.1007/s10115-024-02094-7)] [Code: N/A] [Review: N/A]
-   **Paperweaver: Enriching topical paper alerts by contextualizing recommended papers with user-collected papers** *CHI 2024, 2024.05* [[Paper](https://doi.org/10.1145/3613904.3642196)] [Code: N/A] [Review: N/A]

##### Figure Understanding

-   **A Diagram is Worth a Dozen Images** *ECCV 2016, 2016.09* [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_15)] [Code: N/A] [Review: N/A]
-   **FigureQA: An Annotated Figure Dataset for Visual Reasoning** *arXiv, 2017.10* [[Paper](https://arxiv.org/abs/1710.07300)] [[Code](https://github.com/Maluuba/FigureQA)] [Review: N/A]
-   **A simple neural network module for relational reasoning** *NeurIPS 2017, 2017.12* [[Paper](https://papers.nips.cc/paper_files/paper/2017/hash/e6acf4b0f69f6f6e60e9a815938aa1ff-Abstract.html)] [Code: N/A] [[Review](https://papers.nips.cc/paper_files/paper/2017/file/e6acf4b0f69f6f6e60e9a815938aa1ff-Reviews.html)]
-   **ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning** *ACL Findings 2022, 2022.05* [[Paper](https://aclanthology.org/2022.findings-acl.177/)] [[Code](https://github.com/vis-nlp/chartqa)] [Review: N/A]
-   **ChartSumm: A Comprehensive Benchmark for Automatic Chart Summarization of Long and Short Summaries** *arXiv, 2023.04* [[Paper](https://arxiv.org/abs/2304.13620)] [[Code](https://github.com/pranonrahman/ChartSumm)] [Review: N/A]
-   **SPIQA: A Dataset for Multimodal Question Answering on Scientific Papers** *arXiv, 2024.07* [[Paper](https://arxiv.org/pdf/2407.09413)] [[Code](https://github.com/google/spiqa)] [Review: N/A]
-   **Multimodal ArXiv: A Dataset for Improving Scientific Comprehension of Large Vision-Language Models** *ACL 2024, 2024.08* [[Paper](https://aclanthology.org/2024.acl-long.775/)] [[Code](https://huggingface.co/MMInstruction/Qwen-VL-ArXivQA)] [Review: N/A] [[Project](https://mm-arxiv.github.io/)]
-   **SciMMIR: Benchmarking Scientific Multi-modal Information Retrieval** *ACL Findings 2024, 2024.08* [[Paper](https://aclanthology.org/2024.findings-acl.746/)] [[Code](https://github.com/Wusiwei0410/SciMMIR)] [Review: N/A] [Project: N/A]
-   **CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs** *NeurIPS 2024, 2024.12* [[Paper](https://openreview.net/pdf?id=cy8mq7QYae)] [[Code](https://github.com/princeton-nlp/CharXiv)] [Review: N/A]
-   **ChartAdapter: Large Vision-Language Model for Chart Summarization** *arXiv, 2024.12* [[Paper](https://arxiv.org/abs/2412.20715)] [Code: N/A] [Review: N/A]

##### Table Understanding

-   **ToTTo: A Controlled Table-To-Text Generation Dataset** *EMNLP 2020, 2020.11* [[Paper](https://aclanthology.org/2020.emnlp-main.89/)] [[Code](https://github.com/google-research-datasets/ToTTo)] [Review: N/A]
-   **Towards Table-to-Text Generation with Numerical Reasoning** *ACL 2021, 2021.08* [[Paper](https://aclanthology.org/2021.acl-long.115/)] [[Code](https://github.com/titech-nlp/numeric-nlg)] [Review: N/A]
-   **Structure-Aware Pre-Training for Table-to-Text Generation** *ACL Findings 2021, 2021.08* [[Paper](https://aclanthology.org/2021.findings-acl.200/)] [Code: Repo Only] [Review: N/A]
-   **SciGen: a Dataset for Reasoning-Aware Text Generation from Scientific Tables** *EMNLP Findings 2021, 2021.10* [[Paper](https://openreview.net/forum?id=Jul-uX7EV_I)] [[Code](https://github.com/UKPLab/SciGen)] [Review: N/A]
-   **Robust (Controlled) Table-to-Text Generation with Structure-Aware Equivariance Learning** *NAACL 2022, 2022.07* [[Paper](https://aclanthology.org/2022.naacl-main.371/)] [[Code](https://github.com/luka-group/Lattice)] [Review: N/A]
-   **SciXGen: A Scientific Paper Dataset for Context-Aware Text Generation** *EMNLP Findings 2021, 2022.11* [[Paper](https://aclanthology.org/2021.findings-emnlp.128/)] [Code: N/A] [Review: N/A]
-   **Few-shot Table-to-text Generation with Prefix-Controlled Generator** *COLING 2022, 2022.11* [[Paper](https://aclanthology.org/2022.coling-1.565/)] [Code: N/A] [Review: N/A]
-   **Table-To-Text generation and pre-training with TabT5** *EMNLP Findings 2022, 2022.12* [[Paper](https://aclanthology.org/2022.findings-emnlp.503/)] [Code: N/A] [Review: N/A]
-   **LoFT: Enhancing Faithfulness and Diversity for Table-to-Text Generation via Logic Form Control** *EACL 2023, 2023.05* [[Paper](https://aclanthology.org/2023.eacl-main.40/)] [[Code](https://github.com/Yale-LILY/LoFT)] [Review: N/A]
-   **SORTIE: Dependency-Aware Symbolic Reasoning for Logical Data-to-text Generation** *ACL Findings 2023, 2023.07* [[Paper](https://aclanthology.org/2023.findings-acl.715/)] [[Code](https://github.com/wenhuchen/LogicNLG)] [Review: N/A]
-   **Arithmetic-Based Pretraining Improving Numeracy of Pretrained Language Models** *STARSEM 2023, 2023.07* [[Paper](https://aclanthology.org/2023.starsem-1.42/)] [[Code](https://github.com/UKPLab/starsem2023-arithmetic-based-pretraining)] [Review: N/A]
-   **Structure-aware Table-to-Text Generation with Prefix-tuning** *CIKM 2023, 2023.11* [[Paper](https://dlnext.acm.org/doi/10.1145/3622896.3622919)] [Code: N/A] [Review: N/A]
-   **Unifying Structured Data as Graph for Data-to-Text Pre-Training** *TACL 2024, 2024.01* [[Paper](https://aclanthology.org/2024.tacl-1.12/)] [[Code](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/unid2t)] [Review: N/A]
-   **Table-to-Text Using Pre-trained Large Language Model and LoRA** *IEEE Access 2024, 2024.07* [[Paper](https://ieeexplore.ieee.org/document/10707812)] [Code: N/A] [Review: N/A]
-   **Integrating Table Representations into Large Language Models for Improved Scholarly Document Comprehension** *SDP @ ACL 2024, 2024.08* [[Paper](https://aclanthology.org/2024.sdp-1.28/)] [[Code](https://github.com/buseskorkmaz/Integrating-Table-Representations-into-LLMs)] [Review: N/A]

#### Knowledge Utilization

-   **Multi-document scientific summarization from a knowledge graph-centric view** *COLING 2022, 2022.10* [[Paper](https://aclanthology.org/2022.coling-1.543)] [[Code](https://github.com/muguruzawang/kgsum)] [Review: N/A]
-   **Bio-sieve: exploring instruction tuning large language models for systematic review automation** *arXiv, 2023.08* [[Paper](https://doi.org/10.48550/arXiv.2308.06610)] [[Code](https://github.com/ambroser53/Bio-SIEVE)] [Review: N/A]
-   **Hierarchical catalogue generation for literature review: a benchmark** *EMNLP Findings 2023, 2023.12* [[Paper](https://doi.org/10.18653/v1/2023.findings-emnlp.453)] [Code: N/A] [Review: N/A]
-   **Litllm: A toolkit for scientific literature review** *arXiv, 2024.02* [[Paper](https://doi.org/10.48550/arXiv.2402.01788)] [[Code](https://github.com/LitLLM/litllms-for-literature-review-tmlr)] [Review: N/A]
-   **Assisting in writing wikipedia-like articles from scratch with large language models** *NAACL 2024, 2024.06* [[Paper](https://doi.org/10.18653/v1/2024.naacl-long.347)] [[Code](https://github.com/stanford-oval/storm)] [Review: N/A]
-   **Chime: Llm-assisted hierarchical organization of scientific studies for literature review support** *ACL Findings 2024, 2024.08* [[Paper](https://doi.org/10.18653/v1/2024.findings-acl.8)] [[Code](https://github.com/allenai/chime)] [Review: N/A]
-   **Automating research synthesis with domain-specific large language model fine-tuning** *arXiv, 2024.04* [[Paper](https://arxiv.org/abs/2404.08680)] [[Code](https://github.com/peterjhwang/slr-helper)] [Review: N/A]
-   **Instruct Large Language Models to Generate Scientific Literature Survey Step by Step** *NLPCC 2024, 2024.08* [[Paper](https://arxiv.org/pdf/2408.07884)] [Code: N/A] [Review: N/A]
-   **Language agents achieve superhuman synthesis of scientific knowledge** *arXiv, 2024.09* [[Paper](https://doi.org/10.48550/arXiv.2409.13740)] [Code: N/A] [Review: N/A]
-   **Hireview: Hierarchical taxonomy-driven automatic literature review generation** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.03761)] [Code: N/A] [Review: N/A]
-   **Knowledge Navigator: LLM-guided Browsing Framework for Exploratory Search in Scientific Literature** *EMNLP Findings 2024, 2024.11* [[Paper](https://aclanthology.org/2024.findings-emnlp.516)] [[Code](https://github.com/katzurik/Knowledge_Navigator)] [Review: N/A]
-   **Autosurvey: Large language models can automatically write surveys** *NeurIPS 2024, 2024.12* [[Paper](http://papers.nips.cc/paper_files/paper/2024/hash/d07a9fc7da2e2ec0574c38d5f504d105-Abstract-Conference.html)] [Code: N/A] [Review: N/A]
-   **SurveyX: Academic Survey Automation via Large Language Models** *arXiv, 2025.02* [[Paper](https://arxiv.org/abs/2502.14776)] [Code: N/A] [Review: N/A]
-   **LimTopic: LLM-based Topic Modeling and Text Summarization for Analyzing Scientific Articles limitations** *JCDL 2024, 2025.03* [[Paper](https://www.arxiv.org/abs/2503.10658)] [[Code](https://github.com/IbrahimAlAzhar/LimTopic/tree/master)] [Review: N/A]
-   **What's In Your Field? Mapping Scientific Research with Knowledge Graphs and Large Language Models** *arXiv, 2025.03* [[Paper](https://arxiv.org/abs/2503.09894)] [[Code](https://github.com/chiral-carbon/kg-for-science)] [Review: N/A]

### Idea Generation _(Contributor: minjun)_

#### Generation Strategy

-   **Literature based discovery: models, methods, and trends** *Journal of biomedical informatics 2017, 2017.10* [[Paper](https://doi.org/10.1016/j.jbi.2017.08.011)] [Code: N/A] [Review: N/A]
-   **Predicting the Future of AI with AI: High-quality link prediction in an exponentially growing knowledge network** *arXiv, 2022.10* [[Paper](https://doi.org/10.48550/arXiv.2210.00881)] [[Code](https://github.com/artificial-scientist-lab/FutureOfAIviaAI)] [Review: N/A]
-   **LLM and Simulation as Bilevel Optimizers: A New Paradigm to Advance Physical Scientific Discovery** *ICML 2024, 2024.05* [[Paper](https://openreview.net/forum?id=hz8cFsdz7P)] [Code: N/A] [Review: N/A]
-   **Chain of ideas: Revolutionizing research via novel idea development with llm agents** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.13185)] [[Code](https://github.com/DAMO-NLP-SG/CoI-Agent)] [Review: N/A]
-   **Nova: An iterative planning and search approach to enhance novelty and diversity of llm generated ideas** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.14255)] [Code: N/A] [Review: N/A]

#### Generation Method

-   **Ideas are dimes a dozen: Large language models for idea generation in innovation** *The Wharton School Research Paper Forthcoming 2023, 2023.07* [[Paper](https://christophegirard.com/wp-content/uploads/2023/09/Etude-creation-idees-comparative-ChatGPT-vs-etudiants.pdf)] [Code: N/A] [Review: N/A]
-   **Monte Carlo Thought Search: Large Language Model Querying for Complex Scientific Reasoning in Catalyst Design** *arXiv, 2023.10* [[Paper](https://arxiv.org/abs/2310.14420)] [[Code](https://github.com/pnnl/chemreasoner)] [Review: N/A]
-   **Large Language Models are Zero Shot Hypothesis Proposers** *arXiv, 2023.11* [[Paper](https://arxiv.org/abs/2311.05965)] [Code: N/A] [Review: N/A]
-   **Mathematical discoveries from program search with large language models** *Nature 2023, 2023.12* [[Paper](https://www.nature.com/articles/s41586-023-06924-6)] [[Code](https://github.com/google-deepmind/funsearch)] [Review: N/A]
-   **ChemReasoner: Heuristic Search over a Large Language Model's Knowledge Space using Quantum-Chemical Feedback** *arXiv, 2024.02* [[Paper](https://arxiv.org/abs/2402.10980)] [[Code](https://github.com/pnnl/chemreasoner)] [Review: N/A]
-   **Acceleron: A tool to accelerate research ideation** *arXiv, 2024.03* [[Paper](https://doi.org/10.48550/arXiv.2403.04382)] [Code: N/A] [Review: N/A]
-   **Hypothesis generation with large language models** *arXiv, 2024.04* [[Paper](https://doi.org/10.48550/arXiv.2404.04326)] [[Code](https://github.com/ChicagoHAI/hypothesis-generation)] [Review: N/A]
-   **Researchagent: Iterative research idea generation over scientific literature with large language models** *arXiv, 2024.04* [[Paper](https://doi.org/10.48550/arXiv.2404.07738)] [[Code](https://github.com/JinheonBaek/ResearchAgent)] [Review: N/A]
-   **Data-driven discovery with large generative models** *ICML 2024, 2024.05* [[Paper](https://openreview.net/forum?id=5SpjhZNXtt)] [Code: N/A] [Review: N/A]
-   **Generation and human-expert evaluation of interesting research ideas using knowledge graphs and large language models** *arXiv, 2024.05* [[Paper](https://doi.org/10.48550/arXiv.2405.17044)] [[Code](https://github.com/artificial-scientist-lab/SciMuse)] [Review: N/A]
-   **Large Language Models as Biomedical Hypothesis Generators: A Comprehensive Evaluation** *ICML Workshop 2024, 2024.07* [[Paper](https://openreview.net/forum?id=q36rpGlG9X#discussion)] [[Code](https://github.com/TsinghuaC3I/LLM4BioHypoGen/)] [Review: N/A]
-   **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** *arXiv, 2024.08* [[Paper](https://doi.org/10.48550/arXiv.2408.06292)] [[Code](https://github.com/SakanaAI/AI-Scientist)] [Review: N/A]
-   **SciMON: Scientific Inspiration Machines Optimized for Novelty** *ACL 2024, 2024.08* [[Paper](https://doi.org/10.18653/v1/2024.acl-long.18)] [[Code](https://github.com/EagleW/Scientific-Inspiration-Machines-Optimized-for-Novelty)] [Review: N/A]
-   **Large language models for automated open-domain scientific hypotheses discovery** *ACL Findings 2024, 2024.08* [[Paper](https://doi.org/10.18653/v1/2024.findings-acl.804)] [[Code](https://github.com/SenticNet/MOOSE)] [Review: N/A]
-   **Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers** *arXiv, 2024.09* [[Paper](https://doi.org/10.48550/arXiv.2409.04109)] [[Code](https://github.com/NoviScl/AI-Researcher)] [Review: N/A]
-   **Accelerating scientific discovery with generative knowledge extraction, graph-based representation, and multimodal intelligent graph reasoning** *Machine Learning: Science and Technology 2024, 2024.09* [[Paper](https://doi.org/10.1088/2632-2153/ad7228)] [[Code](https://github.com/lamm-mit/GraphReasoning)] [Review: N/A]
-   **Can Large Language Models Unlock Novel Scientific Research Ideas?** *arXiv, 2024.09* [[Paper](https://doi.org/10.48550/arXiv.2409.06185)] [[Code](https://github.com/sandeep82945/Future-Idea-Generation)] [Review: N/A]
-   **Sciagents: Automating scientific discovery through multi-agent intelligent graph reasoning** *arXiv, 2024.09* [[Paper](https://doi.org/10.48550/arXiv.2409.05556)] [[Code](https://github.com/lamm-mit/SciAgentsDiscovery)] [Review: N/A]
-   **Scideator: Human-LLM Scientific Idea Generation Grounded in Research-Paper Facet Recombination** *arXiv, 2024.09* [[Paper](https://arxiv.org/abs/2409.14634)] [Code: N/A] [Review: N/A]
-   **IdeaSynth: Iterative Research Idea Development Through Evolving and Composing Idea Facets with Literature-Grounded Feedback** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.04025)] [Code: N/A] [Review: N/A]
-   **MOOSE-Chem: Large Language Models for Rediscovering Unseen Chemistry Scientific Hypotheses** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.07076)] [[Code](https://github.com/ZonglinY/MOOSE-Chem)] [Review: N/A]
-   **Two heads are better than one: A multi-agent system has the potential to improve scientific idea generation** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.09403)] [[Code](https://github.com/open-sciencelab/Virtual-Scientists)] [Review: N/A]
-   **Literature meets data: A synergistic approach to hypothesis generation** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.17309)] [[Code](https://github.com/ChicagoHAI/hypothesis-generation)] [Review: N/A]
-   **SciPIP: An LLM-based Scientific Paper Idea Proposer** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.23166)] [Code: N/A] [Review: N/A]
-   **Aigs: Generating Science from Ai-Powered Automated Falsification** *arXiv, 2024.11* [[Paper](https://arxiv.org/pdf/2411.11910)] [Code: N/A] [Review: N/A]
-   **Exploring Scientific Hypothesis Generation with Mamba** *NLP4Science @ EMNLP 2024, 2024.11* [[Paper](https://aclanthology.org/2024.nlp4science-1.17/)] [[Code](https://github.com/fglx-c/Exploring-Scientific-Hypothesis-Generation-with-Mamba)] [Review: N/A]
-   **Improving Scientific Hypothesis Generation with Knowledge Grounded Large Language Models** *arXiv, 2024.11* [[Paper](https://arxiv.org/abs/2411.02382)] [[Code](https://anonymous.4open.science/r/KG-CoI-C203/README.md)] [Review: N/A]
-   **DiscoveryWorld: A Virtual Environment for Developing and Evaluating Automated Scientific Discovery Agents** *NeurIPS 2024 Datasets and Benchmarks Track, 2024.12* [[Paper](http://papers.nips.cc/paper_files/paper/2024/hash/13836f251823945316ae067350a5c366-Abstract-Datasets_and_Benchmarks_Track.html)] [[Code](https://github.com/allenai/discoveryworld)] [Review: N/A]
-   **Learning to Generate Research Idea with Dynamic Control** *arXiv, 2024.12* [[Paper](https://doi.org/10.48550/arXiv.2412.14626)] [[Code: Repo Only](https://github.com/du-nlp-lab/Learn2Gen)] [Review: N/A]
-   **ResearchTown: Simulator of Human Research Community** *arXiv, 2024.12* [[Paper](https://doi.org/10.48550/arXiv.2412.17767)] [[Code](https://github.com/ulab-uiuc/research-town)] [Review: N/A]
-   **Dolphin: Closed-loop Open-ended Auto-research through Thinking, Practice, and Feedback** *arXiv, 2025.01* [[Paper](https://doi.org/10.48550/arXiv.2501.03916)] [[Code: Repo Only](https://github.com/Alpha-Innovator/Dolphin)] [Review: N/A]
-   **Towards an AI co-scientist** *arXiv, 2025.02* [[Paper](https://arxiv.org/pdf/2502.18864)] [Code: N/A] [Review: N/A]
-   **Graph of AI Ideas: Leveraging Knowledge Graphs and LLMs for AI Research Idea Generation** *arXiv, 2025.03* [[Paper](https://arxiv.org/abs/2503.08549)] [Code: N/A] [Review: N/A]

#### Internal Evaluation (Hypothesis Validation & Fact Checking)

-   **Generating fact checking explanations** *ACL 2020, 2020.07* [[Paper](https://doi.org/10.18653/v1/2020.acl-main.656)] [Code: N/A] [Review: N/A]
-   **Generative language modeling for automated theorem proving** *arXiv, 2020.09* [[Paper](https://arxiv.org/abs/2009.03393)] [Code: N/A] [Review: N/A]
-   **Fact or fiction: Verifying scientific claims** *EMNLP 2020, 2020.11* [[Paper](https://doi.org/10.18653/v1/2020.emnlp-main.609)] [[Code](https://github.com/allenai/scifact)] [Review: N/A]
-   **MultiVerS: Improving scientific claim verification with weak supervision and full-document context** *NAACL Findings 2022, 2022.07* [[Paper](https://doi.org/10.18653/v1/2022.findings-naacl.6)] [[Code](https://github.com/dwadden/longchecker)] [Review: N/A]
-   **Proofver: Natural logic theorem proving for fact verification** *TACL 2022, 2022.09* [[Paper](https://doi.org/10.1162/tacl_a_00503)] [[Code](https://github.com/krishnamrith12/ProoFVer)] [Review: N/A]
-   **HyperTree Proof Search for Neural Theorem Proving** *NeurIPS 2022, 2022.12* [[Paper](http://papers.nips.cc/paper_files/paper/2022/hash/a8901c5e85fb8e1823bbf0f755053672-Abstract-Conference.html)] [Code: N/A] [Review: N/A]
-   **Thor: Wielding hammers to integrate language models and automated theorem provers** *NeurIPS 2022, 2022.12* [[Paper](http://papers.nips.cc/paper_files/paper/2022/hash/377c25312668e48f2e531e2f2c422483-Abstract-Conference.html)] [Code: N/A] [Review: N/A]
-   **Missing counter-evidence renders NLP fact-checking unrealistic for misinformation** *EMNLP 2022, 2022.12* [[Paper](https://doi.org/10.18653/v1/2022.emnlp-main.397)] [[Code](https://github.com/UKPLab/emnlp2022-missing-counter-evidence)] [Review: N/A]
-   **The state of human-centered NLP technology for fact-checking** *Information processing & management 2023, 2023.01* [[Paper](https://arxiv.org/pdf/2301.03056)] [Code: N/A] [Review: N/A]
-   **Draft, sketch, and prove: Guiding formal theorem provers with informal proofs** *ICLR 2023, 2023.02* [[Paper](https://openreview.net/forum?id=SMa9EAovKMC)] [[Code](https://github.com/albertqjiang/draft_sketch_prove)] [Review: N/A]
-   **Decomposing the enigma: Subgoal-based demonstration learning for formal theorem proving** *arXiv, 2023.05* [[Paper](https://doi.org/10.48550/arXiv.2305.16366)] [[Code](https://github.com/HKUNLP/subgoal-theorem-prover)] [Review: N/A]
-   **aedFaCT: Scientific Fact-Checking Made Easier via Semi-Automatic Discovery of Relevant Expert Opinions** *arXiv, 2023.05* [[Paper](https://doi.org/10.48550/arXiv.2305.07796)] [[Code](https://github.com/altuncu/aedFaCT)] [Review: N/A]
-   **Fact-checking complex claims with program-guided reasoning** *ACL 2023, 2023.07* [[Paper](https://doi.org/10.18653/v1/2023.acl-long.386)] [[Code](https://github.com/mbzuai-nlp/ProgramFC)] [Review: N/A]
-   **FactKG: Fact verification via reasoning on knowledge graphs** *ACL 2023, 2023.07* [[Paper](https://doi.org/10.18653/v1/2023.acl-long.895)] [[Code](https://github.com/jiho283/FactKG)] [Review: N/A]
-   **Prompt to be Consistent is Better than Self-Consistent? Few-Shot and Zero-Shot Fact Verification with Pre-trained Language Models** *ACL Findings 2023, 2023.07* [[Paper](https://doi.org/10.18653/v1/2023.findings-acl.278)] [[Code](https://github.com/znhy1024/ProToCo)] [Review: N/A]
-   **Dt-solver: Automated theorem proving with dynamic-tree sampling guided by proof-level value function** *ACL 2023, 2023.07* [[Paper](https://doi.org/10.18653/v1/2023.acl-long.706)] [Code: N/A] [Review: N/A]
-   **An in-context learning agent for formal theorem-proving** *arXiv, 2023.10* [[Paper](https://doi.org/10.48550/arXiv.2310.04353)] [[Code](https://github.com/trishullab/copra)] [Review: N/A]
-   **Baldur: Whole-proof generation and repair with large language models** *ESEC/FSE 2023, 2023.11* [[Paper](https://doi.org/10.1145/3611643.3616243)] [Code: N/A] [Review: N/A]
-   **Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method** *IJCNLP 2023, 2023.11* [[Paper](https://doi.org/10.18653/v1/2023.ijcnlp-main.64)] [[Code](https://github.com/jadeCurl/HiSS)] [Review: N/A]
-   **Investigating zero-and few-shot generalization in fact verification** *IJCNLP 2023, 2023.11* [[Paper](https://doi.org/10.18653/v1/2023.ijcnlp-main.34)] [[Code](https://github.com/teacherpeterpan/Fact-Checking-Generalization)] [Review: N/A]
-   **Characterizing and Verifying Scientific Claims: Qualitative Causal Structure is All You Need** *EMNLP 2023, 2023.12* [[Paper](https://doi.org/10.18653/v1/2023.emnlp-main.828)] [[Code](https://github.com/VulnDetector/VerQCS)] [Review: N/A]
-   **Mustard: Mastering uniform synthesis of theorem and proof data** *ICLR 2024, 2024.01* [[Paper](https://openreview.net/forum?id=8xliOUg9EW)] [[Code](https://github.com/Eleanor-H/MUSTARD)] [Review: N/A]
-   **Unsupervised Pretraining for Fact Verification by Language Model Distillation** *ICLR 2024, 2024.01* [[Paper](https://openreview.net/forum?id=1mjsP8RYAw)] [[Code](https://github.com/AdrianBZG/SFAVEL)] [Review: N/A]
-   **Lego-prover: Neural theorem proving with growing libraries** *ICLR 2024, 2024.01* [[Paper](https://openreview.net/forum?id=3f5PALef5B)] [[Code](https://github.com/wiio12/LEGO-Prover)] [Review: N/A]
-   **Can Large Language Models Detect Misinformation in Scientific News Reporting?** *arXiv, 2024.02* [[Paper](https://doi.org/10.48550/arXiv.2402.14268)] [Code: N/A] [Review: N/A]
-   **What Makes Medical Claims (Un)Verifiable? Analyzing Entity and Relation Properties for Fact Verification** *EACL 2024, 2024.03* [[Paper](https://aclanthology.org/2024.eacl-long.124)] [Code: N/A] [Review: N/A]
-   **Comparing knowledge sources for open-domain scientific claim verification** *EACL 2024, 2024.03* [[Paper](https://aclanthology.org/2024.eacl-long.128)] [[Code](https://www.github.com/jvladika/comparing-knowledge-sources)] [Review: N/A]
-   **Towards large language models as copilots for theorem proving in lean** *arXiv, 2024.04* [[Paper](https://doi.org/10.48550/arXiv.2404.12534)] [[Code](https://github.com/lean-dojo/LeanCopilot)] [Review: N/A]
-   **Deepseek-prover: Advancing theorem proving in llms through large-scale synthetic data** *arXiv, 2024.05* [[Paper](https://doi.org/10.48550/arXiv.2405.14333)] [Code: N/A] [Review: N/A]
-   **MAGIC: Multi-Argument Generation with Self-Refinement for Domain Generalization in Automatic Fact-Checking** *LREC-COLING 2024, 2024.05* [[Paper](https://aclanthology.org/2024.lrec-main.951)] [[Code](https://github.com/NYCU-NLP-Lab/XClaimCheck)] [Review: N/A]
-   **Improving health question answering with reliable and time-aware evidence retrieval** *NAACL Findings 2024, 2024.06* [[Paper](https://doi.org/10.18653/v1/2024.findings-naacl.295)] [[Code](https://github.com/jvladika/Improving-Health-QA)] [Review: N/A]
-   **Lean-star: Learning to interleave thinking and proving** *arXiv, 2024.07* [[Paper](https://doi.org/10.48550/arXiv.2407.10040)] [[Code](https://github.com/Lagooon/LeanSTaR)] [Review: N/A]
-   **Grounding fallacies misrepresenting scientific publications in evidence** *arXiv, 2024.08* [[Paper](https://doi.org/10.48550/arXiv.2408.12812)] [[Code](https://github.com/UKPLab/naacl2025-missciplus)] [Review: N/A]
-   **Understanding Fine-grained Distortions in Reports of Scientific Findings** *ACL Findings 2024, 2024.08* [[Paper](https://doi.org/10.18653/v1/2024.findings-acl.369)] [Code: N/A] [Review: N/A] [[Project](https://www.uni-bamberg.de/en/nlproc/resources/sciencecommdistortion/)]
-   **Enhancing natural language inference performance with knowledge graph for COVID-19 automated fact-checking in Indonesian language** *arXiv, 2024.09* [[Paper](https://doi.org/10.48550/arXiv.2409.00061)] [Code: N/A] [Review: N/A]
-   **Augmenting the Veracity and Explanations of Complex Fact Checking via Iterative Self-Revision with LLMs** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.15135)] [Code: N/A] [Review: N/A]
-   **ClaimVer: Explainable claim-level verification and evidence attribution of text through knowledge graphs** *EMNLP Findings 2024, 2024.11* [[Paper](https://aclanthology.org/2024.findings-emnlp.795)] [Code: N/A] [Review: N/A]
-   **Proving theorems recursively** *NeurIPS 2024, 2024.12* [[Paper](http://papers.nips.cc/paper_files/paper/2024/hash/9de7a49945898da86e062e7029baa284-Abstract-Conference.html)] [Code: N/A] [Review: N/A]
-   **Automating Mathematical Proof Generation Using Large Language Model Agents and Knowledge Graphs** *arXiv, 2025.03* [[Paper](https://arxiv.org/abs/2503.11657)] [Code: N/A] [Review: N/A]
-   **Optimizing Decomposition for Optimal Claim Verification** *arXiv, 2025.03* [[Paper](https://arxiv.org/abs/2503.15354)] [[Code](https://github.com/yining610/dynamic-decomposition)] [Review: N/A]

### Experiment Execution _(Contributor: yixuan)_

#### Experiment Design

-   **HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face** *arXiv, 2023.03* [[Paper](https://arxiv.org/abs/2303.17580)] [[Code](https://github.com/microsoft/JARVIS)] [Review: N/A]
-   **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** *arXiv, 2023.08* [[Paper](https://arxiv.org/abs/2308.08155)] [[Code](https://github.com/microsoft/autogen)] [Review: N/A]
-   **Autonomous chemical research with large language models** *Nature 2023, 2023.12* [[Paper](https://www.nature.com/articles/s41586-023-06792-0)] [[Code](https://github.com/gomesgroup/coscientist)] [Review: N/A]
-   **Navigating Complexity: Orchestrated Problem Solving with Multi-Agent LLMs** *arXiv, 2024.02* [[Paper](https://arxiv.org/abs/2402.16713)] [Code: N/A] [Review: N/A]
-   **CRISPR-GPT: An LLM Agent for Automated Design of Gene-Editing Experiments** *arXiv, 2024.04* [[Paper](https://arxiv.org/abs/2404.18021)] [Code: N/A] [Review: N/A]
-   **Simulating Expert Discussions with Multi-agent for Enhanced Scientific Problem Solving** *SDP @ ACL 2024, 2024.08* [[Paper](https://aclanthology.org/2024.sdp-1.23/)] [Code: N/A] [Review: N/A]
-   **An automatic end-to-end chemical synthesis development platform powered by large language models** *Nature Communications 2024, 2024.11* [[Paper](https://www.nature.com/articles/s41467-024-54457-x)] [[Code](https://github.com/Ruan-Yixiang/LLM-RDF)] [Review: N/A]

#### Experiment Conduction

-   **Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences** *PNAS 2020, 2020.12* [[Paper](https://www.pnas.org/doi/full/10.1073/pnas.2016239118)] [[Code](https://github.com/facebookresearch/esm)] [Review: N/A]
-   **Controllable Protein Design with Language Models** *arXiv, 2022.01* [[Paper](https://arxiv.org/abs/2201.07338)] [Code: N/A] [Review: N/A]
-   **Evolutionary-scale prediction of atomic-level protein structure with a language model** *Science 2023, 2023.03* [[Paper](https://www.science.org/doi/10.1126/science.ade2574)] [Code: N/A] [Review: N/A]
-   **Bayesian Optimization of Catalysts With In-context Learning** *arXiv, 2023.04* [[Paper](https://arxiv.org/abs/2304.05341)] [[Code](https://github.com/ur-whitelab/BO-LIFT)] [Review: N/A]
-   **Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering** *arXiv, 2023.05* [[Paper](https://arxiv.org/abs/2305.03403)] [[Code](https://github.com/noahho/CAAFE)] [Review: N/A]
-   **ChatGPT-powered Conversational Drug Editing Using Retrieval and Domain Feedback** *arXiv, 2023.05* [[Paper](https://arxiv.org/abs/2305.18090)] [[Code](https://github.com/chao1224/ChatDrug)] [Review: N/A]
-   **Training Socially Aligned Language Models on Simulated Social Interactions** *arXiv, 2023.05* [[Paper](https://arxiv.org/abs/2305.16960)] [Code: N/A] [Review: N/A]
-   **Are you in a Masquerade? Exploring the Behavior and Impact of Large Language Model Driven Social Bots in Online Social Networks** *arXiv, 2023.07* [[Paper](https://arxiv.org/abs/2307.10337)] [[Code](https://github.com/Litsay/Masquerade-23)] [Review: N/A]
-   **AutoML-GPT: Large Language Model for AutoML** *arXiv, 2023.09* [[Paper](https://arxiv.org/abs/2309.01125)] [Code: N/A] [Review: N/A]
-   **Data-Juicer: A One-Stop Data Processing System for Large Language Models** *arXiv, 2023.09* [[Paper](https://arxiv.org/abs/2309.02033)] [[Code](https://github.com/modelscope/data-juicer)] [Review: N/A]
-   **Monte Carlo Thought Search: Large Language Model Querying for Complex Scientific Reasoning in Catalyst Design** *arXiv, 2023.10* [[Paper](https://arxiv.org/abs/2310.14420)] [[Code](https://github.com/pnnl/chemreasoner)] [Review: N/A]
-   **Jellyfish: A Large Language Model for Data Preprocessing** *arXiv, 2023.12* [[Paper](https://arxiv.org/abs/2312.01678)] [[Code](https://huggingface.co/NECOUDBFM/Jellyfish-8B)] [Review: N/A]
-   **Autonomous chemical research with large language models** *Nature 2023, 2023.12* [[Paper](https://www.nature.com/articles/s41586-023-06792-0)] [[Code](https://github.com/gomesgroup/coscientist)] [Review: N/A]
-   **DrugAssist: A Large Language Model for Molecule Optimization** *arXiv, 2024.01* [[Paper](https://arxiv.org/abs/2401.10334)] [[Code](https://github.com/blazerye/DrugAssist)] [Review: N/A]
-   **ChemReasoner: Heuristic Search over a Large Language Model's Knowledge Space using Quantum-Chemical Feedback** *arXiv, 2024.02* [[Paper](https://arxiv.org/abs/2402.10980)] [[Code](https://github.com/pnnl/chemreasoner)] [Review: N/A]
-   **Augmenting large language models with chemistry tools** *Nature Machine Intelligence 2024, 2024.05* [[Paper](https://www.nature.com/articles/s42256-024-00832-8)] [[Code](https://github.com/ur-whitelab/chemcrow-public)] [Review: N/A]
-   **Efficient Evolutionary Search Over Chemical Space with Large Language Models** *arXiv, 2024.06* [[Paper](https://arxiv.org/abs/2406.16976)] [[Code](https://github.com/zoom-wang112358/MOLLEO)] [Review: N/A]
-   **Tree Search for Language Model Agents** *arXiv, 2024.07* [[Paper](https://arxiv.org/abs/2407.01476)] [[Code](https://github.com/kohjingyu/search-agents)] [Review: N/A]
-   **De novo generation of SARS-CoV-2 antibody CDRH3 with a pre-trained generative large language model** *Nature Communications 2024, 2024.08* [[Paper](https://www.nature.com/articles/s41467-024-50903-y)] [[Code](https://github.com/TencentAILabHealthcare/PALM)] [Review: N/A]
-   **DrugAgent: Automating AI-aided Drug Discovery Programming through LLM Multi-Agent Collaboration** *arXiv, 2024.11* [[Paper](https://arxiv.org/abs/2411.15692)] [[Code](https://anonymous.4open.science/r/drugagent-5C42/README.md)] [Review: N/A]
-   **PaperBench: Evaluating AI’s Ability to Replicate AI Research** *OpenAI Technical Report, 2025.03* [[Paper](https://cdn.openai.com/papers/22265bac-3191-44e5-b057-7aaacd8e90cd/paperbench.pdf)] [[Code](https://github.com/openai/preparedness)] [Review: N/A]

#### Internal Evaluation

-   **Automl-gpt: Automatic machine learning with gpt** *arXiv, 2023.05* [[Paper](https://doi.org/10.48550/arXiv.2305.02499)] [[Code](https://github.com/Significant-Gravitas/AutoGPT)] [Review: N/A]
-   **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** *arXiv, 2023.08* [[Paper](https://arxiv.org/abs/2308.08155)] [[Code](https://github.com/microsoft/autogen)] [Review: N/A]
-   **MentaLLaMA: Interpretable Mental Health Analysis on Social Media with Large Language Models** *arXiv, 2023.09* [[Paper](https://arxiv.org/abs/2309.13567)] [[Code](https://github.com/SteveKGYang/MentaLLaMA)] [Review: N/A]
-   **MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation** *arXiv, 2023.10* [[Paper](https://arxiv.org/abs/2310.03302)] [[Code](https://github.com/snap-stanford/MLAgentBench/)] [Review: N/A]
-   **MechAgents: Large language model multi-agent collaborations can solve mechanics problems, generate new data, and integrate knowledge** *arXiv, 2023.11* [[Paper](https://doi.org/10.48550/arXiv.2311.08166)] [[Code](https://github.com/lamm-mit/MechAgents)] [Review: N/A]
-   **An autonomous laboratory for the accelerated synthesis of novel materials** *Nature 2023, 2023.11* [[Paper](https://www.nature.com/articles/s41586-023-06734-w)] [[Code](https://github.com/mattmcdermott/novel-materials-screening)] [Review: N/A]
-   **Autonomous chemical research with large language models** *Nature 2023, 2023.12* [[Paper](https://doi.org/10.1038/s41586-023-06792-0)] [[Code](https://github.com/gomesgroup/coscientist)] [Review: N/A]
-   **LLM-in-the-loop: Leveraging Large Language Model for Thematic Analysis** *EMNLP Findings 2023, 2023.12* [[Paper](https://aclanthology.org/2023.findings-emnlp.669/)] [[Code](https://github.com/sjdai/LLM-thematic-analysis)] [Review: N/A]
-   **Training socially aligned language models on simulated social interactions** *ICLR 2024, 2024.01* [[Paper](https://openreview.net/forum?id=NddKiWtdUm)] [Code: N/A] [Review: N/A]
-   **SWE-bench: Can Language Models Resolve Real-world Github Issues?** *ICLR 2024, 2024.01* [[Paper](https://openreview.net/forum?id=VTF8yNQM66)] [[Code](https://github.com/swe-bench/SWE-bench)] [Review: N/A]
-   **Can Large Language Models Serve as Data Analysts? A Multi-Agent Assisted Approach for Qualitative Data Analysis** *arXiv, 2024.02* [[Paper](https://arxiv.org/abs/2402.01386)] [Code: N/A] [Review: N/A]
-   **Automated Statistical Model Discovery with Language Models** *arXiv, 2024.02* [[Paper](https://arxiv.org/abs/2402.17879)] [Code: N/A] [Review: N/A]
-   **Large language model agent for hyper-parameter optimization** *arXiv, 2024.02* [[Paper](https://doi.org/10.48550/arXiv.2402.01881)] [Code: N/A] [Review: N/A]
-   **Mlcopilot: Unleashing the power of large language models in solving machine learning tasks** *EACL 2024, 2024.03* [[Paper](https://aclanthology.org/2024.eacl-long.179)] [[Code](https://github.com/microsoft/CoML)] [Review: N/A]
-   **Researchagent: Iterative research idea generation over scientific literature with large language models** *arXiv, 2024.04* [[Paper](https://doi.org/10.48550/arXiv.2404.07738)] [[Code](https://github.com/JinheonBaek/ResearchAgent)] [Review: N/A]
-   **Automated social science: Language models as scientist and subjects** *arXiv, 2024.04* [[Paper](https://arxiv.org/abs/2404.11794)] [Code: N/A] [Review: N/A]
-   **Crispr-gpt: An llm agent for automated design of gene-editing experiments** *arXiv, 2024.04* [[Paper](https://doi.org/10.48550/arXiv.2404.18021)] [Code: N/A] [Review: N/A]
-   **Position: LLMs can't plan, but can help planning in LLM-modulo frameworks** *ICML 2024, 2024.05* [[Paper](https://openreview.net/forum?id=Th8JPEmH4z)] [Code: N/A] [Review: N/A]
-   **Augmenting large language models with chemistry tools** *Nature Machine Intelligence 2024, 2024.05* [[Paper](https://doi.org/10.1038/s42256-024-00832-8)] [[Code](https://github.com/ur-whitelab/chemcrow-public)] [Review: N/A]
-   **Opening a conversation on responsible environmental data science in the age of large language models** *Environmental Data Science 2024, 2024.05* [[Paper](https://doi.org/10.1017/eds.2024.12)] [Code: N/A] [Review: N/A]
-   **Meta-Designing Quantum Experiments with Language Models** *arXiv, 2024.06* [[Paper](https://doi.org/10.48550/arXiv.2406.02470)] [Code: N/A] [Review: N/A]
-   **Automatic benchmarking of large multimodal models via iterative experiment programming** *arXiv, 2024.06* [[Paper](https://arxiv.org/abs/2406.12321)] [[Code](https://github.com/altndrr/apex)] [Review: N/A]
-   **OpenHands: An Open Platform for AI Software Developers as Generalist Agents** *arXiv, 2024.07* [[Paper](https://arxiv.org/abs/2407.16741)] [[Code](https://github.com/All-Hands-AI/OpenHands)] [Review: N/A]
-   **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** *arXiv, 2024.08* [[Paper](https://doi.org/10.48550/arXiv.2408.06292)] [[Code](https://github.com/SakanaAI/AI-Scientist)] [Review: N/A]
-   **Mlr-copilot: Autonomous machine learning research based on large language models agents** *arXiv, 2024.08* [[Paper](https://doi.org/10.48550/arXiv.2408.14033)] [[Code](https://github.com/du-nlp-lab/MLR-Copilot)] [Review: N/A]
-   **DSBench: How Far Are Data Science Agents to Becoming Data Science Experts?** *arXiv, 2024.09* [[Paper](https://arxiv.org/abs/2409.07703)] [[Code](https://github.com/LiqiangJing/DSBench)] [Review: N/A]
-   **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML** *arXiv, 2024.10* [[Paper](https://arxiv.org/abs/2410.02958)] [Code: N/A] [Review: N/A]
-   **Chain of ideas: Revolutionizing research via novel idea development with llm agents** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.13185)] [[Code](https://github.com/DAMO-NLP-SG/CoI-Agent)] [Review: N/A]
-   **ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery** *arXiv, 2024.10* [[Paper](https://arxiv.org/abs/2410.05080)] [[Code](https://github.com/OSU-NLP-Group/ScienceAgentBench)] [Review: N/A]
-   **Agent-as-a-Judge: Evaluate Agents with Agents** *arXiv, 2024.10* [[Paper](https://arxiv.org/abs/2410.10934)] [[Code](https://github.com/metauto-ai/agent-as-a-judge)] [Review: N/A]
-   **SELA: Tree-Search Enhanced LLM Agents for Automated Machine Learning** *arXiv, 2024.10* [[Paper](https://arxiv.org/abs/2410.17238)] [[Code](https://github.com/geekan/MetaGPT)] [Review: N/A]
-   **AAAR-1.0: Assessing AI's Potential to Assist Research** *arXiv, 2024.10* [[Paper](https://doi.org/10.48550/arXiv.2410.22394)] [Code: N/A] [Review: N/A]
-   **An automatic end-to-end chemical synthesis development platform powered by large language models** *Nature communications 2024, 2024.11* [[Paper](https://doi.org/10.1038/s41467-024-54457-x)] [[Code](https://github.com/Ruan-Yixiang/LLM-RDF)] [Review: N/A]
-   **MatPilot: an LLM-enabled AI Materials Scientist under the Framework of Human-Machine Collaboration** *arXiv, 2024.11* [[Paper](https://doi.org/10.48550/arXiv.2411.08063)] [Code: N/A] [Review: N/A]
-   **AI agents in chemical research: GVIM – an intelligent research assistant system** *Digital Discovery 2025, 2025.01* [[Paper](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00398e)] [[Code](https://github.com/KangyongMa/GVIM)] [Review: N/A]
-   **Dolphin: Closed-loop Open-ended Auto-research through Thinking, Practice, and Feedback** *arXiv, 2025.01* [[Paper](https://doi.org/10.48550/arXiv.2501.03916)] [Code: N/A] [Review: N/A]
-   **Agent Laboratory: Using LLM Agents as Research Assistants** *arXiv, 2025.01* [[Paper](https://arxiv.org/abs/2501.04227)] [[Code](https://github.com/SamuelSchmidgall/AgentLaboratory)] [Review: N/A]
-   **SWE-bench Multimodal: Do AI Systems Generalize to Visual Software Domains?** *ICLR 2025 Submission, 2025.01* [[Paper](https://openreview.net/forum?id=riTiq3i21b)] [Code: N/A] [Review: N/A]
-   **AIDE: AI-Driven Exploration in the Space of Code** *arXiv, 2025.02* [[Paper](https://arxiv.org/abs/2502.13138)] [[Code](https://github.com/WecoAI/aideml)] [Review: N/A]
-   **MLGym: A New Framework and Benchmark for Advancing AI Research Agents** *arXiv, 2025.02* [[Paper](https://arxiv.org/abs/2502.14499)] [[Code](https://github.com/facebookresearch/MLGym)] [Review: N/A]

### Paper Writing _(Contributors: minjun, yixuan, qiujie)_

#### Main Writing & Summarization

-   **Towards Automated Related Work Summarization** *COLING 2010, 2010.08* [[Paper](https://aclanthology.org/C10-2049/)] [Code: N/A] [Review: N/A]
-   **Neural Related Work Summarization with a Joint Context-driven Attention Mechanism** *EMNLP 2018, 2018.10* [[Paper](https://aclanthology.org/D18-1204/)] [[Data](https://github.com/kuadmu/2018EMNLP)] [Review: N/A]
-   **PaperRobot: Incremental Draft Generation of Scientific Ideas** *ACL 2019, 2019.07* [[Paper](https://aclanthology.org/P19-1191/)] [[Code](https://github.com/EagleW/PaperRobot)] [Review: N/A]
-   **ScisummNet: A Large Annotated Corpus and Content-Impact Models for Scientific Paper Summarization with Citation Networks** *AAAI 2019, 2019.07* [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4727)] [Code: N/A] [Review: N/A]
-   **Automatic Generation of Citation Texts in Scholarly Papers: A Pilot Study** *ACL 2020, 2020.07* [[Paper](https://aclanthology.org/2020.acl-main.550/)] [[Code](https://github.com/XingXinyu96/citation_generation)] [Review: N/A]
-   **Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen** *ACL 2020, 2020.07* [[Paper](https://aclanthology.org/2020.acl-main.100/)] [Code: N/A] [Review: N/A] [[Project](https://srhthu.github.io/expertise-style-transfer/#disclaimer)]
-   **Automatic related work section generation: experiments in scientific document abstracting** *Scientometrics 2020, 2020.07* [[Paper](https://link.springer.com/article/10.1007/s11192-020-03630-2)] [[Code](https://github.com/AhmedAbuRaed/SPSeq2Seq)] [Review: N/A]
-   **Automatic Related Work Section Generation by Sentence Extraction and Reordering** *BIR @ ECIR 2021, 2021.??* [[Paper](https://ceur-ws.org/Vol-2871/paper8.pdf)] [Code: N/A] [Review: N/A]
-   **Automatic Title Generation for Text with Pre-trained Transformer Language Model** *IEEE Access 2021, 2021.01* [[Paper](https://ieeexplore.ieee.org/abstract/document/9364613)] [[Code](https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control)] [Review: N/A]
-   **AutoCite: Multi-Modal Representation Fusion for Contextual Citation Generation** *WSDM 2021, 2021.03* [[Paper](https://dl.acm.org/doi/10.1145/3437963.3441739)] [[Code](https://github.com/qqingwang/AutoCite)] [Review: N/A]
-   **BACO: A Background Knowledge- and Content-Based Framework for Citing Sentence Generation** *ACL 2021, 2021.08* [[Paper](https://aclanthology.org/2021.acl-long.116/)] [Code: N/A] [Review: N/A]
-   **Intent-Controllable Citation Text Generation** *Mathematics 2022, 2022.04* [[Paper](https://www.mdpi.com/2227-7390/10/10/1763)] [[Code](https://github.com/BradLin0819/Automatic-Citation-Text-Generation-with-Citation-Intent-Control)] [Review: N/A]
-   **Read, revise, repeat: A system demonstration for human-in-the-loop iterative text revision** *ACL Demo 2022, 2022.04* [[Paper](https://doi.org/10.48550/arXiv.2204.03685)] [[Code](https://github.com/vipulraheja/IteraTeR)] [Review: N/A]
-   **Generating Scientific Definitions with Controllable Complexity** *ACL 2022, 2022.05* [[Paper](https://aclanthology.org/2022.acl-long.569/)] [[Code](https://github.com/talaugust/definition-complexity)] [Review: N/A]
-   **Understanding Iterative Revision from Human-Written Text** *ACL 2022, 2022.05* [[Paper](https://aclanthology.org/2022.acl-long.250/)] [[Code](https://github.com/vipulraheja/iterater)] [Review: N/A]
-   **Disencite: Graph-based disentangled representation learning for context-specific citation generation** *AAAI 2022, 2022.06* [[Paper](https://doi.org/10.1609/aaai.v36i10.21397)] [Code: N/A] [Review: N/A]
-   **CORWA: A Citation-Oriented Related Work Annotation Dataset** *NAACL 2022, 2022.07* [[Paper](https://aclanthology.org/2022.naacl-main.397.pdf)] [[Code](https://github.com/jacklxc/CORWA)] [Review: N/A]
-   **Controllable citation sentence generation with language models** *arXiv, 2022.11* [[Paper](https://arxiv.org/abs/2211.07066)] [[Code](https://github.com/nianlonggu/LMCiteGen)] [Review: N/A]
-   **Improving Iterative Text Revision by Learning Where to Edit from Other Revision Tasks** *EMNLP 2022, 2022.12* [[Paper](https://aclanthology.org/2022.emnlp-main.678/)] [[Code](https://github.com/vipulraheja/iterater)] [Review: N/A]
-   **Making Science Simple: Corpora for the Lay Summarisation of Scientific Literature** *EMNLP 2022, 2022.12* [[Paper](https://aclanthology.org/2022.emnlp-main.724/)] [[Code](https://github.com/TGoldsack1/Corpora_for_Lay_Summarisation)] [Review: N/A]
-   **Can artificial intelligence help for scientific writing?** *Critical Care 2023, 2023.02* [[Paper](https://link.springer.com/article/10.1186/s13054-023-04380-2)] [Code: N/A] [Review: N/A]
-   **Text revision in scientific writing assistance: An overview** *arXiv, 2023.03* [[Paper](https://doi.org/10.48550/arXiv.2303.16726)] [Code: N/A] [Review: N/A]
-   **Using ChatGPT for language editing in scientific articles** *Gynecol Oncol Rep 2023, 2023.03* [[Paper](https://link.springer.com/content/pdf/10.1186/s40902-023-00381-x.pdf)] [Code: N/A] [Review: N/A]
-   **Good Practices for Scientific Article Writing with ChatGPT and Other Artificial Intelligence Language Models** *Medicina 2023, 2023.04* [[Paper](https://www.mdpi.com/2673-687X/3/2/9)] [Code: N/A] [Review: N/A]
-   **Comparing scientific abstracts generated by ChatGPT to real abstracts with detectors and blinded human reviewers** *npj Digital Medicine 2023, 2023.04* [[Paper](https://www.nature.com/articles/s41746-023-00819-6)] [Code: N/A] [Review: N/A]
-   **Decoding the End-to-end Writing Trajectory in Scholarly Manuscripts** *In2Writing @CHI 2023, 2023.04* [[Paper](https://arxiv.org/abs/2304.00121)] [[Code](https://github.com/minnesotanlp/reward-system)] [Review: N/A]
-   **Exploring the boundaries of reality: investigating the phenomenon of artificial intelligence hallucination in scientific writing through ChatGPT references** *Cureus 2023, 2023.04* [[Paper](https://pubmed.ncbi.nlm.nih.gov/37182055/)] [Code: N/A] [Review: N/A]
-   **The role of ChatGPT in scientific communication: writing better scientific review articles** *Ann Biomed Eng 2023, 2023.04* [[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10164801/)] [Code: N/A] [Review: N/A]
-   **SciLit: A platform for joint scientific literature discovery, summarization and citation generation** *ACL 2023 Demo, 2023.07* [[Paper](https://doi.org/10.18653/v1/2023.acl-demo.22)] [Code: N/A] [Review: N/A]
-   **Artificial intelligence in scientific writing: a friend or a foe?** *J Transl Med 2023, 2023.07* [[Paper](https://pubmed.ncbi.nlm.nih.gov/37142479/)] [Code: N/A] [Review: N/A]
-   **Transformers Go for the LOLs: Generating (Humourous) Titles from Scientific Abstracts End-to-End** *EVAL4NLP @ IJCNLP 2023, 2023.11* [[Paper](https://aclanthology.org/2023.eval4nlp-1.6/)] [[Code](https://github.com/cyr19/A2T)] [Review: N/A]
-   **Enabling large language models to generate text with citations** *EMNLP 2023, 2023.12* [[Paper](https://doi.org/10.18653/v1/2023.emnlp-main.398)] [[Code](https://github.com/princeton-nlp/ALCE)] [Review: N/A]
-   **Towards a unified framework for reference retrieval and related work generation** *EMNLP Findings 2023, 2023.12* [[Paper](https://doi.org/10.18653/v1/2023.findings-emnlp.385)] [Code: N/A] [Review: N/A]
-   **‘Don’t Get Too Technical with Me’: A Discourse Structure-Based Framework for Science Journalism** *EMNLP 2023, 2023.12* [[Paper](https://aclanthology.org/2023.emnlp-main.76.pdf)] [Code: N/A] [Review: N/A]
-   **Leveraging Large Language Models for Literature Review Tasks - A Case Study Using ChatGPT** *ICDM Workshops 2023, 2023.12* [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-48858-0_25)] [Code: N/A] [Review: N/A]
-   **Explaining Relationships Among Research Papers** *arXiv, 2024.02* [[Paper](https://arxiv.org/abs/2402.13426)] [Code: N/A] [Review: N/A]
-   **Shallow synthesis of knowledge in gpt-generated texts: A case study in automatic related work composition** *arXiv, 2024.02* [[Paper](https://doi.org/10.48550/arXiv.2402.12255)] [Code: N/A] [Review: N/A]
-   **LitLLM: A Toolkit for Scientific Literature Review** *arXiv, 2024.02* [[Paper](https://arxiv.org/abs/2402.01788)] [[Code](https://github.com/LitLLM/litllms-for-literature-review-tmlr)] [Review: N/A]
-   **Techniques for supercharging academic writing with generative AI** *Nature Biomedical Engineering 2024, 2024.03* [[Paper](https://www.nature.com/articles/s41551-024-01185-8)] [Code: N/A] [Review: N/A]
-   **Automating Research Synthesis with Domain-Specific Large Language Model Fine-Tuning** *arXiv, 2024.04* [[Paper](https://arxiv.org/abs/2404.08680)] [[Code](https://github.com/peterjhwang/slr-helper)] [Review: N/A]
-   **Mapping the Increasing Use of LLMs in Scientific Papers** *arXiv, 2024.04* [[Paper](https://arxiv.org/abs/2404.01268v1)] [Code: N/A] [Review: N/A]
-   **Autonomous LLM-Driven Research—from Data to Human-Verifiable Research Papers** *arXiv, 2024.04* [[Paper](https://doi.org/10.48550/arXiv.2404.17605)] [[Code](https://github.com/Technion-Kishony-lab/data-to-paper)] [Review: N/A]
-   **A publishing infrastructure for Artificial Intelligence (AI)-assisted academic authoring** *JAMIA 2024, 2024.06* [[Paper](https://doi.org/10.1093/jamia/ocae139)] [[Code](https://github.com/manubot/manubot-ai-editor)] [Review: N/A]
-   **Step-Back Profiling: Distilling User History for Personalized Scientific Writing** *arXiv, 2024.06* [[Paper](https://doi.org/10.48550/arXiv.2406.14275)] [[Code](https://github.com/gersteinlab/step-back-profiling)] [Review: N/A]
-   **AutoSurvey: Large Language Models Can Automatically Write Surveys** *arXiv, 2024.06* [[Paper](https://arxiv.org/abs/2406.10252)] [[Code](https://github.com/AutoSurveys/AutoSurvey)] [Review: N/A]
-   **The Ability of ChatGPT in Paraphrasing Texts and Reducing Plagiarism: A Descriptive Analysis** *Cureus 2024, 2024.07* [[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11250043/)] [Code: N/A] [Review: N/A]
-   **Pelican: Correcting Hallucination in Vision-LLMs via Claim Decomposition and Program of Thought Verification** *arXiv, 2024.07* [[Paper](https://arxiv.org/abs/2407.02352)] [Code: N/A] [Review: N/A]
-   **Reinforced Subject-Aware Graph Neural Network for Related Work Generation** *KSEM 2024, 2024.07* [[Paper](https://doi.org/10.1007/978-981-97-5492-2_16)] [Code: N/A] [Review: N/A]
-   **Reference hallucination score for medical artificial intelligence chatbots: development and usability study** *JMIR Medical Informatics 2024, 2024.07* [[Paper](https://medinform.jmir.org/2024/1/e54345)] [Code: N/A] [Review: N/A]
-   **LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs** *arXiv, 2024.08* [[Paper](https://arxiv.org/abs/2408.07055)] [[Code](https://github.com/THUDM/LongWriter)] [Review: N/A]
-   **Automated focused feedback generation for scientific writing assistance** *ACL Findings 2024, 2024.08* [[Paper](https://doi.org/10.18653/v1/2024.findings-acl.580)] [[Code](https://github.com/ericchamoun/FocusedFeedbackGeneration)] [Review: N/A]
-   **Instruct Large Language Models to Generate Scientific Literature Survey Step by Step** *NLPCC 2024, 2024.08* [[Paper](https://arxiv.org/pdf/2408.07884)] [Code: N/A] [Review: N/A]
-   **Aries: A corpus of scientific paper edits made in response to peer reviews** *ACL 2024, 2024.08* [[Paper](https://doi.org/10.18653/v1/2024.acl-long.377)] [[Code](https://github.com/allenai/aries)] [Review: N/A]
-   **Toward Structured Related Work Generation with Novelty Statements** *SDP @ ACL 2024, 2024.08* [[Paper](https://aclanthology.org/2024.sdp-1.5/)] [[Code](https://github.com/omron-sinicx/STRoGeNS)] [Review: N/A]
-   **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** *arXiv, 2024.08* [[Paper](https://doi.org/10.48550/arXiv.2408.06292)] [[Code](https://github.com/SakanaAI/AI-Scientist)] [Review: N/A]
-   **LongReward: Improving Long-context Large Language Models with AI Feedback** *arXiv, 2024.10* [[Paper](https://arxiv.org/abs/2410.21252)] [[Code](https://github.com/THUDM/LongReward)] [Review: N/A]
-   **Taxonomy Tree Generation from Citation Graph** *arXiv, 2024.10* [[Paper](https://arxiv.org/abs/2410.03761)] [Code: N/A] [Review: N/A] (*Note: Same paper link as Hireview*)
-   **CycleResearcher: Improving Automated Research via Automated Review** *arXiv, 2024.11* [[Paper](https://arxiv.org/abs/2411.00816)] [[Code](https://github.com/zhu-minjun/Researcher)] [Review: N/A]
-   **Cocoa: Co-Planning and Co-Execution with AI Agents** *arXiv, 2024.12* [[Paper](https://doi.org/10.48550/arXiv.2412.10999)] [Code: N/A] [Review: N/A]
-   **ResearchTown: Simulator of Human Research Community** *arXiv, 2024.12* [[Paper](https://doi.org/10.48550/arXiv.2412.17767)] [[Code](https://github.com/ulab-uiuc/research-town)] [Review: N/A]
-   **Agent Laboratory: Using LLM Agents as Research Assistants** *arXiv, 2025.01* [[Paper](https://arxiv.org/abs/2501.04227)] [[Code](https://github.com/SamuelSchmidgall/AgentLaboratory)] [Review: N/A]
-   **LongEval: A Comprehensive Analysis of Long-Text Generation Through a Plan-based Paradigm** *arXiv, 2025.02* [[Paper](https://arxiv.org/abs/2502.19103)] [[Code](https://github.com/Wusiwei0410/LongEval)] [Review: N/A]
-   **CorpusStudio: Surfacing Emergent Patterns in a Corpus of Prior Work while Writing** *ACM CHI 2025, 2025.03* [[Paper](https://arxiv.org/abs/2503.12436)] [Code: N/A] [Review: N/A]

#### Figure Generation

-   **Data2Vis: Automatic Generation of Data Visualizations Using Sequence-to-Sequence Recurrent Neural Networks** *IEEE TVCG 2019, 2019.06* [[Paper](https://ieeexplore.ieee.org/document/8744242)] [[Code](https://github.com/victordibia/data2vis)] [Review: N/A]
-   **SCICAP: Generating Captions for Scientific Figures** *EMNLP Findings 2021, 2021.11* [[Paper](https://aclanthology.org/2021.findings-emnlp.277/)] [[Code](https://github.com/tingyaohsu/scicap)] [Review: N/A]
-   **ADVISor: Automatic Visualization Answer for Natural-Language Question on Tabular Data** *IEEE PacificVis 2021, 2021.04* [[Paper](https://ieeexplore.ieee.org/document/9438784)] [Code: N/A] [Review: N/A]
-   **Sevi: Speech-to-Visualization through Neural Machine Translation** *IUI Companion 2022, 2022.06* [[Paper](https://dl.acm.org/doi/pdf/10.1145/3514221.3520150)] [[Code](https://github.com/Thanksyy/Sevi)] [Review: N/A]
-   **Chat2VIS: Generating Data Visualizations via Natural Language Using ChatGPT, Codex and GPT-3 Large Language Models** *IEEE PacificVis Workshops 2023, 2023.04* [[Paper](https://ieeexplore.ieee.org/abstract/document/10121440)] [Code: N/A] [Review: N/A]
-   **AutomaTikZ: Text-Guided Synthesis of Scientific Vector Graphics with TikZ** *ICLR 2024, 2024.03* [[Paper](https://openreview.net/forum?id=v3K5TVP8kZ)] [[Code](https://github.com/potamides/AutomaTikZ)] [Review: N/A]
-   **Plots Made Quickly: An Efficient Approach for Generating Visualizations from Natural Language Queries** *LREC-COLING 2024, 2024.05* [[Paper](https://aclanthology.org/2024.lrec-main.1119/)] [Code: N/A] [Review: N/A]
-   **DiagrammerGPT: Generating Open-Domain, Open-Platform Diagrams via LLM Planning** *ACL 2024, 2024.08* [[Paper](https://openreview.net/forum?id=NV8yRJRET1#discussion)] [[Code](https://github.com/aszala/DiagrammerGPT)] [Review: N/A]
-   **SciDoc2Diagrammer-MAF: Towards Generation of Scientific Diagrams from Documents guided by Multi-Aspect Feedback Refinement** *EMNLP Findings 2024, 2024.11* [[Paper](https://aclanthology.org/2024.findings-emnlp.780/)] [[Code](https://github.com/Ishani-Mondal/SciDoc2DiagramGeneration)] [Review: N/A]
-   **DeTikZify: Synthesizing Graphics Programs for Scientific Figures and Sketches with TikZ** *EMNLP 2024, 2024.11* [[Paper](https://openreview.net/forum?id=bcVLFQCOjc)] [[Code](https://github.com/potamides/DeTikZify)] [Review: N/A]
-   **ChartMimic: Evaluating LMM's Cross-Modal Reasoning Capability via Chart-to-Code Generation** *arXiv, 2025.03* [[Paper](https://openreview.net/forum?id=sGpCzsfd1K)] [[Code](https://github.com/ChartMimic/ChartMimic)] [Review: N/A]
-   **ScImage: How Good Are Multimodal Large Language Models at Scientific Text-to-Image Generation?** *arXiv, 2025.03* [[Paper](https://openreview.net/forum?id=ugyqNEOjoU)] [[Code](https://github.com/leixin-zhang/Scimage)] [Review: N/A]

#### Table Generation

-   **gTBLS: Generating Tables from Text by Conditional Question Answering** *arXiv, 2024.03* [[Paper](https://arxiv.org/abs/2403.14457)] [Code: N/A] [Review: N/A]
-   **OpenTE: Open-Structure Table Extraction From Text** *IEEE Access 2024, 2024.04* [[Paper](https://ieeexplore.ieee.org/document/10448427)] [Code: N/A] [Review: N/A]
-   **ArxivDIGESTables: Synthesizing Scientific Literature into Tables using Language Models** *EMNLP 2024, 2024.11* [[Paper](https://aclanthology.org/2024.emnlp-main.538/)] [[Code](https://github.com/bnewm0609/arxivDIGESTables)] [Review: N/A]
-   **Is This a Bad Table? A Closer Look at the Evaluation of Table Generation from Text** *EMNLP 2024, 2024.11* [[Paper](https://aclanthology.org/2024.emnlp-main.1239/)] [Code: N/A] [Review: N/A]
-   **LATTE: Improving Latex Recognition for Tables and Formulae with Iterative Refinement** *arXiv, 2025.02* [[Paper](https://arxiv.org/abs/2409.14201)] [Code: N/A] [Review: N/A]

#### Slides & Poster Generation

-   **SlidesGen: Automatic Generation of Presentation Slides for a Technical Paper Using Summarization** *FLAIRS 2009* [[Paper](https://aaai.org/papers/flairs-2009-22/)] [Code: N/A] [Review: N/A]
-   **PPSGen: learning to generate presentation slides for academic papers** *KDD 2013, 2013.08* [[Paper](https://dl.acm.org/doi/10.5555/2540128.2540430)] [Code: N/A] [Review: N/A]
-   **Learning to Generate Posters of Scientific Papers** *AAAI 2016, 2016.02* [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/10000)] [Code: N/A] [Review: N/A]
-   **Phrase-Based Presentation Slides Generation for Academic Papers** *AAAI 2017* [[Paper](https://aaai.org/papers/10481-aaai-31-2017/)] [Code: N/A] [Review: N/A]
-   **Towards Topic-Aware Slide Generation For Academic Papers With Unsupervised Mutual Learning** *AAAI 2021, 2021.05* [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17564)] [[Code](https://github.com/daviddwlee84/TopicAwarePaperSlideGeneration?tab=readme-ov-file)] [Review: N/A]
-   **D2S: Document-to-Slide Generation Via Query-Based Text Summarization** *NAACL 2021, 2021.06* [[Paper](https://aclanthology.org/2021.naacl-main.111/)] [[Code](https://github.com/IBM/document2slides)] [Review: N/A]
-   **DOC2PPT: Automatic Presentation Slides Generation from Scientific Documents** *AAAI 2022, 2022.06* [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19943)] [Code: N/A] [Review: N/A]
-   **PosterBot: A System for Generating Posters of Scientific Papers with Neural Models** *AAAI Demo 2022, 2022.06* [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/21738)] [Code: N/A] [Review: N/A]
-   **Presentations by the Humans and For the Humans: Harnessing LLMs for Generating Persona-Aware Slides from Documents** *EACL 2024, 2024.03* [[Paper](https://aclanthology.org/2024.eacl-long.163/)] [Code: N/A] [Review: N/A]
-   **Enhancing Presentation Slide Generation by LLMs with a Multi-Staged End-to-End Approach** *INLG 2024, 2024.09* [[Paper](https://aclanthology.org/2024.inlg-main.18/)] [Code: N/A] [Review: N/A]
-   **Presentations are not always linear! GNN meets LLM for Text Document-to-Presentation Transformation with Attribution** *EMNLP Findings 2024, 2024.11* [[Paper](https://aclanthology.org/2024.findings-emnlp.936/)] [Code: N/A] [Review: N/A]

### Peer Review _(Contributor: qiujie)_

-   **Online software spots genetic errors in cancer papers** *Nature News 2017, 2017.11* [[Paper](https://www.nature.com/articles/nature.2017.23003)] [Code: N/A] [Review: N/A]
-   **Argument Mining for Understanding Peer Reviews** *NAACL 2019, 2019.06* [[Paper](https://aclanthology.org/N19-1219/)] [Code: N/A] [Review: N/A]
-   **DeepSentiPeer: Harnessing Sentiment in Review Texts to Recommend Peer Review Decisions** *ACL 2019, 2019.07* [[Paper](https://aclanthology.org/P19-1106/)] [[Code](https://github.com/rajevv/DeepSentiPeer)] [Review: N/A]
-   **Exploring the Potential of GPT-2 for Generating Fake Reviews of Research Papers** *FAIA @ ECAI 2020, 2020.07* [[Paper](https://ebooks.iospress.nl/doi/10.3233/FAIA200717)] [[Code](https://github.com/allenai/PeerRead)] [Review: N/A]
-   **Aspect-based Sentiment Analysis of Scientific Reviews** *JCDL 2020, 2020.08* [[Paper](https://dl.acm.org/doi/10.1145/3383583.3398541)] [Code: N/A] [Review: N/A]
-   **ReviewRobot: Explainable Paper Review Generation based on Knowledge Synthesis** *INLG 2020, 2020.10* [[Paper](https://aclanthology.org/2020.inlg-1.44/)] [[Code](https://github.com/EagleW/ReviewRobot)] [Review: N/A]
-   **Uncertainty-aware machine support for paper reviewing on the interspeech 2019 submission corpus** *Interspeech 2020, 2020.10* [[Paper](https://www.isca-archive.org/interspeech_2020/stappen20_interspeech.html)] [Code: Repo Only] [Review: N/A]
-   **Multi-task Peer-Review Score Prediction** *SDP @ EMNLP 2020, 2020.11* [[Paper](https://aclanthology.org/2020.sdp-1.14/)] [Code: N/A] [Review: N/A]
-   **APE: Argument Pair Extraction from Peer Review and Rebuttal via Multi-task Learning** *EMNLP 2020, 2020.11* [[Paper](https://aclanthology.org/2020.emnlp-main.569/)] [[Code](https://github.com/LiyingCheng95/ArgumentPairExtraction)] [Review: N/A]
-   **Can We Automate Scientific Reviewing?** *JAIR 2022, 2021.03* [[Paper](https://doi.org/10.1613/jair.1.12862)] [[Code](https://github.com/neulab/ReviewAdvisor)] [Review: N/A]
-   **Argument Mining Driven Analysis of Peer-Reviews** *AAAI 2021, 2021.05* [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16607)] [[Code](https://github.com/fromm-m/aaai2021-am-peer-reviews)] [Review: N/A]
-   **A Deep Neural Architecture for Decision-Aware Meta-Review Generation** *IEEE Access 2021, 2021.09* [[Paper](https://ieeexplore.ieee.org/document/9651825)] [Code: N/A] [Review: N/A]
-   **PEERAssist: Leveraging on Paper-Review Interactions to Predict Peer Review Decisions** *ICONIP 2021, 2021.10* [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-91669-5_33)] [[Code](https://github.com/PrabhatkrBharti/PEERAssist)] [Review: N/A]
-   **HedgePeer: A Dataset for Uncertainty Detection in Peer Reviews** *IEEE Access 2022, 2022.06* [[Paper](https://ieeexplore.ieee.org/document/9852963)] [[Code](https://github.com/Tirthankar-Ghosal/HedgePeer-Dataset)] [Review: N/A]
-   **KID-Review: Knowledge-Guided Scientific Review Generation with Oracle Pre-training** *AAAI 2022, 2022.06* [[Paper](https://doi.org/10.1609/aaai.v36i10.21418)] [Code: Link Broken] [Review: N/A]
-   **Assessing Scientific Research Papers with Knowledge Graphs** *SIGIR 2022, 2022.07* [[Paper](https://dl.acm.org/doi/pdf/10.1145/3477495.3531879)] [[Code](https://github.com/kianasun/kg4rr)] [Review: N/A]
-   **SciFact-Open: Towards open-domain scientific claim verification** *arXiv, 2022.10* [[Paper](https://arxiv.org/abs/2210.13777)] [[Code](https://github.com/kianasun/kg4rr)] [Review: N/A]
-   **The Quality Assist: A Technology-Assisted Peer Review Based on Citation Functions to Predict the Paper Quality** *IEEE Access 2022, 2022.11* [[Paper](https://doi.org/10.1109/ACCESS.2022.3225871)] [Code: N/A] [Review: N/A]
-   **Exploiting Labeled and Unlabeled Data via Transformer Fine-tuning for Peer-Review Score Prediction** *EMNLP Findings 2022, 2022.12* [[Paper](https://doi.org/10.18653/v1/2022.findings-emnlp.164)] [[Code](https://github.com/panitan-m/gamma_trans)] [Review: N/A]
-   **CARE: Collaborative AI-Assisted Reading Environment** *arXiv, 2023.02* [[Paper](https://arxiv.org/abs/2302.12611v1)] [[Code](https://github.com/UKPLab/CARE)] [Review: N/A]
-   **PolitePEER: does peer review hurt? A dataset to gauge politeness intensity in the peer reviews** *Scientometrics 2023, 2023.05* [[Paper](https://link.springer.com/article/10.1007/s10579-023-09662-3)] [[Code](https://github.com/PrabhatkrBharti/PolitePEER)] [Review: N/A]
-   **Scientific Fact-Checking: A Survey of Resources and Approaches** *arXiv, 2023.05* [[Paper](https://arxiv.org/abs/2305.16859)] [Code: N/A] [Review: N/A]
-   **Scientific Opinion Summarization: Meta-review Generation with Checklist-guided Iterative Introspection** *arXiv, 2023.05* [[Paper](https://doi.org/10.48550/arXiv.2305.14647)] [[Code](https://github.com/Mankeerat/orsum-meta-review-generation)] [Review: N/A]
-   **Gpt4 is slightly helpful for peer-review assistance: A pilot study** *arXiv, 2023.06* [[Paper](https://doi.org/10.48550/arXiv.2307.05492)] [[Code](https://github.com/zrobertson466920/GPT_Auto_Review)] [Review: N/A]
-   **ReviewerGPT? An Exploratory Study on Using Large Language Models for Paper Reviewing** *arXiv, 2023.06* [[Paper](https://doi.org/10.48550/arXiv.2306.00622)] [[Code](https://github.com/niharshah/ReviewerGPT2023)] [Review: N/A]
-   **Unveiling the Sentinels: Assessing AI Performance in Cybersecurity Peer Review** *arXiv, 2023.09* [[Paper](https://doi.org/10.48550/arXiv.2309.05457)] [Code: N/A] [Review: N/A]
-   **ChatGPT and the Future of Journal Reviews: A Feasibility Study** *Cureus 2023, 2023.09* [[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10524821/#:~:text=Contextual%20Understanding%20and%20Expertise%3A%20ChatGPT,accuracy%20of%20complex%20research%20findings)] [Code: N/A] [Review: N/A]
-   **A Critical Examination of the Ethics of AI-Mediated Peer Review** *arXiv, 2023.09* [[Paper](https://arxiv.org/abs/2309.12356v1)] [Code: N/A] [Review: N/A]
-   **Can large language models provide useful feedback on research papers? A large-scale empirical analysis** *arXiv, 2023.10* [[Paper](https://doi.org/10.48550/arXiv.2310.01783)] [[Code](https://github.com/Weixin-Liang/LLM-scientific-feedback)] [Review: N/A]
-   **Automatic Analysis of Substantiation in Scientific Peer Reviews** *arXiv, 2023.11* [[Paper](https://arxiv.org/abs/2311.11967v1)] [[Code](https://github.com/YanzhuGuo/SubstanReview)] [Review: N/A]
-   **Summarizing Multiple Documents with Conversational Structure for Meta-Review Generation** *EMNLP Findings 2023, 2023.12* [[Paper](https://doi.org/10.18653/v1/2023.findings-emnlp.472)] [[Code](https://github.com/oaimli/PeerSum)] [Review: N/A]
-   **When Reviewers Lock Horn: Finding Disagreement in Scientific Peer Reviews** *EMNLP 2023, 2023.12* [[Paper](https://doi.org/10.18653/v1/2023.emnlp-main.1038)] [[Code](https://github.com/sandeep82945/Contradiction-in-Peer-Review)] [Review: N/A]
-   **PaperMage: A Unified Toolkit for Processing, Representing, and Manipulating Visually-Rich Scientific Documents** *EMNLP Demo 2023, 2023.12* [[Paper](https://aclanthology.org/2023.emnlp-demo.45.pdf)] [[Code](https://github.com/allenai/papermage)] [Review: N/A]
-   **The Intended Uses of Automated Fact-Checking Artefacts: Why, How and Who** *EMNLP Findings 2023, 2023.12* [[Paper](https://aclanthology.org/2023.findings-emnlp.577/)] [[Code](https://github.com/Cartus/Automated-Fact-Checking-Resources)] [Review: N/A]
-   **Exploring Jiu-Jitsu Argumentation for Writing Peer Review Rebuttals** *EMNLP 2023, 2023.12* [[Paper](https://aclanthology.org/2023.emnlp-main.894/)] [[Code](https://github.com/UKPLab/EMNLP2023_jiu_jitsu_argumentation_for_rebuttals)] [Review: N/A]
-   **MARG: Multi-Agent Review Generation for Scientific Papers** *arXiv, 2024.01* [[Paper](https://doi.org/10.48550/arXiv.2401.04259)] [[Code](https://github.com/allenai/marg-reviewer)] [Review: N/A]
-   **ReviewFlow: Intelligent Scaffolding to Support Academic Peer Reviewing** *arXiv, 2024.02* [[Paper](https://arxiv.org/abs/2402.03530)] [Code: Repo Only] [Review: N/A]
-   **Reviewer2: Optimizing Review Generation Through Prompt Generation** *arXiv, 2024.02* [[Paper](https://arxiv.org/abs/2402.10886v1)] [Code: N/A] [Review: N/A]
-   **Emerging plagiarism in peer-review evaluation reports: a tip of the iceberg?** *Scientometrics 2024, 2024.02* [[Paper](https://link.springer.com/article/10.1007/s11192-024-04960-1)] [Code: N/A] [Review: N/A]
-   **MetaWriter: Exploring the Potential and Perils of AI Writing Support in Scientific Peer Review** *PACMHCI @ CSCW 2024, 2024.04* [[Paper](https://doi.org/10.1145/3637371)] [Code: N/A] [Review: N/A]
-   **How We Refute Claims: Automatic Fact-Checking through Flaw Identification and Explanation** *TheWebConf 2024, 2024.05* [[Paper](https://dl.acm.org/doi/10.1145/3589335.3651521)] [[Code](https://github.com/NYCU-NLP-Lab/FlawCheck)] [Review: N/A]
-   **What Can Natural Language Processing Do for Peer Review?** *arXiv, 2024.05* [[Paper](https://arxiv.org/abs/2405.06563)] [[Code](https://github.com/OAfzal/nlp-for-peer-review)] [Review: N/A]
-   **Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews** *arXiv, 2024.06* [[Paper](https://arxiv.org/abs/2403.07183v2)] [[Code](https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers)] [Review: N/A]
-   **RelevAI-Reviewer: A Benchmark on AI Reviewers for Survey Paper Relevance** *arXiv, 2024.06* [[Paper](https://doi.org/10.48550/arXiv.2406.10294)] [[Code](https://github.com/paulohenriquecrs/RelevAI-Reviewer)] [Review: N/A]
-   **Peer Review as A Multi-Turn and Long-Context Dialogue with Role-Based Interactions** *arXiv, 2024.06* [[Paper](https://doi.org/10.48550/arXiv.2406.05688)] [[Code](https://github.com/chengtan9907/ReviewMT)] [Review: N/A]
-   **Automated Peer Reviewing in Paper SEA: Standardization, Evaluation, and Analysis** *arXiv, 2024.07* [[Paper](https://arxiv.org/abs/2407.12857v2)] [[Code](https://github.com/ecnu-sea/sea)] [Review: N/A]
-   **AI-Driven Review Systems: Evaluating LLMs in Scalable and Bias-Aware Academic Reviews** *arXiv, 2024.08* [[Paper](https://arxiv.org/pdf/2408.10365)] [[Code](https://github.com/EagleW/ReviewRobot)] [Review: N/A]
-   **GLIMPSE: Pragmatically Informative Multi-Document Summarization for Scholarly Reviews** *ACL 2024, 2024.08* [[Paper](https://doi.org/10.18653/v1/2024.acl-long.688)] [Code: N/A] [Review: N/A]
-   **Automated Focused Feedback Generation for Scientific Writing Assistance** *ACL Findings 2024, 2024.08* [[Paper](https://doi.org/10.18653/v1/2024.findings-acl.580)] [[Code](https://github.com/ericchamoun/FocusedFeedbackGeneration)] [Review: N/A]
-   **A sentiment consolidation framework for meta-review generation** *ACL 2024, 2024.08* [[Paper](https://doi.org/10.18653/v1/2024.acl-long.547)] [[Code](https://github.com/oaimli/MetaReviewingLogic)] [Review: N/A]
-   **Missci: Reconstructing Fallacies in Misrepresented Science** *ACL 2024, 2024.08* [[Paper](https://aclanthology.org/2024.acl-long.240/)] [[Code](https://github.com/UKPLab/acl2024-missci)] [Review: N/A]
-   **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery** *arXiv, 2024.08* [[Paper](https://doi.org/10.48550/arXiv.2408.06292)] [[Code](https://github.com/SakanaAI/AI-Scientist)] [Review: N/A]
-   **The emergence of Large Language Models (LLM) as a tool in literature reviews: an LLM automated systematic review** *arXiv, 2024.09* [[Paper](https://arxiv.org/abs/2409.04600v1)] [Code: N/A] [Review: N/A]
-   **Language agents achieve superhuman synthesis of scientific knowledge** *arXiv, 2024.09* [[Paper](https://arxiv.org/abs/2409.13740)] [Code: N/A] [Review: N/A]
-   **AgentReview: Exploring Peer Review Dynamics with LLM Agents** *EMNLP 2024, 2024.11* [[Paper](https://aclanthology.org/2024.emnlp-main.70)] [[Code](https://github.com/Ahren09/AgentReview)] [Review: N/A]
-   **Peer Reviews of Peer Reviews: A Randomized Controlled Trial and Other Experiments** *arXiv, 2024.11* [[Paper](https://arxiv.org/abs/2311.09497)] [Code: N/A] [Review: N/A]
-   **CycleResearcher: Improving Automated Research via Automated Review** *arXiv, 2024.11* [[Paper](https://doi.org/10.48550/arXiv.2411.00816)] [[Code](https://github.com/zhu-minjun/Researcher)] [Review: N/A]
-   **On the Rigour of Scientific Writing: Criteria, Analysis, and Insights** *EMNLP Findings 2024, 2024.11* [[Paper](https://aclanthology.org/2024.findings-emnlp.380/)] [Code: N/A] [Review: N/A]
-   **ResearchTown: Simulator of Human Research Community** *arXiv, 2024.12* [[Paper](https://doi.org/10.48550/arXiv.2412.17767)] [[Code](https://github.com/ulab-uiuc/research-town)] [Review: N/A]
-   **Prompting LLMs to Compose Meta-Review Drafts from Peer-Review Narratives of Scholarly Manuscripts** *arXiv, 2025.02* [[Paper](https://doi.org/10.48550/arXiv.2402.15589)] [[Code](https://github.com/BridgeAI-Lab/LLM-as-Meta-Reviewer)] [Review: N/A]
-   **Grounding Fallacies Misrepresenting Scientific Publications in Evidence** *arXiv, 2025.02* [[Paper](https://arxiv.org/abs/2408.12812)] [[Code](https://github.com/UKPLab/naacl2025-missciplus)] [Review: N/A]
-   **Claim Verification in the Age of Large Language Models: A Survey** *arXiv, 2025.02* [[Paper](https://arxiv.org/abs/2408.14317)] [Code: N/A] [Review: N/A]
-   **PeerArg: Argumentative Peer Review with LLMs** *arXiv, 2025.02* [[Paper](https://doi.org/10.48550/arXiv.2409.16813)] [Code: N/A] [Review: N/A]
-   **OpenReviewer: A Specialized Large Language Model for Generating Critical Scientific Paper Reviews** *arXiv, 2025.03* [[Paper](https://doi.org/10.48550/arXiv.2412.11948)] [[Code](https://huggingface.co/maxidl/Llama-OpenReviewer-8B)] [Review: N/A]
-   **ReviewAgents: Bridging the Gap Between Human and AI-Generated Paper Reviews** *arXiv, 2025.03* [[Paper](https://arxiv.org/abs/2503.08506)] [Code: N/A] [Review: N/A]
-   **DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking Process** *arXiv, 2025.03* [[Paper](https://arxiv.org/abs/2503.08569)] [[Code](https://github.com/zhu-minjun/Researcher)] [Review: N/A]
-   **Enabling Inclusive Systematic Reviews: Incorporating Preprint Articles with Large Language Model-Driven Evaluations** *arXiv, 2025.03* [[Paper](https://arxiv.org/abs/2503.13857)] [Code: N/A] [Review: N/A]
-   **Overview of the Context24 Shared Task on Contextualizing Scientific Claims** *SDP @ ACL 2024* [[Paper](https://aclanthology.org/2024.sdp-1.3/)] [Code: N/A] [Review: N/A]

---

## 🏆 Benchmarks

- **OpenAI Science Benchmark** [[Leaderboard]](https://openai.com/research/science-benchmarks)
- **AI2 Scientific Reasoning Challenge** [[Dataset]](https://allenai.org/data/arc)

---

## 🌐 Community

### Upcoming Events

- **NeurIPS 2024 AI4Science Workshop** (December 2024)
- **AAAI Conference on Automated Scientific Discovery** (February 2025)

### Discussion Forums

- [AI Research Discord](https://discord.gg/airesearch)
- [r/MachineLearning](https://reddit.com/r/MachineLearning)

---

## 👋 Contributing

**Contribution Workflow:**

1. **Fork** the repository.
2. **Create a new branch:** `git checkout -b feature/your-feature`
3. **Modify resources** as needed.
4. **Submit a Pull Request (PR)** with a detailed description.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🎖️ Credits and Acknowledgements

* Maintained with ❤️ by [Your Name] and the community.
* Last updated: **July 2024**

---
