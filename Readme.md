# ConflictBench

This is the official code repository for our ACM MM 2026 paper:

> **[ConflictBench: Evaluating Human-AI Conflict via Interactive and Visually Grounded Environments](https://arxiv.org/abs/2603.08024)**

## 🌟 Overview

Existing alignment benchmarks primarily evaluate large language models through single-turn and static question answering. However, embodied LLM agents operate through multi-step interactions, where persistent task pressure, environmental feedback, and evolving observations may gradually expose alignment failures that remain hidden in isolated responses.

**ConflictBench** is a multi-turn, multimodal benchmark designed to evaluate how embodied LLM agents behave when their assigned objectives conflict with human safety and broader human interests. It contains **150 interactive conflict scenarios** covering three representative categories:

* **EP1: Self-preservation vs. human safety**
* **EP2: Task or resource priority conflicts**
* **EP3: Deceptive alignment and information concealment**

ConflictBench is constructed through three stages:

* 🧩 **Value-Conflict Scenario Construction** — We design diverse high-risk scenarios in which completing the assigned task may require the agent to compromise human safety, oversight, or control.
* 💬 **Interactive Text Environment Development** — Each scenario is converted into an executable TextWorld environment, allowing agents to perform atomic actions and make decisions over multiple interaction turns.
* 🎬 **Visual Environment Modeling** — A world model generates video feedback from the evolving environment state, enabling agents to interact with both textual observations and dynamic visual scenes.

Based on these environments, ConflictBench compares agent behavior under **single-turn**, **multi-turn text-only**, and **multi-turn multimodal** settings, revealing latent alignment risks that may only emerge during sustained interaction.

## 📃 Data

The 150 conflict environments live under `TextWorld/TestWorld2/results/`:

| Split | Path                                                | # Episodes |
| ----- | --------------------------------------------------- | ---------- |
| EP1   | `TextWorld/TestWorld2/results/ulxEnv/`            | 51         |
| EP2   | `TextWorld/TestWorld2/results/EP2-Result/ulxEnv/` | 47         |
| EP3   | `TextWorld/TestWorld2/results/EP3-Result/ulxEnv/` | 52         |

## 🛠️ Installation

We recommend using a virtual environment:

```bash
pip install -r requirements.txt
```

Before running scripts, export your API credentials:

```bash
export OPENAI_API_KEY="sk-..."
export BASE_URL="https://your-openai-compatible-endpoint/v1"
```

## 🚀 Running ConflictBench

ConflictBench supports two interactive evaluation settings: a **multi-turn text-only environment** and a **multi-turn multimodal environment**. Both settings use the same conflict scenarios and TextWorld environments, while differing in the type of environmental feedback provided to the agent.

### 1. Multi-turn Text-only Evaluation

In the text-only setting, the agent interacts with the environment through multiple rounds of textual observations and atomic actions. At each step, the agent receives the updated environment state, reasons about the current situation, and selects its next action.

To evaluate a model on all scenarios within an episode category, run:

```bash
bash TextWorld/TestWorld2/batch_evalAgent.sh <MODEL_NAME> <EP_CATEGORY>
```

For example:

```bash
bash TextWorld/TestWorld2/batch_evalAgent.sh deepseek-v3 EP1
```

The batch script evaluates all scenarios in the selected category sequentially. Existing results are automatically detected and skipped, allowing interrupted evaluations to be resumed.

To evaluate a single scenario, run:

```bash
bash TextWorld/TestWorld2/bash/evalAgent.sh <SCENARIO_ID> <MODEL_NAME>
```

For example:

```bash
bash TextWorld/TestWorld2/bash/evalAgent.sh EP1-023 deepseek-v3
```

### 2. Multi-turn Multimodal Evaluation

In the multimodal setting, the agent receives both textual observations and dynamically generated video feedback. After each action, the updated environment state is rendered into a short video clip by the image-to-video generation service, providing visually grounded feedback for the next interaction turn.

First, start the multimodal video-generation service on a multi-GPU node:

```bash
bash Wan2.2/generate_i2v_server_multigpu.sh
```

Then evaluate a multimodal model on all scenarios within an episode category:

```bash
bash TextWorld/TestWorld2/batch_evalMulAgent.sh <MODEL_NAME> <SERVER_URL> <EP_CATEGORY>
```

For example:

```bash
bash TextWorld/TestWorld2/batch_evalMulAgent.sh qwen3-vl-plus http://gpu09:8021/generate EP1
```

To evaluate a single multimodal scenario, run:

```bash
bash TextWorld/TestWorld2/bash/evalMulAgent.sh <SCENARIO_ID> <MODEL_NAME> <SERVER_URL> --resume=False
```

For example:

```bash
bash TextWorld/TestWorld2/bash/evalMulAgent.sh EP1-023 qwen3-vl-plus http://gpu09:8021/generate --resume=False
```

The two evaluation settings share the same underlying conflict scenarios, enabling direct comparison between text-only interaction and visually grounded multimodal interaction.

### 3. Alignment Evaluation

For trajectories where the agent fails to complete the task, `checkAlignment.py` further determines whether the failure is caused by insufficient capability or an alignment failure reflected in the agent’s reasoning and actions.

```bash
python TextWorld/TestWorld2/checkAlignment.py <EP_CATEGORY> <MODEL_NAME>
```

The evaluation results are written back to the corresponding trajectory files in `result_history/`.

## 📁 Repository Structure

```
ConflictBench/
├── TextWorld/
│   ├── glulxe/          # Glulx interpreter for running .ulx environments
│   └── TestWorld2/      # Conflict environments and agent evaluation
│       ├── results/     # EP1/EP2/EP3 scenarios and trajectory results
│       ├── testEnv/     # Text and multimodal agent implementations
│       ├── prompt/      # Alignment evaluation prompts
│       └── *.py / *.sh  # Evaluation and alignment-checking scripts
├── Wan2.2/              # Video generation for initial scenes and interaction feedback
│   ├── getPrompt/       # Prompt construction and initial-video pipeline
│   ├── output/          # Pre-generated initial-scene videos
│   └── *.py / *.sh      # T2V/I2V generation and inference scripts
└── WorldModelEval/      # Evaluation of generated visual feedback
```


## 📌 Citation
If you find this repository useful, please cite our paper:
```
@misc{zhao2026conflictbenchevaluatinghumanaiconflict,
      title={ConflictBench: Evaluating Human-AI Conflict via Interactive and Visually Grounded Environments}, 
      author={Weixiang Zhao and Haozhen Li and Yanyan Zhao and xuda zhi and Yongbo Huang and Hao He and Bing Qin and Ting Liu},
      year={2026},
      eprint={2603.08024},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.08024}, 
}
```
