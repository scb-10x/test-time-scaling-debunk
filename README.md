# Budget Forcing: A Test-Time Scaling Analysis

**This repository is archived and released as-is. If you’re interested in this work, please contact us.**

---

This repository hosts analysis scripts and experimental data for investigating budget forcing, a test-time scaling technique for reasoning models.

## Overview

Budget forcing is a test-time scaling technique that controls model reasoning by:
- **Forcing continuation**: Appending keywords like "Wait" to encourage the model to continue thinking
- **Forcing termination**: Stopping the model's reasoning process when a token budget is exceeded by directly requesting the answer

In this project, we revisit budget forcing and explore several key research questions:

1. **Generalization**: To what extent does budget forcing generalize to other models beyond the original implementation?
2. **Non-reasoning models**: Does budget forcing work effectively with non-reasoning models?
3. **Keyword alternatives**: Can we use keywords other than "Wait" for budget forcing?

## Experimental Setup

### Models Tested

We evaluated budget forcing across multiple model families:
- **Standard Instruction Models**: Qwen2.5 7B, Llama 3.1 8B, Mistral, Gemma
- **Reasoning-Specialized Models**: OpenThinker3-7B, DeepSeek-R1-Distill-Qwen-7B, simplescaling/s1.1-7B
- **Fine-tuned Models**: RFT (Reinforcement Fine-Tuned)

### Prompting Strategies

- **Zero-shot**: Direct prompting without examples
- **CoT** (Chain of Thought): Standard chain of thought prompting
- **CoT+BF** (Chain of Thought + Budget Forcing): CoT with budget forcing applied

### Keywords Tested

- `Wait` - Original keyword
- `Perhaps` - Alternative keyword
- `Let` - Alternative keyword

### Budget Sizes

Token budgets ranging from 256 to 8192 tokens (256, 512, 1024, 2048, 4096, 8192)

### Benchmarks

- **AIME 2025**: Mathematical reasoning
- **MATH500**: Mathematical problem solving
- **MMLU Pro-1K**: Multi-task language understanding
- **SuperGPQA-1K**: General question answering

## Repository Structure

```
.
├── analysis/                                    # Analysis scripts
├── data/
│   └── results.csv                              # Experimental results
├── pyproject.toml                               # Project dependencies
└── README.md                                    # This file
```

## Data Format

The `results.csv` file contains the following columns:
- `Model`: Model name/identifier
- `Prompting`: Prompting strategy used
- `Keyword`: Budget forcing keyword (if applicable)
- `Budget`: Token budget (if applicable)
- `AIME 2025`: Score on AIME 2025 benchmark
- `MATH500`: Score on MATH500 benchmark
- `MMLU Pro-1K`: Score on MMLU Pro-1K benchmark
- `SuperGPQA-1K`: Score on SuperGPQA-1K benchmark

## Getting Started

### Prerequisites

This project uses `uv` for dependency management. Install dependencies:

```bash
uv sync
```