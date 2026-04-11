# BiasScope

Official implementation for **BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation** (ICLR 2026).

[![Paper](https://img.shields.io/badge/paper-arXiv:2602.09383-b31b1b.svg)](https://arxiv.org/abs/2602.09383)

LLM-as-a-judge is widely used, but **evaluation bias** undermines reliability. BiasScope is an **LLM-driven framework** that automatically discovers potential biases at scale—moving bias discovery from manual, predefined lists toward **active, automated exploration**. The method is validated on **JudgeBench**; the paper also introduces **JudgeBench-Pro** as a harder robustness benchmark.

## Repository structure

```
├── attack_judge_and_analysis.py   # Main discovery pipeline
├── synthesis_bias_verification.py # Per-bias error rates & library updates
├── bias_detector.py               # Bias classification + merge
├── prompts.py                     # All LLM prompts
├── utils.py                       # CLI, vLLM batching, data helpers
├── run_biasscope.sh               # Example end-to-end driver
├── data/                          # Bias JSON + example parquet layout
├── requirements.txt       
└── README.md
```

## Requirements

- Linux + NVIDIA GPU (CUDA)  
- **Python 3.12** (reference: conda env `vllm` at `/volume/wzhang/ghchen/laip/miniconda3/envs/vllm`)  
- [vLLM](https://github.com/vllm-project/vllm) for local **judge** and (by default) **teacher** inference  
- Optional: OpenAI-compatible HTTP API for the **teacher** only (`--teacher-backend api`)

Install:

```bash
pip install -r requirements.txt

```

If `torch` / `vLLM` wheels fail, use the [vLLM install guide](https://docs.vllm.ai) and a matching PyTorch CUDA index (e.g. `cu128` for this lock).

## Data layout

Place preference-style **Parquet** files (columns such as `question`, `response_A`, `response_B`, `label`, and optionally `domain`) under `data/`. This repository ships a minimal layout:

| Path | Role |
|------|------|
| `data/bias/basic_biases.json` | Seed bias library (definitions). |
| `data/rewardbench/rewardbench_filtered.parquet` | Example analysis set (configure in `run_biasscope.sh`). |
| `data/judgeBench/judge_bench.parquet` | Example held-out test set for verification. |

Obtain or build **JudgeBench** / **RewardBench** preprocessings according to the paper and your license terms; replace paths in the shell script as needed.

**Outputs** (ignored by `.gitignore` by default):

- `results/<judge_model>/<dataset_stem>/<bias_lib_stem>/` — metrics such as `avg_error_rate.json`, `bias_number.json`
- `data/modified_data/...` — attacked / synthesized parquet caches

## Quick start

From the repository root:

```bash
bash run_biasscope.sh
```

The script `cd`s to the repo root automatically. Edit **`run_biasscope.sh`** to set:

- `GPU`, `BATCH_SIZE`, `model_paths` (judge checkpoints)
- `TEACHER_MODEL_PATH` (or export `TEACHER_MODEL_PATH` before running)
- `ANALYSIS_DATA_PATH_LIST`, `TEST_DATA_PATH`, `BIAS_JSON`

### Stage 1 — Attack, judge, bias analysis

```bash
python attack_judge_and_analysis.py \
  --bias-json data/bias/basic_biases.json \
  --analysis-data-path data/rewardbench/rewardbench_filtered.parquet \
  --model-path /path/to/judge \
  --teacher-model-path /path/to/teacher \
  --self-defined-tp-size 2 \
  --batch-size 64 \
  --detection-mode 2
```

### Stage 2 — Synthesis & verification (error rates per bias)

```bash
python synthesis_bias_verification.py \
  --bias-json data/bias/basic_biases.json \
  --analysis-data-path data/rewardbench/rewardbench_filtered.parquet \
  --test-data-path data/judgeBench/judge_bench.parquet \
  --model-path /path/to/judge \
  --teacher-model-path /path/to/teacher \
  --self-defined-tp-size 2 \
  --batch-size 64
```

### API teacher (optional)

Use an OpenAI-compatible server for the **teacher** only (judge remains vLLM):

```bash
python attack_judge_and_analysis.py \
  --teacher-backend api \
  --api-key "$OPENAI_API_KEY" \
  --base-url "https://api.openai.com/v1" \
  --teacher-model "gpt-4o" \
  ...other flags...
```

See `python attack_judge_and_analysis.py --help` for all options.


## Citation

If you use this code or build on BiasScope, please cite:

```bibtex
@article{lai2026biasscope,
  title={BiasScope: Towards Automated Detection of Bias in LLM-as-a-Judge Evaluation},
  author={Lai, Peng and Ou, Zhihao and Wang, Yong and Wang, Longyue and Yang, Jian and Chen, Yun and Chen, Guanhua},
  journal={arXiv preprint arXiv:2602.09383},
  year={2026}
}
```