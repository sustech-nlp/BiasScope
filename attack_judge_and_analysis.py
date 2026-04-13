#!/usr/bin/env python3
"""
BiasScope — attack, judge, explain, and detect biases (Lai et al., ICLR 2026).

Paper: https://arxiv.org/abs/2602.09383

Stages
------
1. Teacher model rewrites the rejected answer to amplify a sampled cognitive bias.
2. Judge model scores attacked pairs; misaligned predictions are collected.
3. Judge explains its choice; a second pass classifies bias in the explanation.

Teacher models: use local vLLM (``--teacher-backend vllm`` + ``--teacher-model-path``) or an
OpenAI-compatible API (``--teacher-backend api`` + ``--api-key``, ``--teacher-model``, optional ``--base-url``).
The judge model (``--model-path``) is always loaded with vLLM.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams

import bias_detector
from prompts import _PROMPT_EXPLAIN
from utils import (
    OpenAICompatCompletionEngine,
    apply_new_rejected_to_responses,
    assert_api_teacher_config,
    build_judge_prompts,
    build_sampled_bias_col,
    generate_in_batch,
    get_args,
    modify_rejected_responses,
    modify_rejected_responses_api,
    parse_responses,
    resolve_tensor_parallel_size,
    _parse_choice,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def _log_stage(title: str) -> None:
    bar = "=" * 22
    logger.info("%s %s %s", bar, title, bar)


def _explode_by_sampled_biases(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (example, bias): copy core columns and set ``new_rejected`` from
    ``{bias_name}_rejected``.
    """
    rows: list[dict[str, Any]] = []
    core = ["question", "response_A", "response_B", "label"]
    for _, row in df.iterrows():
        biases = row.get("sampled_bias")
        if biases is None or (isinstance(biases, float) and np.isnan(biases)):
            continue
        if not isinstance(biases, (list, tuple)):
            continue
        base = {c: row[c] for c in core}
        for bias_name in biases:
            rej_col = f"{bias_name}_rejected"
            rej_text = row.get(rej_col)
            if pd.isna(rej_text):
                continue
            new_row = {**base, "new_rejected": rej_text}
            rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def _build_explain_prompts(df: pd.DataFrame, tokenizer: Any) -> list[str]:
    """Prompt the judge to justify the decision recorded in ``judge_decision`` (1=A, 2=B)."""
    texts: list[str] = []
    for _, row in df.iterrows():
        jd = int(row["judge_decision"])
        chosen = "A" if jd == 1 else "B"
        user = _PROMPT_EXPLAIN.format(
            question=row["question"],
            answer1=row["response_A"],
            answer2=row["response_B"],
            chosen=chosen,
            reason=row["reasoning process"],
        )
        messages = [{"role": "user", "content": user}]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return texts


def _find_wrong_predictions(df: pd.DataFrame, preds: list[str | None]) -> pd.DataFrame:
    """Rows where the parsed Decision differs from ``label`` (and parsing succeeded)."""
    choices = [_parse_choice(p) for p in preds]
    valid_mask = pd.notnull(choices)
    wrong_mask = (df["label"] != choices) & valid_mask
    wrong_df = df[wrong_mask].reset_index(drop=True)
    wrong_df["judge_decision"] = [c for c, keep in zip(choices, wrong_mask) if keep]
    logger.info(
        "Valid predictions: %.1f%%; misaligned with gold label: %d rows.",
        100 * float(valid_mask.mean()),
        len(wrong_df),
    )
    return wrong_df


def _resolve_paths(
    analysis_data_path: str,
    model_path: str,
    bias_json: str,
) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    """Standard layout: ``data/bias/<model>/<dataset>/<bias_set>/`` and ``modified_data`` mirror."""
    basic_bias = Path(bias_json)
    model_tag = Path(model_path).name
    data_tag = Path(analysis_data_path).stem
    bias_tag = basic_bias.stem

    slug_dir = basic_bias.parent / model_tag / data_tag / bias_tag
    slug_dir.mkdir(parents=True, exist_ok=True)

    model_basic_bias = slug_dir / f"{bias_tag}.json"
    model_extend_bias = slug_dir / "temporary_extended_biases.json"
    temp_bias_num = slug_dir / "temp_bias_number.json"

    modified_root = Path(analysis_data_path).parent.parent / "modified_data"
    md_subdir = modified_root / model_tag / data_tag / bias_tag
    md_subdir.mkdir(parents=True, exist_ok=True)
    attacked_path = md_subdir / "attacked.parquet"
    wrong_path = md_subdir / "wrong_explanation.parquet"

    results_subdir = Path(__file__).resolve().parent / "results" / model_tag / data_tag / bias_tag
    results_subdir.mkdir(parents=True, exist_ok=True)

    return (
        basic_bias,
        model_basic_bias,
        model_extend_bias,
        temp_bias_num,
        attacked_path,
        wrong_path,
        results_subdir,
    )


def main() -> None:
    args = get_args()
    analysis_path = args.analysis_data_path
    model_path = args.model_path
    teacher_path = args.teacher_model_path
    teacher_backend = args.teacher_backend
    llm_tp = resolve_tensor_parallel_size(args.self_defined_tp_size)
    batch_size = args.batch_size
    seed = args.seed
    max_tokens = args.max_new_tokens
    detection_mode = args.detection_mode
    dtype = args.dtype

    sampling = SamplingParams(temperature=0, max_tokens=max_tokens)

    (
        basic_bias,
        model_basic_bias,
        model_extend_bias,
        temp_bias_num,
        attacked_path,
        wrong_path,
        _results_subdir,
    ) = _resolve_paths(analysis_path, model_path, args.bias_json)

    _log_stage("Stage 0/3: Prepare inputs and bias files")
    logger.info("Analysis data: %s", analysis_path)

    if not model_basic_bias.exists():
        shutil.copy2(basic_bias, model_basic_bias)

    with open(model_basic_bias, encoding="utf-8") as f:
        biases = json.load(f)

    _log_stage("Stage 1/3: Teacher rewrites rejected responses")
    df = pd.read_parquet(analysis_path)
    df_tagged = build_sampled_bias_col(df, biases, m=1, seed=seed)
    api_teacher: OpenAICompatCompletionEngine | None = None
    if teacher_backend == "api":
        assert_api_teacher_config(args.api_key, args.teacher_model)
        api_teacher = OpenAICompatCompletionEngine(
            api_key=args.api_key,
            base_url=args.base_url or None,
            model=args.teacher_model,
            max_tokens=max_tokens,
            temperature=0.0,
            max_workers=args.api_max_workers,
        )
        df_rejected = modify_rejected_responses_api(
            df_tagged,
            biases,
            api_teacher,
            batch_size=batch_size,
            sampling_params=sampling,
        )
    else:
        if not teacher_path:
            raise ValueError("vLLM teacher requires --teacher-model-path.")
        teacher_llm = LLM(
            model=teacher_path,
            tensor_parallel_size=llm_tp,
            seed=seed,
            gpu_memory_utilization=0.9,
            dtype=dtype,
            enable_chunked_prefill=False,
            max_num_seqs=batch_size,
        )
        df_rejected = modify_rejected_responses(
            df_tagged,
            biases,
            teacher_llm,
            batch_size=batch_size,
            sampling_params=sampling,
        )
        del teacher_llm
        torch.cuda.empty_cache()

    logger.info("Building attacked pairs from sampled biases.")
    attacked_df = _explode_by_sampled_biases(df_rejected)
    apply_new_rejected_to_responses(attacked_df)
    attacked_df.to_parquet(attacked_path, index=False)
    logger.info("Wrote attacked preference data to %s", attacked_path)

    _log_stage("Stage 2/3: Judge scoring and wrong-case explanations")
    judge_llm = LLM(
        model=model_path,
        tensor_parallel_size=llm_tp,
        seed=seed,
        gpu_memory_utilization=0.6,
        dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = judge_llm.get_tokenizer()
    attacked_df = pd.read_parquet(attacked_path)
    logger.info("Running judge on attacked pairs.")
    judge_prompts = build_judge_prompts(attacked_df, tokenizer)
    judge_raw = generate_in_batch(
        judge_prompts, judge_llm, batch_size=batch_size, sampling_params=sampling
    )
    judge_text = parse_responses(judge_raw)
    attacked_df["reasoning process"] = judge_text

    wrong_df = _find_wrong_predictions(attacked_df, judge_text)
    logger.info("Generating explanations for wrong predictions.")
    explain_prompts = _build_explain_prompts(wrong_df, tokenizer)
    explain_raw = generate_in_batch(
        explain_prompts, judge_llm, batch_size=batch_size, sampling_params=sampling
    )
    wrong_df["explanation"] = parse_responses(explain_raw)
    wrong_df.to_parquet(wrong_path, index=False)
    logger.info("Wrote wrong-prediction explanations to %s", wrong_path)

    del judge_llm
    torch.cuda.empty_cache()

    _log_stage("Stage 3/3: Bias detection and library merge")
    wrong_df = pd.read_parquet(wrong_path)
    if teacher_backend == "api":
        teacher_for_detect = api_teacher
    else:
        teacher_for_detect = LLM(
            model=teacher_path,
            tensor_parallel_size=llm_tp,
            seed=seed,
            gpu_memory_utilization=0.9,
            dtype=dtype,
            enable_chunked_prefill=False,
            max_num_seqs=batch_size,
        )
    bias_detector.run_bias_analysis(
        wrong_df,
        model_basic_bias,
        model_extend_bias,
        teacher_for_detect,
        sampling,
        model_path,
        temp_bias_num=temp_bias_num,
        detection_mode=detection_mode,
        batch_size=batch_size,
    )
    if teacher_backend != "api":
        del teacher_for_detect
        torch.cuda.empty_cache()
    _log_stage("Pipeline finished: attack + analysis completed")


if __name__ == "__main__":
    main()
