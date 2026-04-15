#!/usr/bin/env python3
"""
BiasScope — synthesis and verification stage (error rates under bias-conditioned rewrites).

For every bias in the (possibly extended) library, swaps the rejected side with the
teacher-rewritten text, optionally shuffles A/B for position debiasing, and logs
per-domain error rates. Effective biases (higher error than baseline) can be merged
into the per-run bias JSON (see ``_save_bias_history``).

The teacher that rewrites rejected answers can be vLLM or an OpenAI-compatible API
(``--teacher-backend``); the judge under test is always local vLLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import (
    OpenAICompatCompletionEngine,
    assert_api_teacher_config,
    build_judge_prompts,
    build_sampled_bias_col,
    extract_new_biases_json,
    generate_in_batch,
    get_args,
    modify_rejected_responses,
    modify_rejected_responses_api,
    parse_responses,
    resolve_tensor_parallel_size,
    shuffle_responses_and_label,
    _compute_error_rate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _log_stage(title: str) -> None:
    bar = "=" * 22
    logger.info("%s %s %s", bar, title, bar)


def _compute_domain_error_rates(
    df: pd.DataFrame, preds: list[str | None]
) -> Dict[str, float]:
    """``overall`` plus one key per ``domain`` column value when present."""
    out: Dict[str, float] = {"overall": _compute_error_rate(df, preds)}
    if "domain" not in df.columns:
        return out
    for dom, g in df.groupby("domain"):
        idx = g.index
        sub_preds = [preds[i] for i in idx]
        out[str(dom)] = _compute_error_rate(g, sub_preds)
    return out


def _run_one_judge_pass(
    df: pd.DataFrame,
    batch_size: int,
    sampling_params: Any,
    tokenizer: Any,
    llm: Any,
) -> Dict[str, float]:
    prompts = build_judge_prompts(df, tokenizer)
    preds = parse_responses(
        generate_in_batch(prompts, llm, batch_size=batch_size, sampling_params=sampling_params)
    )
    return _compute_domain_error_rates(df, preds)


def _load_bias_history(bias_num: Path, model_basic_bias: Path) -> list[dict[str, Any]]:
    if not bias_num.exists():
        with open(model_basic_bias, "r", encoding="utf-8") as f:
            base_cnt = len(json.load(f))
        history = [{"exp_id": 0, "step_bias_change": 0, "total_bias": base_cnt}]
        bias_num.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        return history
    return json.loads(bias_num.read_text(encoding="utf-8"))


def _save_bias_history(
    avg_results: dict,
    model_basic_bias: Path,
    biases: dict,
    bias_history: list,
    temp_bias_num: Path,
    bias_log_path: Path,
    test_data_tag: str,
) -> None:
    origin = avg_results.get("origin")
    origin_err = origin["overall"] if isinstance(origin, dict) else origin
    need_add = {}
    for k, v in avg_results.items():
        if k == "origin":
            continue
        err = v["overall"] if isinstance(v, dict) else v
        if err > origin_err:
            need_add[k] = v
    if need_add:
        with open(model_basic_bias, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for key in list(need_add.keys()):
            if key not in existing:
                existing[key] = biases[key]
        with open(model_basic_bias, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
    delta = len(need_add)
    if temp_bias_num.exists():
        tc = json.loads(temp_bias_num.read_text(encoding="utf-8"))
        original_new_cnt = tc.get("original_new_biases")
        after_merge_cnt = tc.get("new_biases_after_merging")
    else:
        original_new_cnt = after_merge_cnt = None
    new_entry = {
        "exp_id": len(bias_history),
        "step_bias_change": delta,
        "total_bias": bias_history[-1]["total_bias"] + delta,
        "original_new_biases": original_new_cnt,
        "new_biases_after_merging": after_merge_cnt,
        "test_data_tag": test_data_tag,
    }
    bias_history.append(new_entry)
    bias_log_path.write_text(
        json.dumps(bias_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_result_history(result_path: Path) -> list:
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_results_history(
    avg_results: dict,
    result_history: list,
    result_path: Path,
    test_data_tag: str,
) -> None:
    origin = avg_results.get("origin")
    origin_err = origin["overall"] if isinstance(origin, dict) else origin
    higher_or_equal = {}
    lower = {}
    for k, v in avg_results.items():
        err = v["overall"] if isinstance(v, dict) else v
        if err >= origin_err:
            higher_or_equal[k] = v
        else:
            lower[k] = v
    entry = {
        "iteration": len(result_history) + 1,
        "effective biases and origin": higher_or_equal,
        "ineffective biases": lower,
        "test_data_tag": test_data_tag,
    }
    result_history.append(entry)
    result_path.write_text(
        json.dumps(result_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = get_args()
    analysis_path = args.analysis_data_path
    test_path = args.test_data_path
    model_path = args.model_path
    llm_tp = resolve_tensor_parallel_size(args.self_defined_tp_size)
    seed = args.seed
    batch_size = args.batch_size
    max_tokens = args.max_new_tokens
    teacher_path = args.teacher_model_path
    teacher_backend = args.teacher_backend

    _log_stage("Stage 0/4: Prepare paths, history, and input data")
    sampling = SamplingParams(temperature=0, max_tokens=max_tokens)
    basic_bias = Path(args.bias_json)

    model_tag = Path(model_path).name
    data_tag = Path(analysis_path).stem
    bias_tag = basic_bias.stem
    test_data_tag = Path(test_path).stem

    slug_dir = basic_bias.parent / model_tag / data_tag / bias_tag
    slug_dir.mkdir(parents=True, exist_ok=True)
    model_basic_bias = slug_dir / f"{bias_tag}.json"
    model_extend_bias = slug_dir / "temporary_extended_biases.json"
    temp_bias_num = slug_dir / "temp_bias_number.json"

    results_root = Path(__file__).resolve().parent / "results"
    bias_subdir = results_root / model_tag / data_tag / bias_tag
    bias_subdir.mkdir(parents=True, exist_ok=True)
    bias_number_path = bias_subdir / "bias_number.json"
    results_path = bias_subdir / "avg_error_rate.json"

    modified_root = Path(analysis_path).parent.parent / "modified_data"
    md_subdir = modified_root / model_tag / data_tag / bias_tag
    md_subdir.mkdir(parents=True, exist_ok=True)
    attacked_all_biases_path = md_subdir / "attacked_data_with_all_new_biases.parquet"

    bias_history = _load_bias_history(bias_number_path, model_basic_bias)
    result_history = _load_result_history(results_path)

    test_df = pd.read_parquet(test_path)

    _log_stage("Stage 1/4: Build bias-conditioned attacked test set")
    biases = extract_new_biases_json(model_basic_bias, model_extend_bias)
    n_biases = len(biases)
    df_tagged = build_sampled_bias_col(test_df, biases, m=n_biases, seed=seed)
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
            dtype=args.dtype,
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

    df_rejected.to_parquet(attacked_all_biases_path, index=False)
    logger.info("Saved attacked test data to %s", attacked_all_biases_path)

    _log_stage("Stage 2/4: Judge baseline error on original responses")
    judge_llm = LLM(
        model=model_path,
        tensor_parallel_size=llm_tp,
        seed=seed,
        gpu_memory_utilization=0.6,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    tokenizer = judge_llm.get_tokenizer()

    avg_results: dict[str, Any] = {}
    origin_df = df_rejected[["question", "response_A", "response_B", "label", "domain"]].copy()
    origin_rates = _run_one_judge_pass(
        origin_df, batch_size, sampling, tokenizer, judge_llm
    )
    logger.info("origin overall error = %.4f", origin_rates["overall"])
    avg_results["origin"] = origin_rates

    _log_stage("Stage 3/4: Per-bias verification loop")
    for bias_name in tqdm(list(biases.keys()), desc="per-bias verification"):
        rej_col = f"{bias_name}_rejected"
        tmp_df = df_rejected.copy()
        cond_a = tmp_df["label"] == 1
        cond_b = tmp_df["label"] == 2
        tmp_df["response_A"] = np.where(cond_b, tmp_df[rej_col], tmp_df["response_A"])
        tmp_df["response_B"] = np.where(cond_a, tmp_df[rej_col], tmp_df["response_B"])
        eval_df = tmp_df[["question", "response_A", "response_B", "label", "domain"]]
        shuffled = shuffle_responses_and_label(eval_df, seed=seed)
        rates = _run_one_judge_pass(shuffled, batch_size, sampling, tokenizer, judge_llm)
        logger.info("%s overall error = %.4f", bias_name, rates["overall"])
        avg_results[bias_name] = rates

    _log_stage("Stage 4/4: Save results and update bias history")
    _save_bias_history(
        avg_results=avg_results,
        model_basic_bias=model_basic_bias,
        biases=biases,
        bias_history=bias_history,
        temp_bias_num=temp_bias_num,
        bias_log_path=bias_number_path,
        test_data_tag=test_data_tag,
    )
    _save_results_history(
        avg_results=avg_results,
        result_history=result_history,
        result_path=results_path,
        test_data_tag=test_data_tag,
    )
    _log_stage("Pipeline finished: synthesis + verification completed")


if __name__ == "__main__":
    main()
