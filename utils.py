"""
Shared helpers: CLI arguments, vLLM batching, preference-data transforms, and API helpers.

This package is intended to be run with the repository root as the working directory
so that sibling modules (``prompts``, ``utils``, ``bias_detector``, …) resolve on ``sys.path``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

from prompts import _PROMPT_JUDGE, _PROMPT_SYTHESIS, _PROMPT_SYTHESIS2

# ---------------------------------------------------------------------------
# Preference / judge prompts
# ---------------------------------------------------------------------------


def build_judge_prompts(df: pd.DataFrame, tokenizer: Any) -> list[str]:
    """Format each row as a chat template string for pairwise judging."""
    texts: list[str] = []
    for _, row in df.iterrows():
        p = _PROMPT_JUDGE.format(
            question=row["question"],
            answer1=row["response_A"],
            answer2=row["response_B"],
        )
        messages = [{"role": "user", "content": p}]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return texts


def build_judge_prompts_with_sys_prompt(
    df: pd.DataFrame,
    system_prompt: str,
    tokenizer: Any,
) -> list[str]:
    """Prepend debiasing instructions to the user turn (no separate system role)."""
    texts: list[str] = []
    for _, row in df.iterrows():
        p = _PROMPT_JUDGE.format(
            question=row["question"],
            answer1=row["response_A"],
            answer2=row["response_B"],
        )
        messages = [{"role": "user", "content": system_prompt + p}]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return texts


def format_bias_prompt(library: dict, prompt_mode: int = 2) -> str:
    """Build a debiasing preamble listing known biases (for ``build_judge_prompts_with_sys_prompt``)."""
    lines = []
    for name, d in library.items():
        if prompt_mode == 1:
            lines.append(f"{name}: {d['Definition']} {d['Manifestations']}")
        else:
            lines.append(f"{name}: {d['Definition']}")
    return (
        "You are acting as an impartial evaluator.\n\n"
        "You will be given two candidate answers to the same question. "
        "Your job is to decide which answer is superior based on the following criteria:\n"
        "- Accuracy: Is the content factually correct and precise?\n"
        "- Coherence: Is the reasoning logical and internally consistent?\n"
        "- Helpfulness: Does it fully address the question and provide useful information?\n"
        "- Ethics: Does it avoid harmful, biased, or unethical content?\n\n"
        "While making your judgment, you must carefully avoid common cognitive biases "
        "that can distort decision-making. These biases are listed below. "
        "Read them carefully and ensure they do not influence your preference. "
        "If you detect any such bias creeping in, consciously adjust your reasoning "
        "to stay fair and objective:\n\n"
        "## Biases\n"
        + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_tensor_parallel_size(requested: int | None) -> int:
    """
    Tensor-parallel degree for vLLM teacher and judge.

    When ``requested`` is None, use every visible CUDA device (at least 1).
    """
    if requested is not None:
        return requested
    return max(1, int(torch.cuda.device_count()))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BiasScope — attack preference data, evaluate judges, discover biases (see arXiv:2602.09383).",
    )
    parser.add_argument("--bias-json", type=str, default=None, help="Path to JSON bias library.")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size.")
    parser.add_argument(
        "--self-defined-tp-size",
        type=int,
        default=None,
        help="Tensor parallel size for vLLM teacher and judge (default: all visible GPUs, at least 1).",
    )
    parser.add_argument("--model-path", type=str, help="HuggingFace or local path to the judge model.")
    parser.add_argument("--analysis-data-path", type=str, help="Parquet used in the attack / analysis stage.")
    parser.add_argument("--test-data-path", type=str, help="Parquet used in the verification stage.")
    parser.add_argument(
        "--teacher-backend",
        type=str,
        choices=("vllm", "api"),
        default="vllm",
        help="Teacher for bias rewriting & bias detection: local vLLM or OpenAI-compatible HTTP API.",
    )
    parser.add_argument("--api-key", type=str, help="OpenAI-compatible API key (required when --teacher-backend api).")
    parser.add_argument("--base-url", type=str, help="OpenAI-compatible base URL (optional; e.g. https://api.openai.com/v1).")
    parser.add_argument(
        "--teacher-model",
        type=str,
        help="Remote teacher model id (API name) when --teacher-backend api.",
    )
    parser.add_argument("--teacher-model-path", type=str, help="Local teacher weights path when --teacher-backend vllm.")
    parser.add_argument(
        "--api-max-workers",
        type=int,
        default=32,
        help="Max parallel HTTP requests per batch when using API teacher (thread pool).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum new tokens to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype (e.g. bfloat16).")
    parser.add_argument(
        "--num-random-biases",
        type=int,
        default=4,
        help="Number of bias names sampled per row (when sampling is enabled).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="Legacy threshold for fragile-sample detection in related experiments.",
    )
    parser.add_argument(
        "--detection-mode",
        type=int,
        default=2,
        help="Bias detector: 1 = reasoning only; 2 = reasoning + explanation (default).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# vLLM outputs
# ---------------------------------------------------------------------------


def parse_responses(raw_outputs: Any) -> list[str]:
    return [out.outputs[0].text for out in raw_outputs]


def _parse_choice(text: str) -> int | None:
    m = re.search(r"Decision:\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _compute_error_rate(df: pd.DataFrame, preds: list[str | None]) -> float:
    choices = [_parse_choice(p) for p in preds]
    valid = np.array([c is not None for c in choices])
    correct = np.array([c == lbl for c, lbl in zip(choices, df["label"])])
    if valid.sum() == 0:
        return 1.0
    return 1.0 - correct[valid].mean()


def swap_df(df: pd.DataFrame) -> pd.DataFrame:
    """Swap A/B and flip labels (1 <-> 2)."""
    swapped = df.copy()
    swapped["response_A"], swapped["response_B"] = swapped["response_B"], swapped["response_A"]
    swapped["label"] = 3 - swapped["label"]
    return swapped


def generate_in_batch(
    prompts: list[Any],
    llm: Any,
    batch_size: int,
    sampling_params: Any,
    tqdm_: bool = True,
) -> list[Any]:
    all_outputs: list[Any] = []
    if tqdm_:
        for start in tqdm(range(0, len(prompts), batch_size)):
            batch_prompts = prompts[start : start + batch_size]
            batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            all_outputs.extend(batch_outputs)
    else:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            batch_outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            all_outputs.extend(batch_outputs)
    return all_outputs


# ---------------------------------------------------------------------------
# OpenAI-compatible HTTP backend (mimics vLLM ``LLM.generate`` for our batching code)
# ---------------------------------------------------------------------------


class _SimpleChoice:
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class _SimpleRequestOutput:
    __slots__ = ("outputs",)

    def __init__(self, text: str) -> None:
        self.outputs = [_SimpleChoice(text)]


class PlainTokenizer:
    """Use raw user text as the prompt (no HuggingFace chat template)."""

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        return messages[-1]["content"]


class OpenAICompatCompletionEngine:
    """
    Drives ``generate_in_batch`` via synchronous ``chat.completions`` calls.
    Each prompt string is sent as a single user message.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_workers: int = 32,
    ) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._default_max_tokens = max_tokens
        self._temperature = temperature
        self._max_workers = max_workers
        self._tokenizer = PlainTokenizer()

    def get_tokenizer(self) -> PlainTokenizer:
        return self._tokenizer

    def generate(
        self,
        prompts: list[str],
        sampling_params: Any,
        use_tqdm: bool = False,
    ) -> list[_SimpleRequestOutput]:
        max_tok = getattr(sampling_params, "max_tokens", None) or self._default_max_tokens
        temp = getattr(sampling_params, "temperature", None)
        if temp is None:
            temp = self._temperature

        def one(p: str) -> str:
            r = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": p}],
                max_tokens=max_tok,
                temperature=temp,
            )
            return r.choices[0].message.content or ""

        if len(prompts) <= 1:
            texts = [one(prompts[0])] if prompts else []
        else:
            from concurrent.futures import ThreadPoolExecutor

            max_w = max(1, min(self._max_workers, len(prompts)))
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                texts = list(pool.map(one, prompts))
        return [_SimpleRequestOutput(t) for t in texts]


def assert_api_teacher_config(api_key: str | None, teacher_model: str | None) -> None:
    if not api_key or not teacher_model:
        raise ValueError(
            "API teacher requires non-empty --api-key and --teacher-model "
            "(remote model id for the OpenAI-compatible server)."
        )


def collect_plain_modify_rejected_prompts(
    df: pd.DataFrame,
    bias_json: dict,
) -> tuple[list[str], list[tuple[int, str]]]:
    """Build one user message per (row, sampled bias) using ``_PROMPT_SYTHESIS2``."""
    prompts_text: List[str] = []
    location_map: list[tuple[int, str]] = []
    for idx, row in df.iterrows():
        question = str(row["question"]).strip()
        wrong = str(row["response_A"] if row["label"] == 2 else row["response_B"]).strip()
        for bias_name in row.get("sampled_bias", []):
            if bias_name not in bias_json:
                continue
            info = bias_json[bias_name]
            bias_text = f"{bias_name}\nDefinition: {info['Definition']}\n"
            user_content = _PROMPT_SYTHESIS2.format(
                question=question,
                answer=wrong,
                bias=bias_text,
            )
            prompts_text.append(user_content)
            location_map.append((idx, bias_name))
    return prompts_text, location_map


# ---------------------------------------------------------------------------
# API-based teacher (optional)
# ---------------------------------------------------------------------------


async def _make_api_call(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    question: str,
    answer: str,
    bias_name: str,
    definition: str,
    manifestations: str,
) -> str:
    bias_prompt = (
        f"{bias_name}\n\n## Definition:\n{definition}\n\n## Manifestations:\n{manifestations}"
    )
    prompt = _PROMPT_SYTHESIS.format(
        question=question,
        correct_answer=answer,
        bias=bias_prompt,
    )
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            print(f"[Error] API call failed: {exc}")
            return " "


def add_rejected_answers_concurrent(
    df: pd.DataFrame,
    json_bias: dict,
    api_key: str,
    base_url: str,
    teacher_model: str,
    max_parallel: int = 10,
) -> pd.DataFrame:
    """Fill ``{bias}_rejected`` columns using an OpenAI-compatible HTTP API."""
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(max_parallel)
    bias_names = list(json_bias.keys())
    bias_columns = [f"{name}_rejected" for name in bias_names]
    for col in bias_columns:
        if col not in df.columns:
            df[col] = pd.NA

    async def async_main() -> None:
        tasks: list[Tuple[Any, str, asyncio.Task]] = []
        for idx, row in df.iterrows():
            question = row["question"]
            correct_answer = row["response_A"] if row["label"] == 1 else row["response_B"]
            sampled_biases = row["sampled_bias"]
            for bias_name in sampled_biases:
                bias_obj = json_bias[bias_name]
                defn = bias_obj["Definition"]
                mani = bias_obj["Manifestations"]
                col_name = f"{bias_name}_rejected"
                task = asyncio.create_task(
                    _make_api_call(
                        semaphore,
                        client=client,
                        model=teacher_model,
                        question=question,
                        answer=correct_answer,
                        bias_name=bias_name,
                        definition=defn,
                        manifestations=mani,
                    )
                )
                tasks.append((idx, col_name, task))
        for idx, col_name, task in tqdm(tasks, desc="Generating rejected answers"):
            result = await task
            df.at[idx, col_name] = result

    asyncio.run(async_main())
    return df


# ---------------------------------------------------------------------------
# vLLM: synthesize / modify rejected side
# ---------------------------------------------------------------------------


def synthesize_rejected_responses(
    df: pd.DataFrame,
    bias_json: dict,
    llm: Any,
    batch_size: int = 32,
    sampling_params: Any = None,
) -> pd.DataFrame:
    """Generate wrong answers from the *chosen* response using ``_PROMPT_SYTHESIS``."""
    prompts_text: List[str] = []
    location_map: list[tuple[int, str]] = []
    tokenizer = llm.get_tokenizer()
    for idx, row in df.iterrows():
        question = str(row["question"]).strip()
        correct = str(row["response_A"] if row["label"] == 1 else row["response_B"]).strip()
        for bias_name in row.get("sampled_bias", []):
            if bias_name not in bias_json:
                continue
            info = bias_json[bias_name]
            bias_text = f"{bias_name}\nDefinition: {info['Definition']}\n"
            user_content = _PROMPT_SYTHESIS.format(
                question=question,
                correct_answer=correct,
                bias=bias_text,
            )
            turns = [{"role": "user", "content": user_content}]
            chat_str = tokenizer.apply_chat_template(
                turns,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts_text.append(chat_str)
            location_map.append((idx, bias_name))
    if not prompts_text:
        return df
    raw_outputs = generate_in_batch(
        prompts_text,
        llm=llm,
        batch_size=batch_size,
        sampling_params=sampling_params,
    )
    generated_texts = parse_responses(raw_outputs)
    for (idx, bias_name), g_text in zip(location_map, generated_texts):
        df.at[idx, f"{bias_name}_rejected"] = g_text
    return df


def modify_rejected_responses(
    df: pd.DataFrame,
    bias_json: dict,
    llm: Any,
    batch_size: int = 32,
    sampling_params: Any = None,
) -> pd.DataFrame:
    """Rewrite the *rejected* response in-place for each sampled bias (``_PROMPT_SYTHESIS2``)."""
    plain_prompts, location_map = collect_plain_modify_rejected_prompts(df, bias_json)
    if not plain_prompts:
        return df
    tokenizer = llm.get_tokenizer()
    prompts_text = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in plain_prompts
    ]
    raw_outputs = generate_in_batch(
        prompts_text,
        llm=llm,
        batch_size=batch_size,
        sampling_params=sampling_params,
    )
    generated_texts = parse_responses(raw_outputs)
    for (idx, bias_name), g_text in zip(location_map, generated_texts):
        df.at[idx, f"{bias_name}_rejected"] = g_text
    return df


def modify_rejected_responses_api(
    df: pd.DataFrame,
    bias_json: dict,
    engine: OpenAICompatCompletionEngine,
    batch_size: int = 32,
    sampling_params: Any = None,
) -> pd.DataFrame:
    """
    Same behavior as ``modify_rejected_responses`` but calls an OpenAI-compatible server
    with plain user prompts (no HuggingFace chat template).
    """
    plain_prompts, location_map = collect_plain_modify_rejected_prompts(df, bias_json)
    if not plain_prompts:
        return df
    raw_outputs = generate_in_batch(
        plain_prompts,
        engine,
        batch_size=batch_size,
        sampling_params=sampling_params,
    )
    generated_texts = parse_responses(raw_outputs)
    for (idx, bias_name), g_text in zip(location_map, generated_texts):
        df.at[idx, f"{bias_name}_rejected"] = g_text
    return df


def shuffle_responses_and_label(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Randomly swap A/B (and flip label) on ~50% of rows to reduce position bias in eval."""
    rng = np.random.default_rng(seed)
    df_out = df.copy()
    swap_mask = rng.random(len(df)) < 0.5
    df_out.loc[swap_mask, "response_A_new"] = df.loc[swap_mask, "response_B"]
    df_out.loc[swap_mask, "response_B_new"] = df.loc[swap_mask, "response_A"]
    df_out.loc[~swap_mask, "response_A_new"] = df.loc[~swap_mask, "response_A"]
    df_out.loc[~swap_mask, "response_B_new"] = df.loc[~swap_mask, "response_B"]
    df_out["response_A"] = df_out["response_A_new"]
    df_out["response_B"] = df_out["response_B_new"]
    df_out.loc[swap_mask, "label"] = 3 - df.loc[swap_mask, "label"]
    return df_out.drop(columns=["response_A_new", "response_B_new"]).reset_index(drop=True)


def build_sampled_bias_col(
    df: pd.DataFrame,
    json_bias: dict,
    m: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Add ``sampled_bias`` column: each row lists ``m`` distinct bias keys with near-uniform
    global frequency across the dataset.
    """
    if m == 0:
        out = df.copy()
        out["sampled_bias"] = [[] for _ in range(len(df))]
        return out
    np.random.seed(seed)
    bias_names = list(json_bias.keys())
    k = len(bias_names)
    n = len(df)
    if m > k:
        raise ValueError("m must be <= number of biases")
    total_needed = n * m
    base, rem = divmod(total_needed, k)
    freq_list = [base + 1] * rem + [base] * (k - rem)
    pool: list[str] = []
    for name, freq in zip(bias_names, freq_list):
        pool.extend([name] * freq)
    np.random.shuffle(pool)
    result: list[list[str]] = []
    pos = 0
    for _ in range(n):
        seen: set[str] = set()
        row: list[str] = []
        attempts = 0
        while len(row) < m and attempts < 20:
            if pos >= len(pool):
                break
            name = pool[pos]
            pos += 1
            if name not in seen:
                row.append(name)
                seen.add(name)
            else:
                attempts += 1
        while len(row) < m:
            for name in np.random.permutation(bias_names):
                if name not in seen:
                    row.append(name)
                    seen.add(name)
                    break
        result.append(row)
    out = df.copy()
    out["sampled_bias"] = result
    return out


def apply_new_rejected_to_responses(df: pd.DataFrame) -> None:
    """In-place: replace the ground-truth *worse* answer with ``new_rejected``."""
    df["response_A"] = np.where(
        df["label"] == 2,
        df["new_rejected"],
        df["response_A"],
    )
    df["response_B"] = np.where(
        df["label"] == 1,
        df["new_rejected"],
        df["response_B"],
    )


def extract_new_biases_json(basic_path: Path, extend_path: Path) -> Dict[str, Dict]:
    """Return entries present in ``extend_path`` but not in ``basic_path``."""
    with open(basic_path, "r", encoding="utf-8") as fp:
        basic: Dict = json.load(fp)
    with open(extend_path, "r", encoding="utf-8") as fp:
        extend: Dict = json.load(fp)
    return {k: v for k, v in extend.items() if k not in basic}
