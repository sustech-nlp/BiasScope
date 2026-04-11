"""
BiasScope — bias detection and library merge (Lai et al., ICLR 2026; arXiv:2602.09383).

Uses a teacher backend with a vLLM-compatible ``.generate`` API (local vLLM or HTTP wrapper).

``detection_mode`` selects which fields from each row feed the classifier prompt:
  1 — reasoning trace only
  2 — reasoning trace + textual explanation
"""

from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

from prompts import _BIAS_USER_TEMPLATE1, _BIAS_USER_TEMPLATE2, _CLASSIFY_NEW_BIAS_SYS
from utils import extract_new_biases_json, generate_in_batch, parse_responses, _parse_choice

logger = logging.getLogger(__name__)

_JSON_MD_PATTERN = re.compile(r"(```(?:json)?\s*)([\s\S]*?)(\s*```)")


class BiasDetector:
    """
    Batch bias labeling with vLLM. Copies ``basic_json`` to ``extend_json`` on init.
    """

    def __init__(
        self,
        llm: Any,
        sampling_params: Any,
        basic_json: str | Path,
        extend_json: str | Path,
    ) -> None:
        from shutil import copy2

        self.llm = llm
        self.sampling_params = sampling_params
        self.tokenizer = llm.get_tokenizer()
        self.basic_json = Path(basic_json)
        self.extend_json = Path(extend_json)
        copy2(self.basic_json, self.extend_json)

    def load_library(self) -> Dict[str, Dict[str, str]]:
        return dict(json.loads(self.extend_json.read_text(encoding="utf-8")))

    def library_to_str(self) -> str:
        lines = []
        for name, d in self.load_library().items():
            lines.append(f"{name}:\nDefinition: {d['Definition']}\n")
        return "\n".join(lines)

    def detect_df(
        self,
        df: pd.DataFrame,
        detection_mode: int,
        batch_size: int = 8,
    ) -> pd.DataFrame:
        """
        Append columns: ``bias_whether``, ``bias_name``, ``bias_def``.
        New bias definitions are merged into ``extend_json`` when present in the model output.
        """
        questions = df["question"].tolist()
        resp_as = df["response_A"].tolist()
        resp_bs = df["response_B"].tolist()
        labels = df["label"].tolist()
        expls = df["explanation"].tolist()
        chosens = [2 if lab == 1 else 1 for lab in labels]
        reasons = df["reasoning process"].tolist()

        prompts_text = []
        for q, a, b, c, e, r in zip(questions, resp_as, resp_bs, chosens, expls, reasons):
            if detection_mode == 1:
                template = _BIAS_USER_TEMPLATE1.format(
                    question=q,
                    resp_a=a,
                    resp_b=b,
                    chosen=c,
                    reason=r,
                )
            else:
                template = _BIAS_USER_TEMPLATE2.format(
                    question=q,
                    resp_a=a,
                    resp_b=b,
                    chosen=c,
                    explanation=e,
                    reason=r,
                )
            msgs = [{"role": "user", "content": template}]
            prompts_text.append(
                self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        raw_outputs = generate_in_batch(
            prompts_text,
            self.llm,
            batch_size=batch_size,
            sampling_params=self.sampling_params,
        )
        results: list[dict[str, Any]] = []
        for text in parse_responses(raw_outputs):
            m = _JSON_MD_PATTERN.search(text)
            if m:
                raw_json = m.group(2).strip()
                try:
                    results.append(json.loads(raw_json))
                except json.JSONDecodeError:
                    cleaned = raw_json.replace("\\", "\\\\")
                    try:
                        results.append(json.loads(cleaned))
                    except json.JSONDecodeError as err:
                        results.append(
                            {
                                "whether": "yes",
                                "name": f"json decode error: {err}",
                                "Definition": None,
                            }
                        )
            else:
                results.append(
                    {
                        "whether": "yes",
                        "name": "error, this response is not in a json format",
                        "Definition": None,
                    }
                )

        out = df.copy()
        out["bias_whether"] = [r.get("whether") for r in results]
        out["bias_name"] = [
            r.get("name") if r.get("whether") == "yes" else None for r in results
        ]
        out["bias_def"] = [
            r.get("Definition") if r.get("whether") == "yes" else None for r in results
        ]

        for r in results:
            if r.get("whether") == "yes" and r.get("Definition") is not None:
                new_name = r.get("name")
                if not new_name:
                    continue
                appendix = {new_name: {"Definition": r["Definition"]}}
                lib = self.load_library()
                lib.update(appendix)
                self.extend_json.write_text(
                    json.dumps(lib, indent=3, ensure_ascii=False),
                    encoding="utf-8",
                )
        return out


def _init_or_load_bias_counts(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    init = {"original_new_biases": 0, "new_biases_after_merging": 0}
    path.write_text(json.dumps(init, ensure_ascii=False, indent=2), encoding="utf-8")
    return init


def run_bias_analysis(
    df: pd.DataFrame,
    basic_bias: str | Path,
    extended_bias: str | Path,
    llm: Any,
    sampling_params: Any,
    model_path: str | Path,
    temp_bias_num: str | Path,
    detection_mode: int,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Run row-wise bias detection, write parquet under ``basic_bias``'s parent, merge novel
    biases into ``extended_bias``, and persist count statistics to ``temp_bias_num``.

    ``model_path`` is reserved for logging and downstream provenance.
    """
    _ = model_path  # reserved for experiment tracking
    stats_file = Path(temp_bias_num)
    bias_counts = _init_or_load_bias_counts(stats_file)

    detector = BiasDetector(
        llm=llm,
        sampling_params=sampling_params,
        basic_json=basic_bias,
        extend_json=extended_bias,
    )

    logger.info("Starting batch bias detection (vLLM).")
    enriched_df = detector.detect_df(df, detection_mode=detection_mode, batch_size=batch_size)

    out_parent = Path(basic_bias).parent / "analysis_results"
    out_parent.mkdir(parents=True, exist_ok=True)
    out_path = out_parent / "wrong_with_reason_and_bias.parquet"
    enriched_df.to_parquet(out_path, index=False)
    logger.info("Bias analysis saved to %s", out_path)

    logger.info("Merging new biases into library.")
    tmp_library = copy.deepcopy(
        json.loads(Path(basic_bias).read_text(encoding="utf-8"))
    )
    len_basic = len(tmp_library)
    new_bias_dict = extract_new_biases_json(Path(basic_bias), Path(extended_bias))

    for name, meta in tqdm(new_bias_dict.items(), desc="merge new biases"):
        is_new = True
        for old_name, old_meta in tmp_library.items():
            bias_text = f"{old_name}:\nDefinition: {old_meta['Definition']}\n"
            user_msg = _CLASSIFY_NEW_BIAS_SYS.format(
                bias_name=name,
                bias_library_text=bias_text,
                definition=meta["Definition"],
            )
            msgs = [{"role": "user", "content": user_msg}]
            prompt_full = detector.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            out = generate_in_batch(
                [prompt_full],
                detector.llm,
                batch_size=1,
                sampling_params=sampling_params,
                tqdm_=False,
            )
            decision = _parse_choice(parse_responses(out)[0])
            if decision == 0:
                is_new = False
                break
        if is_new:
            tmp_library[name] = meta

    Path(extended_bias).write_text(
        json.dumps(tmp_library, ensure_ascii=False, indent=3),
        encoding="utf-8",
    )
    logger.info(
        "Original candidate new biases: %d; after merge: +%d vs basic.",
        len(new_bias_dict),
        len(tmp_library) - len_basic,
    )

    bias_counts.update(
        {
            "original_new_biases": len(new_bias_dict),
            "new_biases_after_merging": len(tmp_library) - len_basic,
        }
    )
    stats_file.write_text(
        json.dumps(bias_counts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Bias merge counts saved to %s", stats_file)

    return enriched_df
