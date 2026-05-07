"""Evaluate ForSeeBench MCQs using generated audio-description context."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forseebench.evaluation.metrics import compute_mcq_metrics
from forseebench.io.write_jsonl import read_jsonl, write_jsonl
from forseebench.qwen.parse_outputs import parse_json_object
from forseebench.qwen.qwen_client import QwenClient


@dataclass(frozen=True, slots=True)
class GeneratedADRow:
    movie: str
    start: float
    end: float
    text: str
    ground_truth: str


def normalize_movie_name(value: str) -> str:
    """Normalize MAD/ForSeeBench movie identifiers for cross-file matching."""

    normalized = re.sub(r"^\d+_", "", value).lower()
    normalized = normalized.replace("and_the_goblet_of_fire", "and_the_goblet_of_fire")
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized


def load_generated_ad_rows(path: str | Path, *, column: str) -> list[GeneratedADRow]:
    """Load generated audio-description rows from Results.csv."""

    rows: list[GeneratedADRow] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path} does not contain column {column!r}")
        for row in reader:
            text = (row.get(column) or "").strip()
            if not text:
                continue
            rows.append(
                GeneratedADRow(
                    movie=normalize_movie_name(row["movie"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                    text=text,
                    ground_truth=(row.get("ground_truth") or "").strip(),
                )
            )
    return rows


def build_generated_ad_index(rows: list[GeneratedADRow]) -> dict[str, list[GeneratedADRow]]:
    index: dict[str, list[GeneratedADRow]] = {}
    for row in rows:
        index.setdefault(row.movie, []).append(row)
    for movie_rows in index.values():
        movie_rows.sort(key=lambda row: (row.start, row.end))
    return index


def find_generated_ad(
    index: dict[str, list[GeneratedADRow]],
    *,
    movie: str,
    start: float,
    end: float,
    tolerance: float = 0.02,
) -> GeneratedADRow | None:
    """Find a generated AD row by normalized movie and timestamp."""

    normalized_movie = normalize_movie_name(movie)
    for row in index.get(normalized_movie, []):
        if abs(row.start - start) <= tolerance and abs(row.end - end) <= tolerance:
            return row
    return None


def normalize_ad_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def normalize_ad_lookup_key(value: str) -> str:
    value = re.sub(r"\b([a-z0-9]+)'s\b", r"\1", value.lower())
    return re.sub(r"[^a-z0-9]+", "", value)


def build_ground_truth_index(rows: list[GeneratedADRow]) -> dict[tuple[str, str], list[GeneratedADRow]]:
    index: dict[tuple[str, str], list[GeneratedADRow]] = {}
    for row in rows:
        if row.ground_truth:
            index.setdefault((row.movie, normalize_ad_text(row.ground_truth)), []).append(row)
            index.setdefault((row.movie, normalize_ad_lookup_key(row.ground_truth)), []).append(row)
    return index


def materialize_autoad_context(
    example: dict[str, Any],
    index: dict[str, list[GeneratedADRow]],
    *,
    text_index: dict[tuple[str, str], list[GeneratedADRow]] | None = None,
    tolerance: float = 0.02,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return generated AD context sentences plus any unmatched source clips."""

    return materialize_generated_context_clips(
        selected_prior_context_clips(example),
        index,
        text_index=text_index,
        tolerance=tolerance,
    )


def selected_prior_context_clips(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Return selected adaptive context clips that strictly precede the hidden target."""

    clips = list(example.get("context", []))
    target_sequence_index = (example.get("target") or {}).get("sequence_index")
    if target_sequence_index is None:
        return clips
    target_sequence_index = int(target_sequence_index)
    return [
        clip
        for clip in clips
        if clip.get("sequence_index") is None
        or int(clip["sequence_index"]) < target_sequence_index
    ]


def materialize_generated_context_clips(
    clips: list[dict[str, Any]],
    index: dict[str, list[GeneratedADRow]],
    *,
    text_index: dict[tuple[str, str], list[GeneratedADRow]] | None = None,
    tolerance: float = 0.02,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return generated AD sentences for arbitrary context clips."""

    generated_context: list[str] = []
    unmatched: list[dict[str, Any]] = []
    for clip in clips:
        match = find_generated_ad(
            index,
            movie=clip["movie"],
            start=float(clip["timestamp_start"]),
            end=float(clip["timestamp_end"]),
            tolerance=tolerance,
        )
        if match is None and text_index is not None:
            text_key = (
                normalize_movie_name(clip["movie"]),
                normalize_ad_text(clip.get("audio_description") or ""),
            )
            candidates = text_index.get(text_key, [])
            if not candidates:
                text_key = (
                    normalize_movie_name(clip["movie"]),
                    normalize_ad_lookup_key(clip.get("audio_description") or ""),
                )
                candidates = text_index.get(text_key, [])
            if len(candidates) == 1:
                match = candidates[0]
            elif len(candidates) > 1:
                nearest = min(
                    candidates,
                    key=lambda row: abs(row.start - float(clip["timestamp_start"]))
                    + abs(row.end - float(clip["timestamp_end"])),
                )
                if (
                    abs(nearest.start - float(clip["timestamp_start"]))
                    + abs(nearest.end - float(clip["timestamp_end"]))
                    <= 6.0
                ):
                    match = nearest
        if match is None:
            unmatched.append(
                {
                    "clip_id": clip.get("clip_id"),
                    "movie": clip.get("movie"),
                    "timestamp_start": clip.get("timestamp_start"),
                    "timestamp_end": clip.get("timestamp_end"),
                }
            )
        else:
            generated_context.append(match.text)
    return generated_context, unmatched


def load_sequence_clip_index(parsed_sequences_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load ordered clip metadata from per-movie parsed sequence files."""

    base = Path(parsed_sequences_dir)
    if not base.exists():
        raise FileNotFoundError(f"parsed sequence directory does not exist: {base}")

    index: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(base.glob("*/parsed_sequences.jsonl")):
        rows = read_jsonl(path)
        if not rows:
            continue
        rows.sort(key=lambda row: int(row["sequence_index"]))
        movie = rows[0]["movie"]
        index[movie] = rows
    return index


def fixed_prior_context(
    example: dict[str, Any],
    clip_index: dict[str, list[dict[str, Any]]],
    *,
    k: int,
) -> list[dict[str, Any]]:
    """Return the last k clips before the hidden target for an example."""

    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return []

    movie = example["movie"]
    target = example.get("target") or {}
    target_clip_id = example.get("target_clip_id") or target.get("clip_id")
    target_sequence_index = target.get("sequence_index")
    if target_sequence_index is None:
        for clip in clip_index.get(movie, []):
            if clip.get("clip_id") == target_clip_id:
                target_sequence_index = clip["sequence_index"]
                break
    if target_sequence_index is None:
        raise KeyError(f"could not locate target sequence index for {example.get('id')}")

    prior = [
        clip
        for clip in clip_index.get(movie, [])
        if int(clip["sequence_index"]) < int(target_sequence_index)
    ]
    return prior[-k:]


def make_option_order(example_id: str, num_options: int) -> list[int]:
    """Return a stable per-example option permutation."""

    seed = int(hashlib.sha256(example_id.encode("utf-8")).hexdigest()[:16], 16)
    order = list(range(num_options))
    random.Random(seed).shuffle(order)
    return order


def make_mcq_prompt(
    example: dict[str, Any],
    generated_context: list[str],
    *,
    source_name: str | None,
    option_order: list[int] | None = None,
) -> str:
    if option_order is None:
        option_order = list(range(len(example["options"])))
    displayed_options = [example["options"][original_idx] for original_idx in option_order]
    options = "\n".join(f"{idx}. {option}" for idx, option in enumerate(displayed_options))
    if not generated_context:
        return f"""You are answering a multiple-choice question about what happens next in a movie.

Choose the best answer.

Question:
{example["question"]}

Options:
{options}

Return strict JSON with keys "prediction_idx" and "prediction_text".
"prediction_idx" must be one of 0, 1, 2, or 3.
"prediction_text" must exactly match the selected option.
"""
    context = "\n".join(f"{idx + 1}. {sentence}" for idx, sentence in enumerate(generated_context))
    return f"""You are answering a multiple-choice question about what happens next in a movie.

Base your answer on the prior audio descriptions below. First identify which answer option is best supported by the prior descriptions, then return only the final JSON answer.

Prior audio descriptions:
{context}

Question:
{example["question"]}

Options:
{options}

Return strict JSON with keys "prediction_idx" and "prediction_text".
"prediction_idx" must be one of 0, 1, 2, or 3.
"prediction_text" must exactly match the selected option.
"""


def normalize_prediction(
    parsed: dict[str, Any],
    options: list[str],
    *,
    option_order: list[int] | None = None,
) -> tuple[int | None, str | None, int | None, str | None]:
    if option_order is None:
        option_order = list(range(len(options)))
    displayed_options = [options[original_idx] for original_idx in option_order]
    raw_idx = parsed.get("prediction_idx", parsed.get("answer_idx", parsed.get("prediction")))
    if isinstance(raw_idx, str) and raw_idx.strip().isdigit():
        raw_idx = int(raw_idx.strip())
    if isinstance(raw_idx, int) and 0 <= raw_idx < len(displayed_options):
        original_idx = option_order[raw_idx]
        return original_idx, options[original_idx], raw_idx, displayed_options[raw_idx]

    raw_text = parsed.get("prediction_text", parsed.get("answer_text"))
    if isinstance(raw_text, str):
        stripped = raw_text.strip()
        for displayed_idx, option in enumerate(displayed_options):
            if stripped == option or stripped.lower() == option.lower():
                original_idx = option_order[displayed_idx]
                return original_idx, options[original_idx], displayed_idx, option
    return None, None, None, None


def load_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    return read_jsonl(path)


def compute_autoad_metrics(examples: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_predictions = {
        row["id"]: row["prediction_idx"]
        for row in prediction_rows
        if isinstance(row.get("prediction_idx"), int)
    }
    metrics = compute_mcq_metrics(examples, valid_predictions)
    metrics["num_invalid_predictions"] = sum(
        1 for row in prediction_rows if not isinstance(row.get("prediction_idx"), int)
    )
    return metrics


def _evaluate_examples_with_contexts(
    *,
    examples: list[dict[str, Any]],
    contexts_by_id: dict[str, list[str]],
    output_dir: str | Path,
    qwen_client: QwenClient | None,
    source_name: str | None,
    dry_run_prompts: int = 0,
    shuffle_options: bool = True,
    run_name: str = "answer_autoad_mcq",
    batch_size: int = 1,
) -> dict[str, Any]:
    """Evaluate examples with already materialized text contexts."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path / "predictions.jsonl"
    prompts_path = output_path / "prompt_preview.jsonl"
    metrics_path = output_path / "metrics.json"

    prediction_rows = load_prediction_rows(predictions_path)
    completed_ids = {row["id"] for row in prediction_rows}
    prompt_rows: list[dict[str, Any]] = []

    pending: list[tuple[dict[str, Any], list[int], str]] = []
    for example in examples:
        option_order = (
            make_option_order(example["id"], len(example["options"]))
            if shuffle_options
            else list(range(len(example["options"])))
        )
        prompt = make_mcq_prompt(
            example,
            contexts_by_id[example["id"]],
            source_name=source_name,
            option_order=option_order,
        )
        if len(prompt_rows) < dry_run_prompts:
            prompt_rows.append({"id": example["id"], "option_order": option_order, "prompt": prompt})
        if dry_run_prompts and qwen_client is None:
            continue
        if example["id"] in completed_ids:
            continue
        pending.append((example, option_order, prompt))

    if pending and qwen_client is None:
        raise RuntimeError("qwen_client is required unless dry_run_prompts is used")

    with predictions_path.open("a", encoding="utf-8") as prediction_handle:
        if qwen_client is not None and batch_size > 1 and qwen_client.backend == "local_hf":
            for batch in _batched(pending, batch_size):
                raw_texts = _generate_json_batch_local_hf(
                    qwen_client,
                    [prompt for _, _, prompt in batch],
                )
                for (example, option_order, _prompt), raw_text in zip(batch, raw_texts, strict=True):
                    row = _prediction_row_from_raw_text(
                        example=example,
                        option_order=option_order,
                        raw_text=raw_text,
                        source_name=source_name,
                        context_length=len(contexts_by_id[example["id"]]),
                        run_name=run_name,
                    )
                    if row["prediction_idx"] is None:
                        row = _prediction_row_from_qwen_call(
                            qwen_client=qwen_client,
                            example=example,
                            option_order=option_order,
                            prompt=_prompt,
                            source_name=source_name,
                            context_length=len(contexts_by_id[example["id"]]),
                            run_name=f"{run_name}_retry",
                        )
                    prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    prediction_handle.flush()
                    prediction_rows.append(row)
                    completed_ids.add(example["id"])
        else:
            for example, option_order, prompt in pending:
                row = _prediction_row_from_qwen_call(
                    qwen_client=qwen_client,
                    example=example,
                    option_order=option_order,
                    prompt=prompt,
                    source_name=source_name,
                    context_length=len(contexts_by_id[example["id"]]),
                    run_name=run_name,
                )
                prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                prediction_handle.flush()
                prediction_rows.append(row)
                completed_ids.add(example["id"])

    if prompt_rows:
        write_jsonl(prompts_path, prompt_rows)
    metrics = compute_autoad_metrics(examples, prediction_rows)
    metrics.update(
        {
            "source_name": source_name,
            "shuffle_options": shuffle_options,
            "num_evaluated_examples": len(examples),
            "predictions_path": str(predictions_path),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def _prediction_row_from_raw_text(
    *,
    example: dict[str, Any],
    option_order: list[int],
    raw_text: str,
    source_name: str | None,
    context_length: int,
    run_name: str,
) -> dict[str, Any]:
    parsed: dict[str, Any] | None = None
    error: str | None = None
    prediction_idx: int | None = None
    prediction_text: str | None = None
    displayed_prediction_idx: int | None = None
    displayed_prediction_text: str | None = None
    try:
        parsed = parse_json_object(raw_text)
        (
            prediction_idx,
            prediction_text,
            displayed_prediction_idx,
            displayed_prediction_text,
        ) = normalize_prediction(parsed, example["options"], option_order=option_order)
    except Exception as exc:  # noqa: BLE001 - record and continue evaluation.
        error = f"{type(exc).__name__}: {exc}"
    return {
        "id": example["id"],
        "prediction_idx": prediction_idx,
        "prediction_text": prediction_text,
        "displayed_prediction_idx": displayed_prediction_idx,
        "displayed_prediction_text": displayed_prediction_text,
        "option_order": option_order,
        "displayed_options": [example["options"][idx] for idx in option_order],
        "answer_idx": example.get("answer_idx"),
        "answer_text": example.get("answer_text"),
        "displayed_answer_idx": option_order.index(example["answer_idx"])
        if isinstance(example.get("answer_idx"), int)
        else None,
        "correct": prediction_idx == example.get("answer_idx"),
        "raw_text": raw_text,
        "parsed": parsed,
        "error": error,
        "context_length": context_length,
        "source_name": source_name,
        "run_name": run_name,
    }


def _prediction_row_from_qwen_call(
    *,
    qwen_client: QwenClient | None,
    example: dict[str, Any],
    option_order: list[int],
    prompt: str,
    source_name: str | None,
    context_length: int,
    run_name: str,
) -> dict[str, Any]:
    if qwen_client is None:
        raise RuntimeError("qwen_client is required")
    try:
        result = qwen_client.generate_json(run_name, prompt, temperature=0.0)
        return _prediction_row_from_raw_text(
            example=example,
            option_order=option_order,
            raw_text=result.raw_text,
            source_name=source_name,
            context_length=context_length,
            run_name=run_name,
        )
    except Exception as exc:  # noqa: BLE001 - record and continue evaluation.
        row = _prediction_row_from_raw_text(
            example=example,
            option_order=option_order,
            raw_text="",
            source_name=source_name,
            context_length=context_length,
            run_name=run_name,
        )
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


def _generate_json_batch_local_hf(qwen_client: QwenClient, prompts: list[str]) -> list[str]:
    """Generate a batch of text-only JSON responses with the local HF backend."""

    model, processor = qwen_client._load_local_hf()
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    texts: list[str] = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "Return only strict JSON."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    inputs = processor(text=texts, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=qwen_client.max_tokens,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def summarize_predicc(
    metrics_by_source_and_k: dict[str, dict[int, dict[str, Any]]],
    *,
    oracle_accuracy: float | None = None,
    shared_acc0_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute PrediCC@k and optional normalized PrediCC from accuracy rows."""

    summary: dict[str, Any] = {"sources": {}, "oracle_accuracy": oracle_accuracy}
    for source, by_k in metrics_by_source_and_k.items():
        if shared_acc0_metrics is None and 0 not in by_k:
            raise KeyError(f"source {source!r} is missing k=0 metrics")
        acc0_row = shared_acc0_metrics or by_k[0]
        acc0 = float(acc0_row["accuracy"])
        rows: list[dict[str, Any]] = []
        normalized_values: list[float] = []
        metric_rows = dict(by_k)
        if shared_acc0_metrics is not None:
            metric_rows[0] = shared_acc0_metrics
        for k in sorted(metric_rows):
            acc = float(metric_rows[k]["accuracy"])
            predicc = acc - acc0
            row = {
                "k": k,
                "accuracy": acc,
                "accuracy_percent": 100.0 * acc,
                "predicc": predicc,
                "predicc_points": 100.0 * predicc,
                "num_correct": metric_rows[k].get("num_correct"),
                "num_examples": metric_rows[k].get("num_examples"),
                "num_invalid_predictions": metric_rows[k].get("num_invalid_predictions"),
            }
            if k == 0 and shared_acc0_metrics is not None:
                row["shared_no_context"] = True
            if oracle_accuracy is not None and oracle_accuracy != acc0:
                normalized = predicc / (oracle_accuracy - acc0)
                row["normalized_predicc"] = normalized
                if k != 0:
                    normalized_values.append(normalized)
            rows.append(row)
        source_summary: dict[str, Any] = {
            "acc0": acc0,
            "shared_acc0": shared_acc0_metrics is not None,
            "rows": rows,
        }
        if normalized_values:
            source_summary["aupredicc"] = sum(normalized_values) / len(normalized_values)
        summary["sources"][source] = source_summary
    return summary


def evaluate_predicc_mcq(
    *,
    dataset_path: str | Path,
    results_csv: str | Path,
    output_dir: str | Path,
    qwen_client: QwenClient | None,
    ad_columns: list[str],
    context_lengths: list[int] | tuple[int, ...] = (0, 1, 2, 4, 8),
    parsed_sequences_dir: str | Path = "data/interim/per_movie",
    tolerance: float = 0.02,
    limit: int | None = None,
    dry_run_prompts: int = 0,
    shuffle_options: bool = True,
    oracle_accuracy: float | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Run PrediCC fixed-context MCQ evaluation for one or more AD sources."""

    all_examples = read_jsonl(dataset_path)
    if limit is not None:
        all_examples = all_examples[:limit]

    clip_index = load_sequence_clip_index(parsed_sequences_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_by_source_and_k: dict[str, dict[int, dict[str, Any]]] = {}
    unmatched_rows: list[dict[str, Any]] = []
    shared_acc0_metrics: dict[str, Any] | None = None
    context_lengths_set = {int(k) for k in context_lengths}

    if 0 in context_lengths_set:
        contexts_by_id = {example["id"]: [] for example in all_examples}
        shared_acc0_metrics = _evaluate_examples_with_contexts(
            examples=all_examples,
            contexts_by_id=contexts_by_id,
            output_dir=output_path / "no_context" / "k_0",
            qwen_client=qwen_client,
            source_name=None,
            dry_run_prompts=dry_run_prompts,
            shuffle_options=shuffle_options,
            run_name="answer_predicc_no_context_k0",
            batch_size=batch_size,
        )
        shared_acc0_metrics.update(
            {
                "dataset_path": str(dataset_path),
                "results_csv": str(results_csv),
                "ad_column": None,
                "context_length_k": 0,
                "num_total_examples": len(all_examples),
                "num_unmatched_examples": 0,
                "context_protocol": "source_neutral_no_context",
                "shared_no_context": True,
            }
        )
        (output_path / "no_context" / "k_0" / "metrics.json").write_text(
            json.dumps(shared_acc0_metrics, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    for ad_column in ad_columns:
        generated_rows = load_generated_ad_rows(results_csv, column=ad_column)
        index = build_generated_ad_index(generated_rows)
        text_index = build_ground_truth_index(generated_rows)
        metrics_by_source_and_k[ad_column] = {}

        for k in context_lengths:
            if int(k) == 0:
                if shared_acc0_metrics is None:
                    raise RuntimeError("shared no-context metrics were not computed")
                metrics_by_source_and_k[ad_column][0] = shared_acc0_metrics
                continue
            contexts_by_id: dict[str, list[str]] = {}
            evaluated_examples: list[dict[str, Any]] = []
            for example in all_examples:
                clips = fixed_prior_context(example, clip_index, k=int(k))
                generated_context, unmatched = materialize_generated_context_clips(
                    clips,
                    index,
                    text_index=text_index,
                    tolerance=tolerance,
                )
                if unmatched:
                    unmatched_rows.append(
                        {
                            "source": ad_column,
                            "k": int(k),
                            "id": example["id"],
                            "unmatched_context": unmatched,
                        }
                    )
                    continue
                contexts_by_id[example["id"]] = generated_context
                evaluated_examples.append(example)

            source_slug = ad_column.lower().replace("-", "_").replace(" ", "_")
            k_output_dir = output_path / source_slug / f"k_{int(k)}"
            metrics = _evaluate_examples_with_contexts(
                examples=evaluated_examples,
                contexts_by_id=contexts_by_id,
                output_dir=k_output_dir,
                qwen_client=qwen_client,
                source_name=ad_column,
                dry_run_prompts=dry_run_prompts,
                shuffle_options=shuffle_options,
                run_name=f"answer_predicc_{source_slug}_k{int(k)}",
                batch_size=batch_size,
            )
            metrics.update(
                {
                    "dataset_path": str(dataset_path),
                    "results_csv": str(results_csv),
                    "ad_column": ad_column,
                    "context_length_k": int(k),
                    "num_total_examples": len(all_examples),
                    "num_unmatched_examples": len(all_examples) - len(evaluated_examples),
                    "context_protocol": "fixed_last_k_prior_clips",
                }
            )
            (k_output_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            metrics_by_source_and_k[ad_column][int(k)] = metrics

    write_jsonl(output_path / "unmatched.jsonl", unmatched_rows)
    summary = summarize_predicc(
        metrics_by_source_and_k,
        oracle_accuracy=oracle_accuracy,
        shared_acc0_metrics=shared_acc0_metrics,
    )
    summary.update(
        {
            "dataset_path": str(dataset_path),
            "results_csv": str(results_csv),
            "ad_columns": ad_columns,
            "context_lengths": [int(k) for k in context_lengths],
            "parsed_sequences_dir": str(parsed_sequences_dir),
            "output_dir": str(output_path),
        }
    )
    (output_path / "predicc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def evaluate_autoad_mcq(
    *,
    dataset_path: str | Path,
    results_csv: str | Path,
    output_dir: str | Path,
    qwen_client: QwenClient | None,
    ad_column: str = "AutoAD-Zero",
    tolerance: float = 0.02,
    limit: int | None = None,
    dry_run_prompts: int = 0,
    shuffle_options: bool = True,
) -> dict[str, Any]:
    """Run or resume generated-AD MCQ evaluation."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    predictions_path = output_path / "predictions.jsonl"
    unmatched_path = output_path / "unmatched.jsonl"
    prompts_path = output_path / "prompt_preview.jsonl"
    metrics_path = output_path / "metrics.json"

    all_examples = read_jsonl(dataset_path)
    if limit is not None:
        all_examples = all_examples[:limit]
    generated_rows = load_generated_ad_rows(results_csv, column=ad_column)
    index = build_generated_ad_index(generated_rows)
    text_index = build_ground_truth_index(generated_rows)

    prediction_rows = load_prediction_rows(predictions_path)
    completed_ids = {row["id"] for row in prediction_rows}
    unmatched_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    evaluated_examples: list[dict[str, Any]] = []

    with predictions_path.open("a", encoding="utf-8") as prediction_handle:
        for example in all_examples:
            generated_context, unmatched = materialize_autoad_context(
                example,
                index,
                text_index=text_index,
                tolerance=tolerance,
            )
            if unmatched:
                unmatched_rows.append({"id": example["id"], "unmatched_context": unmatched})
                continue
            evaluated_examples.append(example)
            option_order = (
                make_option_order(example["id"], len(example["options"]))
                if shuffle_options
                else list(range(len(example["options"])))
            )
            prompt = make_mcq_prompt(
                example,
                generated_context,
                source_name=ad_column,
                option_order=option_order,
            )
            if len(prompt_rows) < dry_run_prompts:
                prompt_rows.append({"id": example["id"], "option_order": option_order, "prompt": prompt})
            if dry_run_prompts and qwen_client is None:
                continue
            if example["id"] in completed_ids:
                continue
            if qwen_client is None:
                raise RuntimeError("qwen_client is required unless dry_run_prompts is used")

            raw_text = ""
            parsed: dict[str, Any] | None = None
            error: str | None = None
            prediction_idx: int | None = None
            prediction_text: str | None = None
            displayed_prediction_idx: int | None = None
            displayed_prediction_text: str | None = None
            try:
                result = qwen_client.generate_json("answer_autoad_mcq", prompt, temperature=0.0)
                raw_text = result.raw_text
                parsed = result.parsed
                (
                    prediction_idx,
                    prediction_text,
                    displayed_prediction_idx,
                    displayed_prediction_text,
                ) = normalize_prediction(parsed, example["options"], option_order=option_order)
            except Exception as exc:  # noqa: BLE001 - record and continue evaluation.
                error = f"{type(exc).__name__}: {exc}"

            row = {
                "id": example["id"],
                "prediction_idx": prediction_idx,
                "prediction_text": prediction_text,
                "displayed_prediction_idx": displayed_prediction_idx,
                "displayed_prediction_text": displayed_prediction_text,
                "option_order": option_order,
                "displayed_options": [example["options"][idx] for idx in option_order],
                "answer_idx": example.get("answer_idx"),
                "answer_text": example.get("answer_text"),
                "displayed_answer_idx": option_order.index(example["answer_idx"])
                if isinstance(example.get("answer_idx"), int)
                else None,
                "correct": prediction_idx == example.get("answer_idx"),
                "raw_text": raw_text,
                "parsed": parsed,
                "error": error,
            }
            prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            prediction_handle.flush()
            prediction_rows.append(row)
            completed_ids.add(example["id"])

    write_jsonl(unmatched_path, unmatched_rows)
    if prompt_rows:
        write_jsonl(prompts_path, prompt_rows)
    metrics = compute_autoad_metrics(evaluated_examples, prediction_rows)
    metrics.update(
        {
            "dataset_path": str(dataset_path),
            "results_csv": str(results_csv),
            "ad_column": ad_column,
            "shuffle_options": shuffle_options,
            "num_total_examples": len(all_examples),
            "num_evaluated_examples": len(evaluated_examples),
            "num_unmatched_examples": len(unmatched_rows),
            "predictions_path": str(predictions_path),
            "unmatched_path": str(unmatched_path),
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics
