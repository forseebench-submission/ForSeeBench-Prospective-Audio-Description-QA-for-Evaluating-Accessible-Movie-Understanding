# Reproducibility

ForSeeBench is an evaluation benchmark, not a training corpus. Reproducibility is organized around fixed public inputs, prediction files, and an answer-bearing scoring file.

## Smoke-Test Reproducibility

```bash
export PYTHONPATH="$PWD/src"
python scripts/validate_dataset.py --input hf_dataset/sample_data/sample_public.jsonl --schema public
python scripts/validate_dataset.py --input hf_dataset/sample_data/sample_with_answers.jsonl --schema with_answers
python scripts/evaluate_mcq.py \
  --input hf_dataset/sample_data/sample_with_answers.jsonl \
  --predictions hf_dataset/sample_data/sample_predictions.jsonl
```

This path runs on CPU in a few seconds and requires no raw videos, source assets, GPUs, or Qwen.

## Full Benchmark Evaluation

Full HF files:

```text
data/qna_test.jsonl
data/qna_with_answers.jsonl
```

External systems should produce predictions against the public no-answer file:

```json
{"id": "example-id", "prediction": 0}
```

Scoring uses the answer-bearing file:

```bash
python scripts/evaluate_mcq.py \
  --input hf_dataset/data/qna_with_answers.jsonl \
  --predictions predictions.jsonl
```

The release-facing files can be regenerated locally from the processed internal benchmark file:

```bash
python scripts/export_release_qna.py
```

## Optional Reconstruction

Construction scripts can rebuild benchmark candidates from source AD rows, but raw source assets are not part of the GitHub or Hugging Face release. Users who reconstruct from original MAD/MAD-eval assets must obtain those assets from the original providers under their own terms.

## Implemented Pipeline

```bash
python scripts/01_parse_lsmdc.py --help
python scripts/02_build_windows.py --help
python scripts/02b_select_valid_contexts.py --help
python scripts/03_qwen_extract_actions.py --help
python scripts/04_generate_benchmark_examples.py --help
python scripts/05_validate_and_filter.py --help
python scripts/06_split_dataset.py --help
python scripts/07_export_dataset_card.py --help
```

The construction code includes export helpers that can produce multiple files, but the release representation is benchmark-oriented: public no-answer inputs plus answer-bearing scoring data. No official train/validation/test training split is defined.

## Determinism And Runtime

- `scripts/evaluate_mcq.py` computes exact MCQ accuracy from supplied predictions.
- Sample validation/scoring is CPU-only and should finish in seconds.
- Full Qwen-assisted construction/evaluation requires model, prompt, hardware, and dependency details that remain TODO(author).

TODO(author): add dependency pins, Qwen model revision, decoding parameters, command manifests, and output checksums for reported paper tables.
