# Reviewer Quickstart

This quickstart uses only derived sample Q/A JSONL files under `hf_dataset/sample_data/`. It does not require raw movie videos, movie clips, audio, subtitles, full MAD/MAD-eval assets, private paths, GPUs, or Qwen.

Run from the repository root.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

Optional test dependency:

```bash
python -m pip install pytest
```

## Validate The Public No-Answer Sample

```bash
python scripts/validate_dataset.py --input hf_dataset/sample_data/sample_public.jsonl --schema public
```

Expected output:

```text
PASS hf_dataset/sample_data/sample_public.jsonl: 2 rows validated as public schema
```

## Validate The Answer-Bearing Sample

```bash
python scripts/validate_dataset.py --input hf_dataset/sample_data/sample_with_answers.jsonl --schema with_answers
```

Expected output:

```text
PASS hf_dataset/sample_data/sample_with_answers.jsonl: 2 rows validated as with_answers schema
```

## Run Sample Scoring

```bash
python scripts/evaluate_mcq.py \
  --input hf_dataset/sample_data/sample_with_answers.jsonl \
  --predictions hf_dataset/sample_data/sample_predictions.jsonl
```

Expected output is JSON with keys such as `accuracy`, `num_examples`, `num_predictions`, `num_missing_predictions`, `num_correct`, `gold_label_distribution`, and `prediction_distribution`.

The sample predictions are only a smoke test. They are not a reported baseline.

## Release Checks

```bash
python scripts/check_anonymization.py
python scripts/check_submission_ready.py
```

Warnings from the anonymization checker are informational; they identify files that must be scrubbed or excluded from a final anonymous release.

## Full Benchmark

The full benchmark should be hosted on Hugging Face as:

- `data/forseebench_public.jsonl`: public no-answer file for prediction;
- `data/forseebench_with_answers.jsonl`: answer-bearing scoring file.

Full files are not fabricated in this repository. They should be added only after the author confirms source-field redistribution rights.
