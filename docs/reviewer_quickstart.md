# Reviewer Quickstart

This quickstart uses only derived Q/A JSONL files. It does not require raw movie videos, movie clips, audio, subtitles, full MAD/MAD-eval assets, private paths, GPUs, or Qwen.

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

## Validate The Public No-Answer Benchmark File

```bash
python scripts/validate_dataset.py --input hf_dataset/data/qna_test.jsonl --schema public
```

Expected output:

```text
PASS hf_dataset/data/qna_test.jsonl: 787 rows validated as public schema
```

## Validate The Answer-Bearing Scoring File

```bash
python scripts/validate_dataset.py --input hf_dataset/data/qna_with_answers.jsonl --schema with_answers
```

Expected output:

```text
PASS hf_dataset/data/qna_with_answers.jsonl: 787 rows validated as with_answers schema
```

## Run Sample Scoring

```bash
python scripts/evaluate_mcq.py \
  --input hf_dataset/sample_data/sample_with_answers.jsonl \
  --predictions hf_dataset/sample_data/sample_predictions.jsonl
```

Expected output is JSON with keys such as `accuracy`, `num_examples`, `num_predictions`, `num_missing_predictions`, `num_correct`, `gold_label_distribution`, and `prediction_distribution`.

The sample predictions are only a smoke test. They are not a reported baseline.

## Regenerate The Release Q/A Files

The release-facing Q/A files are derived from the internal processed benchmark artifact:

```bash
python scripts/export_release_qna.py
```

Expected output:

```text
Wrote 787 rows to hf_dataset/data/qna_test.jsonl
Wrote 787 rows to hf_dataset/data/qna_with_answers.jsonl
```

## Release Checks

```bash
python scripts/check_anonymization.py
python scripts/check_submission_ready.py
```

Warnings from the anonymization checker are informational; they identify files that must be scrubbed or excluded from a final anonymous release.

## Full Benchmark

The full benchmark is represented as:

- `data/qna_test.jsonl`: public no-answer file for prediction;
- `data/qna_with_answers.jsonl`: answer-bearing scoring file.
