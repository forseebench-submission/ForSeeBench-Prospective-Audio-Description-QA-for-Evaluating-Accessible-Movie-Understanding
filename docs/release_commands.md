# Release Commands

These commands are for the author to run manually. They do not include real tokens. Do not paste or commit credentials.

## Local Checks

```bash
python scripts/check_anonymization.py
python scripts/check_submission_ready.py
python scripts/validate_dataset.py --input hf_dataset/data/qna_test.jsonl --schema public
python scripts/validate_dataset.py --input hf_dataset/data/qna_with_answers.jsonl --schema with_answers
python scripts/evaluate_mcq.py --input hf_dataset/sample_data/sample_with_answers.jsonl --predictions hf_dataset/sample_data/sample_predictions.jsonl
pytest
```

## Prepare Clean GitHub Release

```bash
python scripts/prepare_anonymous_release.py --output ../forseebench_anonymous_release
```

Review the copied and skipped file list before publishing. The script excludes raw media, archives, `.env`, caches, private-path files, large files, paper source, and nested git metadata by default.

## Initialize Anonymous Git Repository

Run these commands from the clean release directory:

```bash
cd ../forseebench_anonymous_release
git init
git config user.name "Anonymous Authors"
git config user.email "anonymous-authors@example.com"
git add .
git commit -m "Initial anonymous benchmark review artifact"
git remote add origin https://github.com/forseebench-submission/ForSeeBench-Prospective-Audio-Description-QA-for-Evaluating-Accessible-Movie-Understanding.git
git branch -M main
git push -u origin main
```

The push command is documented for the author only. This agent did not run it.

## Hugging Face Dataset Dry Run

```bash
python scripts/upload_hf_dataset.py --repo-id forseebench/forseebench --folder hf_dataset --dry-run
```

## Hugging Face Dataset Upload

Use either an existing `huggingface-cli login` session or an `HF_TOKEN` environment variable. Do not write tokens into files.

```bash
python scripts/upload_hf_dataset.py --repo-id forseebench/forseebench --folder hf_dataset --private true
```

The upload script refuses raw media extensions, secret-like files, and files over 25MB unless `--allow-large` is explicitly passed after manual review.
