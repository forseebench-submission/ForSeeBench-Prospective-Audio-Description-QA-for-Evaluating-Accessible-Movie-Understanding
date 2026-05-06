# Hugging Face Release Plan

Target dataset repo: `forseebench/forseebench`

The Hugging Face Dataset artifact should host derived ForSeeBench Q/A benchmark artifacts and metadata only. It should not host raw MAD/MAD-eval assets or raw media.

## Final Layout

```text
README.md
schema.md
croissant_rai_notes.md
croissant_metadata_draft.json
data/
  forseebench_public.jsonl
  forseebench_with_answers.jsonl
sample_data/
  sample_public.jsonl
  sample_with_answers.jsonl
  sample_predictions.jsonl
```

`data/forseebench_public.jsonl` is the no-answer file for model prediction. `data/forseebench_with_answers.jsonl` is the answer-bearing scoring file for reproducibility. Full `data/` files should be added only after source-field redistribution review.

## Do Not Upload

- raw videos;
- movie clips;
- audio files;
- subtitles/dialogue files if restricted;
- full MAD/MAD-eval source assets;
- local paths;
- credentials or `.env` files;
- raw Qwen prompt/response logs;
- private outputs with unresolved provenance;
- files over 25MB unless explicitly reviewed.

## Upload Dry Run

```bash
python scripts/upload_hf_dataset.py --repo-id forseebench/forseebench --folder hf_dataset --dry-run
```

The dry run should list only the dataset card, schema, Croissant/RAI metadata, sample files, and final approved benchmark JSONL files when present.

## Croissant

After upload, retrieve Croissant metadata:

```bash
curl -L https://huggingface.co/api/datasets/forseebench/forseebench/croissant \
  -o croissant.json
```

Then validate JSON syntax and the current Croissant/RAI requirements.

TODO(author): complete final license, source dataset metadata, source license/access terms, provenance, synthetic-data status, sensitive-information statement, and maintenance fields.
