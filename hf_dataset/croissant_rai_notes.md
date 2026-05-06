# Croissant And Responsible AI Metadata Notes

## Hugging Face Auto-Generated Croissant

Hugging Face can generate core Croissant JSON-LD for dataset repositories that the Dataset Viewer can process, especially common formats such as JSONL, CSV, or Parquet.

After upload, retrieve the generated Croissant file at:

```text
https://huggingface.co/api/datasets/<namespace>/<dataset_name>/croissant
```

Planned target:

```text
https://huggingface.co/api/datasets/forseebench/forseebench/croissant
```

TODO(author): replace with the final anonymous review dataset namespace if different.

## NeurIPS-Style RAI Requirement

Dataset submissions need Croissant metadata with Responsible AI fields grounded in the actual hosted files and construction process. Hugging Face may populate structural fields, but authors usually must manually complete RAI content.

Expected RAI topics:

- data limitations;
- data biases;
- sensitive or personal information;
- intended and out-of-scope use cases;
- social impact;
- synthetic data status;
- source datasets;
- provenance, annotation, preprocessing, and generation activities;
- maintenance and access terms.

## Current Draft Files

This repository includes:

- `hf_dataset/croissant_metadata_draft.json`: conservative hand-written draft.
- `hf_dataset/README.md`: draft Hugging Face dataset card.

The draft intentionally uses `TODO(author)` placeholders for unresolved facts.

## Remaining TODO(author) Fields

- final dataset URL and version;
- final code and dataset licenses/access terms;
- source dataset names, versions, URLs, and license/access terms;
- whether derived AD text, source IDs, evidence spans, and generated outputs can be publicly hosted;
- synthetic-data classification for Qwen-assisted questions/options and generated AD outputs;
- exact provenance records for parsing, Qwen curation, validation, filtering, splitting, and export;
- sensitive-information statement for movie-derived text and any source references;
- intended and out-of-scope use statements aligned with final release files;
- maintenance/contact policy after double-blind review.

## Validation Steps

1. Upload final derived Q/A files to the anonymous Hugging Face Dataset repository. The intended full files are `data/forseebench_public.jsonl` and `data/forseebench_with_answers.jsonl`; sample files alone are sufficient only for staging.
2. Confirm the Dataset Viewer can read the JSONL files.
3. Retrieve Croissant:

```bash
curl -L https://huggingface.co/api/datasets/forseebench/forseebench/croissant \
  -o croissant.json
```

4. Syntax-check JSON:

```bash
python -m json.tool croissant.json >/tmp/forseebench_croissant_pretty.json
```

5. Validate against the current MLCommons Croissant validator or the validator specified by the conference submission system.

TODO(author): add the exact validator command required by final submission guidance.

6. Manually inspect RAI fields. Auto-generation may not infer source restrictions, derived-field policies, sensitive-information statements, or social-impact framing. It also will not decide whether hidden target AD or evidence spans are legally safe to host.

## Known Risk

HF auto-generation may produce valid structural Croissant metadata but incomplete RAI metadata. The final submission should host or attach a Croissant file with manually completed RAI fields.
