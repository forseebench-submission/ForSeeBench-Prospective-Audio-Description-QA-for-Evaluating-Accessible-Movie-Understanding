# Anonymization

ForSeeBench targets anonymous double-blind review.

## Artifact Targets

- GitHub code artifact: `https://github.com/forseebench-submission/ForSeeBench-Prospective-Audio-Description-QA-for-Evaluating-Accessible-Movie-Understanding.git`
- Hugging Face Dataset artifact: `forseebench/forseebench`
- Optional Space: `forseebench/forseebench-reviewer-demo`

## Release Boundary

The GitHub artifact should include code, docs, configs, scripts, tests, and tiny sample Q/A files. The Hugging Face Dataset should include derived Q/A benchmark artifacts and metadata only.

Neither artifact should include author names, affiliations, personal accounts, private paths, raw source assets, old git history, `.env` files, or credentials.

## Known Risks In The Working Tree

- private absolute paths in non-release configs/scripts;
- generated caches and build products;
- nested git metadata;
- paper template email strings;
- local checkout paths;
- hosting account metadata if not configured anonymously.

Run:

```bash
python scripts/check_anonymization.py
```

Warnings are informational and should be reviewed before creating or updating anonymous artifacts.
