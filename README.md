# ForSeeBench: Prospective Audio-Description QA for Evaluating Accessible Movie Understanding

[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffcc00)](https://huggingface.co/datasets/forseebench/forseebench)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](pyproject.toml)

This repository is an anonymous review artifact for a double-blind benchmark submission.

## Overview

Sighted movie viewers receive dense visual cues continuously and without asking for them: actions, bodies, facial expressions, scene changes, object reveals, spatial layouts, and other visual updates all contribute to a viewer's mental model of the story. Blind and low-vision (BLV) viewers often depend on audio description (AD) for that information, because the soundtrack alone does not carry much of it.

Current automatic evaluation for video-language systems is usually framed as video captioning, video question answering, or general video understanding: describe the visible scene, answer a question about a clip, or recognize what is happening. Those tasks are useful, but they do not directly test whether an AD stream gives BLV viewers the kind of unfolding narrative evidence that sighted viewers receive from vision.

ForSeeBench evaluates this forward-looking requirement. Each item gives prior AD context and a multiple-choice question about a withheld future human-written AD target. A system must choose the option that best matches the upcoming target, without seeing the target sentence during prediction. This tests whether an AD stream preserves forward-relevant evidence for the next salient visual update, rather than merely describing isolated scenes after the fact.

The benchmark is instantiated from MAD-eval movie audio-description data and is released as an evaluation artifact, not a training corpus.

## Benchmark Components

ForSeeBench has three artifact components:

1. **Benchmark data**: 787 derived Q/A items released as a public no-answer prediction file and an answer-bearing scoring file.
2. **Construction pipeline**: scripts for parsing ordered AD clips, selecting future targets and prior evidence, generating Q/A items, and exporting release files.
3. **Evaluation tools**: schema validation, multiple-choice scoring, and scripts for the no-context, fixed-context, adaptive-context, and PrediCC analyses used in the paper.

| Property | Value |
| --- | --- |
| Task | Multiple-choice prospective audio-description QA |
| Size | 787 Q/A items |
| Language | English |
| Modality | Text derived from movie audio descriptions |
| Source | MAD/MAD-eval audio-description data |
| Main files | `qna_test.jsonl`, `qna_with_answers.jsonl` |
| Raw media redistributed | No |

## What The Benchmark Measures

ForSeeBench measures prospective AD question answering: whether prior descriptions support the same kind of story-model update that visual cues support for sighted viewers. A model receives:

- prior AD context;
- a question about the next hidden target;
- four answer options;
- release-safe diagnostic metadata such as question type, target type, and context length.

The main release has two full Q/A files:

- `hf_dataset/data/qna_test.jsonl`: public no-answer benchmark questions for model prediction;
- `hf_dataset/data/qna_with_answers.jsonl`: answer-bearing scoring file for evaluation and reproducibility.

This separation supports no-context baselines, adaptive selected-context evaluation, fixed-window context evaluation, and PrediCC-style context contribution analysis. In the paper, PrediCC@k compares accuracy with the last `k` prior AD clips against the no-context condition.

## Dataset Source And Boundary

ForSeeBench is derived from MAD/MAD-eval audio-description data. The released files contain derived Q/A benchmark artifacts and metadata only.

Raw movie videos, movie clips, audio tracks, subtitles, dialogue files, full MAD/MAD-eval source assets, and other restricted source assets are not redistributed. Users who need source assets must obtain them from the original providers under their own terms.

The Q/A files include derived benchmark text and release-safe identifiers needed to run the benchmark. They do not include raw media, local paths, credentials, prompt logs, or full source datasets.

## Benchmark Construction

The paper instantiates ForSeeBench on 10 MAD-eval movies and the current release contains 787 Q/A items. Construction proceeds over temporally ordered AD clips:

1. Ordered MAD-eval AD clips are converted into bounded search regions.
2. A future human-written AD target is selected and withheld.
3. Strictly prior AD context is selected as evidence.
4. Qwen-assisted generation converts the target/context pair into a four-option question.
5. Typed distractors are generated to separate the correct target-grounded answer from already-happened, unsupported, or contradictory alternatives.
6. Validation filters remove trivial, unsupported, leakage-prone, or malformed examples.

The result is a multiple-choice evaluation benchmark: models answer questions from prior AD context, and scoring is performed against the answer-bearing file.

## Evaluation In The Paper

The paper evaluates the 787-item benchmark with a Qwen2.5-VL text-only answerer. It compares:

- no-context answering;
- adaptive selected-context answering;
- fixed-window contexts with `k in {0, 1, 2, 4, 8}`;
- context sources including human AD, NarrAD, and AutoAD-Zero;
- PrediCC@k context contribution relative to no-context accuracy.

This repository provides the validation and scoring interface for the released Q/A files. Full model-running commands depend on the author’s final environment and model-access decisions.

## Released Files

Primary benchmark files:

```text
hf_dataset/data/qna_test.jsonl
hf_dataset/data/qna_with_answers.jsonl
hf_dataset/schema.md
```

The Q/A files contain 787 derived benchmark items. `qna_test.jsonl` is the prediction file; `qna_with_answers.jsonl` is the scoring/reproducibility file. They are not train/validation/test splits.

The Hugging Face Dataset artifact is:

```text
https://huggingface.co/datasets/forseebench/forseebench
```

The anonymous GitHub code artifact is:

```text
https://github.com/forseebench-submission/ForSeeBench-Prospective-Audio-Description-QA-for-Evaluating-Accessible-Movie-Understanding.git
```

## Installation

The validation and scoring scripts use the Python standard library plus this repository's `src` package.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Full model-based construction/evaluation may require additional model dependencies and access credentials; see the relevant script help text and configuration files.

## Quickstart

Validate the full public no-answer Q/A file:

```bash
python scripts/validate_dataset.py --input hf_dataset/data/qna_test.jsonl --schema public
```

Validate the full answer-bearing scoring file:

```bash
python scripts/validate_dataset.py --input hf_dataset/data/qna_with_answers.jsonl --schema with_answers
```

Score predictions:

```bash
python scripts/evaluate_mcq.py \
  --input hf_dataset/data/qna_with_answers.jsonl \
  --predictions path/to/predictions.jsonl
```

Create the full release-facing Q/A files from the internal processed benchmark artifact:

```bash
python scripts/export_release_qna.py
```

## Repository Structure

- `pyproject.toml`: editable install and minimal dependency metadata.
- `configs/`: release-safe configuration templates.
- `src/`: ForSeeBench package code.
- `scripts/`: construction, export, validation, scoring, release, and upload helpers.
- `hf_dataset/`: Hugging Face dataset card, schema, metadata draft, and full Q/A files.
- `tests/`: validation, metric, CLI, and pipeline unit tests.

## Citation

ForSeeBench:

```bibtex
@misc{forseebench2026,
  title = {ForSeeBench: Prospective Audio-Description QA for Evaluating Accessible Movie Understanding},
  author = {Anonymous Author(s)},
  year = {2026},
  note = {Anonymous benchmark submission artifact}
}
```

MAD / MAD-eval source data:

```bibtex
@inproceedings{soldan2022mad,
  author = {Soldan, Mattia and Pardo, Alejandro and Alc{\'a}zar, Juan Le{\'o}n and Caba Heilbron, Fabian and Zhao, Chen and Giancola, Silvio and Ghanem, Bernard},
  title = {{MAD}: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {5026--5035},
  year = {2022}
}
```

The MAD-eval subset is part of the MAD source-data ecosystem and traces to movie audio-description resources. If using or citing LSMDC/Movie Description lineage directly, also cite:

```bibtex
@article{rohrbach2017movie,
  author = {Rohrbach, Anna and Torabi, Atousa and Rohrbach, Marcus and Tandon, Niket and Pal, Christopher and Larochelle, Hugo and Courville, Aaron and Schiele, Bernt},
  title = {Movie Description},
  journal = {International Journal of Computer Vision},
  volume = {123},
  number = {1},
  pages = {94--120},
  year = {2017},
  doi = {10.1007/s11263-016-0987-1}
}
```

The paper evaluates context sources and model components including human AD, NarrAD, AutoAD-Zero, and Qwen2.5-VL. This repository does not redistribute NarrAD outputs, AutoAD outputs, Qwen weights, or any raw source assets.

## License And Source Terms

TODO(author): finalize the code license and derived benchmark license or access terms.

Original MAD/MAD-eval and underlying movie source assets remain governed by their original providers' terms. This repository does not redistribute those raw source assets.
