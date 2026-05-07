---
pretty_name: ForSeeBench
license: other
task_categories:
- question-answering
language:
- en
tags:
- audio-description
- accessibility
- benchmark
- multiple-choice-qa
- video-understanding
- prospective-qa
---

# ForSeeBench

## Dataset Summary

ForSeeBench is an evaluation benchmark for prospective audio-description question answering. Sighted viewers receive dense visual cues continuously: actions, bodies, expressions, scene changes, objects, and spatial updates all help them build a mental model of a movie. Blind and low-vision viewers often depend on audio description (AD) for that information, because the soundtrack alone does not carry it.

Each ForSeeBench item gives prior AD context and a multiple-choice question about a withheld future human-written AD target. The task is to choose the answer option that best matches the upcoming target, measuring whether the prior AD stream preserves evidence for the next salient visual update. This complements captioning, video QA, and general video-understanding evaluations, which often ask systems to describe or answer questions about visible content without directly testing whether AD supports the unfolding viewing experience.

This is an anonymous review artifact for a double-blind benchmark submission. It is not a training corpus.

| Property | Value |
| --- | --- |
| Task | Multiple-choice prospective audio-description QA |
| Size | 787 Q/A items |
| Language | English |
| Modality | Text derived from movie audio descriptions |
| Source | MAD/MAD-eval audio-description data |
| Raw media redistributed | No |

## Data

The dataset contains two JSONL files:

- `data/qna_test.jsonl`: the public no-answer benchmark file for model prediction.
- `data/qna_with_answers.jsonl`: the answer-bearing scoring file for evaluation and reproducibility.

Both files contain the same 787 ForSeeBench items. `qna_test.jsonl` contains the inputs a system should see. `qna_with_answers.jsonl` adds the answer key and clean audit fields used for scoring. These are benchmark files, not dataset splits.

The main benchmark size is 787 examples. A separate repository audit may list answer options that exactly match the hidden target AD for optional manual paraphrasing review; that audit is not a separate split and does not remove examples from the main benchmark by default.

## Figures

The release includes two paper figures for reviewers and dataset users:

- `assets/teaser.png`: overview of the prospective AD-QA task, showing that the model answers from prior AD while the future Target AD remains withheld.
- `assets/data_pipeline.png`: construction pipeline diagram, showing source AD streams, construction-time target/evidence selection, question generation, filtering, and release files.

For code, validation scripts, scoring scripts, and construction utilities, see the anonymous GitHub artifact:

```text
https://github.com/forseebench-submission/ForSeeBench-Prospective-Audio-Description-QA-for-Evaluating-Accessible-Movie-Understanding
```

## Source Data

ForSeeBench is derived from MAD/MAD-eval audio-description data. This release contains derived Q/A benchmark artifacts and metadata only.

Raw movie videos, movie clips, audio tracks, subtitles, dialogue files, full MAD/MAD-eval source assets, and other restricted source assets are not redistributed. Users who need source assets must obtain them from the original providers under their own terms.

MAD / MAD-eval citation:

```bibtex
@inproceedings{soldan2022mad,
  author = {Soldan, Mattia and Pardo, Alejandro and Alc{\'a}zar, Juan Le{\'o}n and Caba Heilbron, Fabian and Zhao, Chen and Giancola, Silvio and Ghanem, Bernard},
  title = {{MAD}: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {5026--5035},
  year = {2022}
}
```

If using the underlying Movie Description/LSMDC lineage directly, cite Rohrbach et al., *Movie Description*, IJCV 2017.

## Task And Evaluation

Systems receive prior AD context, a question, and four answer options from `data/qna_test.jsonl`. They output one JSONL prediction per example:

```json
{"id": "example-id", "prediction": 0}
```

Scoring uses `data/qna_with_answers.jsonl`:

```bash
python scripts/evaluate_mcq.py \
  --input hf_dataset/data/qna_with_answers.jsonl \
  --predictions path/to/predictions.jsonl
```

The evaluator reports multiple-choice accuracy, missing predictions, number correct, and label distributions.

For the fixed-window PrediCC evaluation in the paper code, `k=0` is a source-neutral no-context prompt containing only the question and shuffled answer options. That shared `Acc@0` baseline is reused when computing PrediCC@k for different AD sources. The hidden target AD is not supplied as evaluation context; answer options are generated from the hidden target in the multiple-choice QA construction.

## Croissant And Responsible AI Metadata

This release includes a Croissant metadata file with core and minimal Responsible AI fields:

```text
croissant.json
```

It documents the hosted JSONL resources, figures, record schemas, source-data restrictions, synthetic Q/A generation, intended uses, limitations, social-impact considerations, and provenance. The repository also includes a lightweight pre-upload checker:

```bash
python scripts/validate_croissant_metadata.py --input hf_dataset/croissant.json
```

The official Croissant validator specified by the conference submission system should also be run on the hosted metadata before OpenReview submission.

## Intended Use And Limits

ForSeeBench is intended for evaluating prospective QA from prior AD context and comparing context policies or AD sources under fixed questions and options. It should not be used as a training corpus unless a future release explicitly defines such a protocol.

The benchmark does not redistribute raw media or source datasets, does not certify complete accessibility quality, and should not be used for high-stakes deployment claims or unrestricted future-event prediction.

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

MAD / MAD-eval:

```bibtex
@inproceedings{soldan2022mad,
  author = {Soldan, Mattia and Pardo, Alejandro and Alc{\'a}zar, Juan Le{\'o}n and Caba Heilbron, Fabian and Zhao, Chen and Giancola, Silvio and Ghanem, Bernard},
  title = {{MAD}: A Scalable Dataset for Language Grounding in Videos from Movie Audio Descriptions},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = {5026--5035},
  year = {2022}
}
```

## License

For anonymous review, the derived ForSeeBench benchmark files and included figures are provided for paper review, reproducibility checking, and non-commercial research evaluation. Do not redistribute raw source assets, attempt to reconstruct restricted movie media, or use the artifact for deployment or accessibility-certification claims.

Original MAD/MAD-eval and underlying movie source assets remain governed by their original providers' terms and are not redistributed here.

See `LICENSE.md` for the review access terms included with this dataset artifact.
