# ForSeeBench Benchmark Card

## Benchmark Summary

ForSeeBench is a prospective audio-description QA benchmark. Each item asks a multiple-choice question about a hidden future AD target using only prior context.

The implemented v1 artifact is derived from MAD/MAD-eval-style movie audio-description data. The release-facing artifact should contain derived Q/A benchmark records and metadata only.

## Intended Use

- Evaluate whether prior AD context supports inference about an imminent AD target.
- Compare context policies or AD sources under fixed question/option sets.
- Analyze errors through target type, evidence, and distractor metadata where those fields are included.
- Support anonymous review of the benchmark artifact.

## Out-of-Scope Use

- Certifying full accessibility quality or blind/low-vision viewer comprehension.
- High-stakes deployment claims.
- Long-horizon planning or unrestricted future forecasting.
- Training on restricted source media without respecting source licenses.
- Treating the hidden AD target as the only valid future in open-ended settings.
- Redistributing raw movie videos, clips, audio, subtitles/dialogue, or restricted source assets.

## Data And Modalities

The release-facing rows are text/metadata records derived from temporally ordered movie AD. The full project framing references video and AD context, but the GitHub quickstart and HF release plan use Q/A JSONL and metadata only.

TODO(author): clarify whether final hosted files include source IDs/timestamps, text-only derived records, or additional audit metadata.

## Evaluation

Primary implemented metric:

- Multiple-choice accuracy from `answer_idx`.

Additional implemented analyses:

- Generated-AD context evaluation with Qwen answerer.
- Fixed-context and PrediCC-style context-scaling summaries.

## Source-Data Constraints

Raw source assets are not part of the release. The artifact must not upload raw movie videos, movie clips, audio tracks, restricted subtitles/dialogue files, full MAD/MAD-eval assets, or private source paths.

Users who need raw source assets must obtain them from the original providers under their own terms. The ForSeeBench release should cite source datasets and models without redistributing restricted materials.

TODO(author): confirm which derived text fields, identifiers, generated outputs, and audit metadata can be publicly distributed.

## Limitations

- Current curation is automated and Qwen-assisted.
- No human or blind/low-vision validation is included in the inspected v1 artifacts.
- Source data are movie AD and may reflect narrative, genre, casting, and description-style biases.
- Generated-AD evaluations depend on a Qwen answerer and prompt/model configuration.
- License and redistribution terms for derived AD text require author/legal confirmation.

## Ethical Considerations

The benchmark is motivated by evaluating forward-relevant information in AD streams, but benchmark accuracy should not be interpreted as evidence of real viewer comprehension or safety. Misuse risks include overclaiming accessibility quality, surveillance-style anticipation, and deploying predictive systems without user-centered validation.

## Privacy And Security

The current sample rows contain movie-derived descriptions and character/person references. Raw media is not included in the release scaffold.

TODO(author): complete source-dataset sensitive-information and redistribution review in the final HF card and Croissant RAI metadata.

## Maintenance Plan

TODO(author): specify anonymous review contact policy and post-review maintainer/contact details.

TODO(author): define release versioning, issue reporting, leaderboard/submission policy if any, and expected support period.

## Current Status

Current assessment: PARTIAL. The implementation exists, but reviewer-facing release, license, hosting, Croissant/RAI metadata, and anonymization cleanup remain active blockers.
