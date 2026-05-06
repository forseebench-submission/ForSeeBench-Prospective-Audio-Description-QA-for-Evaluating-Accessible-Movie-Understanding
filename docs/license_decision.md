# License Decision Notes

The final code and dataset licenses are unresolved.

## Derived Benchmark Dataset

The ForSeeBench Hugging Face dataset release is intended to contain derived Q/A benchmark artifacts and metadata. The license for these derived artifacts may differ from the licenses or access terms of the raw source assets.

TODO(author): choose a final license or access policy for the derived benchmark dataset.

## Raw Source Assets

Raw source assets remain governed by their original terms. These include:

- movie videos,
- movie clips,
- raw audio,
- original subtitles/dialogue files if restricted,
- original MAD/MAD-eval assets,
- any other restricted source material.

The ForSeeBench release must not imply that those raw assets are relicensed. Users must obtain them from the original providers under the providers' own terms.

## Code

TODO(author): choose a final code license.

## Generated Outputs

Generated AD outputs and Qwen-assisted benchmark artifacts may have separate licensing considerations depending on the model, source input terms, and redistribution policy.

TODO(author): decide whether generated outputs are included in the dataset release and under what terms.

## Required Submission Action

Before OpenReview submission:

- add a root `LICENSE` or `LICENSE.md`,
- state dataset license/access terms in `hf_dataset/README.md`,
- state source-asset restrictions in the dataset card,
- confirm that released fields are allowed to be hosted.
