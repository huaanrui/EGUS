# Sources and attribution

The MIT license covers original project code. External data, model checkpoints
and downloaded reference implementations retain their original terms.

## BBQ

BBQ is by Alicia Parrish, Angelica Chen, Nikita Nangia and collaborators:
*BBQ: A Hand-Built Bias Benchmark for Question Answering*, Findings of ACL 2022.

- Paper: https://aclanthology.org/2022.findings-acl.165/
- Source: https://github.com/nyu-mll/BBQ
- Pinned revision: `bea11bd97d79217245b5871acd247b9d6eb24598`
- Upstream license: Creative Commons Attribution 4.0 International.

This repository distributes identifiers, selected labels and transformation code.
The preparation script retrieves original question files and additional metadata.
It adds stereotype targets and Unknown indices, applies the recorded canonical-family
mapping, then selects fixed experimental subsets. Question wording and gold answer
labels remain the upstream values.

## FairSteer and utility data

The local baseline adapts FairSteer's detector/direction design. Our shared-gate
comparisons use a separately supervised evidence detector. Neither row reproduces
all original FairSteer experiments.

- Source: https://github.com/LiYichen99/FairSteer
- Pinned revision: `0545cdbaa71b3301030accf1b98130820e08026f`
- Paper: https://aclanthology.org/2025.findings-acl.589/

The utility panel is selected from that revision's `generate_data/mmlu.json`.
It is downloaded separately, under the original MMLU and upstream source terms.
It is not redistributed here. Full third-party source checkouts are excluded.
Historical scripts requiring them need separate upstream checkouts.

## Models

Checkpoint files are not redistributed. Cloud revisions and available upstream
weight hashes are in `reproducibility/cloud_models.json`. Downloads use official
repository identifiers through hf-mirror. Each checkpoint retains its repository's
own license and access requirements.

Models include Qwen3-1.7B, MiniCPM5-2B, Qwen2.5-3B-Instruct, Qwen3-8B,
Qwen3.5-4B and Qwen3.5-9B. Prompting and architecture details are recorded in
`docs/PROTOCOL.md`.

## Historical studies

Research records cite their datasets and baseline papers. External datasets and
downloaded reference code for earlier explorations are excluded. The BBQ reproduction
guide does not claim to restore every historical exploration.
