# Third-party code and data

EGUS code is MIT-licensed. This does not relicense model weights, datasets or external source code.

* **FairSteer**: https://github.com/LiYichen99/FairSteer, pinned revision `0545cdbaa71b3301030accf1b98130820e08026f`. Four unmodified MIT-licensed files are included in `third_party/FairSteer/`; retain the included copyright and license notice.
* **KnowBias**: https://github.com/JP-25/KnowBias, pinned revision `ccfd3daea0067707cbda6238839206c194794881`. Source and concept questions are downloaded on demand and excluded from this repository. No explicit repository license was found in the pinned snapshot. The EGUS MIT license does not apply to these downloaded materials.
* **BBQ**: https://github.com/nyu-mll/BBQ, pinned revision `bea11bd97d79217245b5871acd247b9d6eb24598`. Question text and the target metadata are fetched from their respective upstream sources. This repository stores identifiers and reconstruction manifests. Follow the upstream data terms and cite Parrish et al., *BBQ: A hand-built bias benchmark for question answering*, Findings of ACL 2022.
* **Models**: official checkpoint repositories and exact revisions are listed in `reproducibility/models.json`. Model files are downloaded separately; each checkpoint's upstream terms apply.

Author-source adapters verify checksums before execution. Downloaded author scripts are not run as standalone model-loading programs. The reproducibility manifest records every invoked source file.
