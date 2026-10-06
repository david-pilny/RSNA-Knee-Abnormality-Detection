# RSNA Knee: options to improve our model

Oct 6, 2026 · @David Pilný

## Where we stand

Our own model scores 0.918 on the public leaderboard, but it does not yet lift the public fork (0.943). Better labels and the knee crop gave the big gains; a higher resolution (320 px) gave almost nothing, so the next gains must come from more information or better use of it, not sharper images.

| Step | Own public LB | Fork + our member (weight 0.25 / 0.4) |
| --- | --- | --- |
| v4 report labels, 224 px, EfficientNet-B0 | 0.905–0.906 | 0.943 / 0.936 |
| Knee crop 140 mm, 256 px, B0 + ConvNeXt-Nano | 0.918 | 0.943 / 0.939 |
| 320 px, 130 mm (fold 0 only) | not submitted: +0.002 to +0.004 validation AUC | – |

- Fold-0 baselines for every new experiment (validation macro AUC with mirror TTA, 903 studies): B0 0.8607, ConvNeXt-Nano 0.8568.
- To clearly help the fork, our member probably needs about 0.925–0.93 on its own.
- GPU quota: about 19 h left this week; the weekly reset should add about 30 h before the final deadline of 22 Oct 2026.
- Nothing tried here risks what we have: the two candidate final submissions (plain fork, fork + our member at 0.25) are already scored, both 0.943.

## Options, ranked by expected value per GPU hour

Option 1 adds the most new information; option 2 is the cheapest test and something no public pipeline does. Others report that backbone swaps on the same input give almost nothing (+0.001), while new labels or input pipelines do help.

| # | Option | What changes | Why it could help | Cost of the fold-0 test |
| --- | --- | --- | --- | --- |
| 1 | 5 series instead of 3 | Add sagittal PD/T1 without fat suppression and a coronal T1 to the three fluid-sensitive series | Meniscal tears and cartilage are read on non-fat-suppressed PD, fracture lines on T1; our model has never seen these series. The strongest public models use 5 such slots. | New cache in 2 parts (CPU only) + about 3.5 GPU h |
| 2 | Better head and richer targets | A small transformer over all slices and series instead of pooling each series alone; extra targets from the labeller's full answer distribution (P/M/U/A/N in `llm_probs_v4.csv`) | Lets the model combine views (a tear seen on sagittal and coronal); "unclear" and "below threshold" answers carry information the single soft label discards | About 45 min (B0, fold 0) |
| 3 | Bigger backbone | ConvNeXt-Tiny at 256 px (needs gradient checkpointing, 1.29 s/step) | Stronger own model; adds little diversity to the fork | About 1.5 h |
| 4 | Final models on all data | Train the winning setup on all 4,349 studies instead of 5 × 80 % | Typically +0.002 to +0.005; a finishing step, not an experiment | Same as a five-fold run |

## Not recommended

- **A 3D CNN.** A genuinely different pipeline, but without good pretrained weights it is usually clearly weaker than our 2.5D approach with ImageNet-pretrained 2D backbones.
- **Per-finding weights tuned on the public leaderboard.** The top public forks do this, but it is leaderboard probing and exactly how they overfit; the private leaderboard decides.

## Schedule and decision rules

About 5 GPU hours of tests decide what goes into the full run; that leaves 14 h or more for it this week.

&#91;embedded content: experiment plan · 4 steps, 2 outcomes\]

Steps 1 and 2 run at the same time. Every fold-0 test is compared with B0 0.8607 and ConvNeXt-Nano 0.8568; a gain under about +0.005 counts as noise. Whatever happens, the final picks can be made from submissions already scored.
