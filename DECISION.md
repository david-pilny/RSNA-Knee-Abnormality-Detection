# Decision: what to do after 0.906

**Outcome (5 Oct 2026):** David chose option A. The forked public notebook scores **0.943** on its own; our v3/v4 member
blended in at weight 0.25 also gives 0.943, at weight 0.4 it gives 0.936. So our member currently adds nothing. The next
decision (make the member much stronger via resolution, or stop and pick final submissions) is described in `CLAUDE.md` §7.

---

Written 2 Oct 2026, after the v3 + v4 blend submission. This is the assistant's message from that evening, kept as written.
Background and all numbers are in `CLAUDE.md` (§6.5 results, §6.6 public notebooks).

---

The run was clean and the blend did what it could: 0.906 against 0.905 is inside the small gain I expected. Two models with the same architecture and the same images agree too much for a blend to give a jump.

## Where the progress came from so far

| Step | Public LB |
|---|---|
| v3 labels, five folds | 0.889 |
| v4 labels, five folds | 0.905 |
| v3 + v4 blend | 0.906 |

The label upgrade was the real gain (+0.016). Labels are now at the level other teams report, so that lever is mostly used up.

## What is left

Two levers remain, and they differ a lot in size and cost:

| Option | Expected result | Cost | Main drawback |
|---|---|---|---|
| **A. Start from the public 0.94 notebook** and add our v4 models as an extra member | about 0.94, with our models adding perhaps 0.001–0.003 | a day or two, no training | built mostly on other people's weights and label tables |
| **B. Improve our own image input** (higher effective resolution, later a larger backbone) | uncertain; maybe +0.005–0.015 per step, aimed at menisci and lateral OA | 1.5 h CPU for a new cache plus 40 min GPU per test | reaching 0.94 this way in 19 days is unlikely |

I recommend A as the base for the final submissions and B as the experiment that makes our member stronger and more different. In the public write-ups, a member that differs in labels and preprocessing was the kind of addition that helped; ours differs in both.

On the rules: the copy of the rules I saw allows external data and models that are reasonably accessible to all, and public Kaggle notebooks and datasets fit that. The remaining risk is that some public label tables were made with hosted LLMs by their authors. That is their conduct, not ours, but it is your call whether you want to build on them.

## Decision for you

Do you want me to prepare A? If yes, the first step is on your side: open "Bend the Knee to Speedy Raptors" on Kaggle and use Copy & Edit, which brings its attached datasets along. I would then write the cell that adds our v4 models to its output.

If you would rather stay with our own pipeline only, I'll start B with a centre-cropped cache, which raises the resolution on the joint without making the images larger.
