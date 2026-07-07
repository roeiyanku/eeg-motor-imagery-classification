# 10-Minute Presentation Guide

A slide-by-slide script for presenting the EEG motor-imagery project. Timed to
~10 minutes (11 slides). Each slide has: **timing**, **what to say**, **what to
show**. Numbers are the real results from the benchmark.

> Language note: written in English; can be delivered in Hebrew (matches
> `PROJECT_HISTORY.md`) — ask and I'll translate.

---

## The arc (memorize this one sentence)
> "We decode imagined movement from brain signals; classical geometry beats both
> the 2008 competition winner *and* modern deep learning on this small dataset,
> and session alignment pushed us further — a result that only makes sense once
> you understand data scarcity."

Three acts: **(1) the problem**, **(2) what won and why**, **(3) where it goes live.**

---

## Slide 1 — Title & hook · (0:00–0:45)
**Say:** "Can a computer tell which limb you're *imagining* moving, just from
electrical activity on your scalp? That's a brain-computer interface — the
non-invasive cousin of Neuralink. This project builds and benchmarks one."
**Show:** Project title, your name, a brain/EEG image or the cursor-demo GIF
(`results/cursor_demo_A03_riemann_fbcsp_vote.gif`) looping silently.

## Slide 2 — Goal & motivation · (0:45–1:45)
**Say:** "Two goals. Scientifically: beat the 2008 BCI Competition winner.
Practically: move toward a live system that turns *intent* into a *command* —
a cursor, a wheelchair, a prosthetic."
**Show:** The 4 classes with icons — left hand, right hand, feet, tongue.
**Key line:** "Every class is a different *imagined* movement — the person never
actually moves."

## Slide 3 — The data & task · (1:45–2:45)
**Say:** "BCI Competition IV Dataset 2a: 9 people, 22 EEG electrodes, 4 motor-
imagery classes. The catch — only **288 trials per person, ~72 per class.**
That number drives the whole story."
**Show:** Electrode montage + a raw EEG snippet; highlight "72 trials/class."
**Key line:** "Tiny data. Remember that."

## Slide 4 — How we measure success · (2:45–3:45)
**Say:** "We use the exact competition protocol: train on session T, test on the
held-out session E, and score **Cohen's kappa** averaged over all 9 subjects.
Kappa, not accuracy, because with 4 classes random guessing is already 25%.
The bar to beat: the 2008 winner's **kappa = 0.57.**"
**Show:** `train T → test E → kappa` diagram; a big "0.57" target line.

## Slide 5 — The contenders · (3:45–5:00)
**Say:** "We pitted two families against each other. Classical, EEG-specific
methods — CSP, Filter-Bank CSP, and Riemannian geometry — versus modern deep
nets — EEGNet, ShallowConvNet, EEG-TCNet. Everyone expects deep learning to win."
**Show:** Two columns: Classical vs Deep Learning.
**Key line (plant the twist):** "Spoiler: the expectation was wrong — and *why*
it's wrong is the interesting part."

## Slide 6 — Result 1: classical wins, beats 2008 · (5:00–6:00)
**Say:** "Riemannian tangent-space decoding hit **kappa 0.616** — clearly past
the 0.57 bar. FBCSP reproduced the winner at 0.566. And the best deep net,
ShallowConvNet, trailed at **0.503.** The neural nets lost."
**Show:** `results/benchmark_kappa.png` (bar chart with the 0.57 line).
```
riemann          0.616   ← beats 2008
fbcsp            0.566
shallow_convnet  0.503   ← best deep net, still behind
```

## Slide 7 — Result 2: ensemble + alignment → 0.670 · (6:00–7:00)
**Say:** "Two upgrades. First, a soft-vote ensemble of Riemann + FBCSP —
they make different mistakes — lifted us to **0.647.** Then the key idea:
**session alignment.** Train and test are recorded in different sessions, so the
signal *drifts*. We recenter each session's statistics to cancel that drift —
**Euclidean Alignment.** That took us to **kappa 0.670**, improving 7 of 9
subjects, including the hardest ones."
**Show:** Before/after per-subject bars, or the 0.647 → 0.670 jump.
**Key line:** "0.670 versus the winner's 0.57 — a wide margin."

## Slide 8 — Honest science: what DIDN'T work · (7:00–8:15)
**Say:** "Good science reports failures. We tried three more things and they
didn't help: a stacking meta-learner (a wash), Riemannian instead of Euclidean
alignment (an exact tie), and crop augmentation — which actually *hurt*,
dropping to 0.562. And here's the punchline: crop augmentation is the trick that
*helps* deep nets. It hurt us because a covariance matrix needs the *whole*
trial; short windows make it noisy."
**Show:** A small "tried / result" table; ✅ EA, ❌ stacking, ❌ crop, ➖ RA.
**Key line — THE insight:** "With only 72 examples per class, the method with the
right built-in assumptions wins. Deep nets have to *learn* that brain signals
live on the covariance manifold; Riemannian geometry *starts* there. **Inductive
bias beats raw capacity when data is scarce.**"

## Slide 9 — Two findings that matter beyond the number · (8:15–9:00)
**Say:** "Two takeaways with real consequences. One: **calibration must be
per-person.** One model pooled across everyone scored 0.360; per-person it was
0.647 — nearly double. Brains differ. Two: **the drift problem is real**, and
alignment is the fix — which sets up the live system."
**Show:** `pooled 0.360  vs  per-subject 0.647`.

## Slide 10 — From offline to live · (9:00–9:45)
**Say:** "The endgame is live decoding. The system calibrates on *you*, then
slides a 2-second window over your live EEG and re-decodes 8 times a second, with
smoothing and **online alignment** that tracks drift in real time — no labels
needed. That's the same adaptation idea Neuralink uses, in miniature."
**Show:** The live loop diagram (stream → window → decode → smooth → command) or
the cursor GIF.

## Slide 11 — Conclusions & future · (9:45–10:00)
**Say:** "In short: classical, EEG-aware methods beat both the 2008 winner and
modern deep learning on small data; alignment gave the best result at 0.670; and
we built the path to a live, self-calibrating decoder. Next: bigger datasets to
give deep learning a fair shot, and cross-subject transfer to cut calibration
time. Thank you."
**Show:** Three bullets: **0.670 > 0.57**, **inductive bias wins on small data**,
**live-ready**.

---

## Key numbers to have cold
| Thing | Number |
|---|---|
| 2008 winner (bar to beat) | **0.57** |
| Riemannian alone | 0.616 |
| Best deep net (ShallowConvNet) | 0.503 |
| Ensemble (Riemann + FBCSP) | 0.647 |
| **Best: + session alignment** | **0.670** |
| Trials per class | ~72 |
| Pooled vs per-subject | 0.360 vs 0.647 |
| Crop augmentation (failed) | 0.562 |

## Likely questions (prep answers)
- **"Why did deep learning lose?"** → Only ~72 trials/class. Riemannian geometry
  encodes the neuroscience (motor imagery = covariance change) as a built-in
  prior; the net must learn it from scratch and can't, on this little data.
- **"What is kappa?"** → Accuracy corrected for chance; fairer than raw accuracy
  with 4 balanced classes (chance = 25%).
- **"What is session alignment?"** → Train and test are different recordings, so
  the signal distribution shifts; we recenter each session's covariances to a
  common reference to cancel that shift. No test labels used.
- **"Could deep learning ever win?"** → Yes — with much more data (thousands of
  trials, or cross-subject pretraining) or invasive high-SNR signals. On small
  scalp-EEG, classical wins.
- **"Is this real-time?"** → The decoding loop is; a full live demo needs an EEG
  headset streaming over LSL. Software is built.
- **"Can it decode more than 4 movements / individual fingers?"** → Not from
  scalp EEG — cortical regions blur together. Fine control needs an implant.

## Delivery tips
- **One number per slide.** Don't read tables aloud; point to the bar that beats 0.57.
- **Plant the twist on Slide 5, pay it off on Slide 8** — the "deep learning
  lost, here's why" arc is your most memorable moment. Land it.
- **Lead with the picture, not the method.** Show the GIF/bar chart, then explain.
- **Rehearse Slides 6–8** (your core result + the insight); they carry the talk.
- If short on time, compress Slides 3–4 — keep 6, 7, 8 intact.
