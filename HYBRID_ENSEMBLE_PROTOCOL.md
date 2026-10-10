# Pre-specified evaluation of the tree-neural hybrid ensemble

Status: written and committed **before** the confirmatory runs. Nothing below is changed after the results are seen;
any later deviation is listed at the end of this file.

## Why this test exists

An exploratory look at saved out-of-fold predictions (random seed 42) suggested that averaging Random Forest,
Gradient Boosting and the Proposed model (MLP + causal TCN + LSTM) beats each of them alone. That look compared 10
combinations on the Colombian data and 6 on the Spanish data, so the gain is optimistic. This protocol tests **one
fixed rule** on fresh data splits.

## Hypothesis

The *Tree-Neural Ensemble* (TNE), the unweighted mean of the predicted probabilities of Random Forest (`rf`),
HistGradientBoosting (`hgb`) and the Proposed model (`proposed`), scores higher on the primary metric than each of its
three components.

## Design

| Item | Specification |
| --- | --- |
| Settings (5) | Colombian pre-university (`FEATURE_SET=extended`); Colombian with university (`extended_univ`, a recommendation task); Spanish pooled sample after semester 1 (H2); Spanish pooled full year (H3); Spanish new entrants after semester 1 (H2) |
| Fresh randomness | `SEED=2024` (the exploratory look used 42): new outer folds, new inner splits, new model seeds. The datasets are the same, so this is **not** independent data |
| Outer folds | Colombian: stratified 5-fold. Spanish: 5-fold stratified, grouped by student (no student in two folds) |
| Components | Each fitted with the main study's nested protocol: inner student-disjoint hold-out, 8 trials (Colombian) or 6 trials (Spanish), search fraction 0.5 (pooled) / 1.0 (entrants), then refit on the whole training part. The ensemble itself has **no** tuning, no fitted weights, no stacking, no threshold |
| Ensemble | Mean of the three probability vectors of each test row, each component using its own tuned configuration |
| Primary metric | Spanish: PR-AUC (tuned configuration). Colombian: Macro-F1 (configuration selected on inner Macro-F1) |
| Secondary metrics | Spanish: ROC-AUC, share of dropouts in the top 10% by risk. Colombian: Top-1 and Top-3 accuracy (configurations selected on inner Top-3), balanced accuracy |
| Contrasts | TNE minus each component (3 per setting): paired t-test over the 5 folds, mean difference with 95% t-interval. Holm correction within each setting (3 tests); a conservative Holm over all 15 primary contrasts is also reported |
| Baselines | Colombian: always-the-biggest-programme (majority). Colombian with university: most common programmes at the student's university (lookup fitted on the training fold); TNE vs lookup is tested on Macro-F1 (primary) and Top-1 / Top-3 (secondary). Spanish: prevalence |

## Decision rule

* **Supported** in a setting: TNE's mean primary metric exceeds all three components **and** the within-setting
  Holm-adjusted p for the contrast against the best component is below 0.05.
* **Not supported** otherwise (a tie or a loss is reported as such).
* The overall claim "the tree-neural ensemble is the more suitable hybrid" is made only if it is supported in at least
  3 of the 5 settings and is not significantly worse than any component in any setting. All five results are reported.

## Known limits

* Five folds give low power; small effects (about 0.01 PR-AUC) may not reach significance.
* Same datasets as the exploratory look. A new fold seed removes the selection of the rule on specific folds but not
  any dataset-specific luck.
* The Colombian pre-university task carries little signal, so ties are expected there.
* The ensemble costs about three times the compute of one component.

## Files

`hybrid_ensemble_eval.py` (one setting per run), `hybrid_ensemble_collate.py` (joins the settings), outputs in
`hybrid_ensemble_<setting>/` and `hybrid_ensemble_summary/`.

## Deviations after this file was committed

None yet.
