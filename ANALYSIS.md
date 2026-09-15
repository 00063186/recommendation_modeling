Written Analysis
5.1 Target and signal definition
I treat all logged app events as implicit positive signals, but not as equally strong evidence of engagement. For each observed (user_id, app_id) pair I aggregate event counts into a heuristic engagement score:

0.10*open + 0.15*view + 0.20*click + 0.20*install + 0.30*registration + 0.40*level_achieved + 0.60*purchase.

The continuous score is converted into eight ordinal relevance grades for LambdaMART. af_view counts as a weak positive rather than a negative: in the data it behaves like a logged in-app view event and often has a short non-zero duration, but it is less convincing evidence of durable preference than deeper post-install events. The weights are deliberately heuristic and are used only to define ordering, not calibrated utility.

I do not label every unobserved user-app pair as a negative. An unobserved pair may simply mean that the app was never exposed to that user. Training therefore uses observed pairs only and extrapolates the learned user/app feature relationships to the full catalog at inference. The cost is important: the model learns engagement strength conditional on an observed interaction more directly than it learns the probability of an interaction occurring. This is a likely reason that the simple popularity baseline remains strong. A next iteration would test weak sampled pseudo-negatives or an exposure-aware objective if reliable exposure logs were available.
5.2 Model choice and the one I rejected
I chose LambdaMART (LGBMRanker) because the product output is explicitly a ranked list, the catalog is small enough (~500 apps) to score every app directly, and the dataset contains useful side information for both users and apps. The model can combine profile attributes (device, country, language, age bucket), app metadata (category, rating, size, status), and train-window behavioral aggregates. This is a better match for the available data than treating recommendation as a pure ID-only problem.

I deliberately did not use standard implicit ALS as the main model. ALS is a natural implicit-feedback baseline, but the standard formulation relies mostly on the user-item interaction matrix and does not directly use the rich user/app side features. It is also weaker for users with little or no interaction history. A hybrid factorization model could address this, but it would add complexity beyond the intended scope of the exercise.
5.3 Critique the evaluation
The main offline metrics are Recall@10 and binary NDCG@10. They are useful for checking whether future interacted apps appear near the top of the list, but they do not measure business value: a future view and a future purchase are equally relevant in the supplied evaluator. They also inherit historical exposure bias: an app that is absent from the future log may be irrelevant, or it may simply never have been shown.

For the reported development evaluation I use a single global time cutoff (2026-04-23): all earlier interactions are training data and later interactions are test data. This avoids one risk in the supplied per-user leave-last-20% split, where one user's training events can occur chronologically after another user's test events. Even with a fixed cutoff, offline logs remain policy-biased and repeated interactions with already-known apps can dominate the metric.

I would not use these numbers alone to approve a production rollout. Under the supplied evaluator, LambdaMART achieved Recall@10 = **0.0711** and NDCG@10 = **0.0936**, while the popularity baseline achieved **0.0790** and **0.0876** respectively over 8,776 users. Popularity therefore achieved better coverage, while LambdaMART achieved better ranking quality according to NDCG. I additionally evaluated the models using a single global cutoff at `2026-04-23`. Under this split, LambdaMART achieved Recall@10 = **0.0545** and NDCG@10 = **0.0974**, while popularity achieved **0.0779** and **0.1079**. The personalized model therefore did not consistently outperform the simple baseline, so I would treat it as a modeling experiment rather than a production-ready replacement.
5.4 Hyperparameters: predict, don't list
num_leaves controls tree complexity. Increasing it from a very small value should initially improve offline NDCG because the model can represent more user-app interactions. Beyond a point, I expect validation quality to plateau or decline as the model fits narrow combinations of categorical attributes and noisy behavioral aggregates. Too few leaves underfit; too many create brittle recommendations that may look good offline but generalize poorly when traffic or app mix changes.

n_estimators controls how many boosting trees are added. With learning rate fixed at 0.05, increasing the number of trees should improve offline ranking at first and then plateau; eventually it can overfit weak patterns. Too few trees leave systematic ranking errors unresolved. Too many increase latency/model size and can amplify noise without producing better real recommendation quality.
5.5 From offline to online
The current model does not beat popularity offline, so I would not run a shipping A/B test yet. If a future personalized candidate beats the baseline robustly offline, I would test the following hypothesis: personalized ranking increases meaningful app engagement relative to the current popularity policy.

The primary metric would be meaningful engagement rate per recommendation session, counting a recommendation as successful when it leads to an install/open/registration/level/purchase event within the agreed attribution window; I would not use views alone as the primary success event. Randomization would be sticky at user_id so a user sees one policy consistently. Sample size would be based on the baseline engagement rate, a pre-specified minimum detectable relative lift, two-sided alpha 0.05, and 80% power, with the test covering at least a full weekly cycle.

Two guardrails would be (1) recommendation-serving latency/error rate and (2) downstream retention or short-lived/uninstall behavior, to ensure engagement is not bought at the cost of experience quality. The biggest validity threat is contamination/exposure outside the experiment: users can encounter the same apps through other campaigns or devices, making downstream events difficult to attribute causally to the assigned recommender.
Data-Quality Findings
Finding
How found
Treatment
Reason
Some interaction user_ids do not have a matching profile row
Left joins produced missing user-profile fields for some IDs
Flagged; model uses missing-value handling and unknown-user popularity fallback
Dropping these interactions would remove real behavior; inventing profile values would be worse
Negative duration_seconds values exist
Range inspection during EDA
Flagged, not silently corrected in the submitted baseline
Exact upstream semantics are undocumented; production follow-up should confirm whether to null/repair them
af_view semantics are weaker/less certain than deeper events
Event distributions, durations, attribution combinations
Kept as a low-weight positive
Excluding it would discard observed behavior; treating it like purchase would overstate intent
status includes active/paused/blocked but snapshot timing is not documented
apps.status frequency inspection
Used as a feature, not a hard serving filter
A current status snapshot may not describe historical eligibility during the interaction window
No explicit negative preference labels
Interaction schema
Unobserved pairs excluded from training rather than called true negatives
Missing interaction can be lack of exposure, not dislike



