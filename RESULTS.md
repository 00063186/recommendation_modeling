# Results

Two evaluation setups are reported:

1. the evaluator supplied with the assignment;
2. an additional fixed global time-cutoff evaluation.

## 1. Supplied evaluator

The supplied evaluator uses a per-user leave-last-20%-out split.

- Train interactions: **396,677**
- Test interactions: **93,212**
- Users evaluated: **8,776**

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| LambdaMART | 0.0711 | 0.0936 |
| Popularity | 0.0790 | 0.0876 |

The popularity baseline achieves higher Recall@10, meaning that globally popular apps cover more of the future interacted apps.

However, LambdaMART achieves higher NDCG@10. This indicates that, when the personalized model retrieves relevant apps, it tends to rank them higher in the recommendation list.

Neither model produced empty recommendation lists.

## 2. Fixed global time-cutoff evaluation

As an additional production-oriented evaluation, I use a single global cutoff at `2026-04-23`.

Interactions before the cutoff are used for training and later interactions are held out for testing.

- Train interactions: **337,280**
- Test interactions: **152,609**
- Test users: **9,485**

| Model | Recall@10 | NDCG@10 |
|---|---:|---:|
| LambdaMART | 0.054504 | 0.097372 |
| Popularity | 0.077871 | 0.107932 |

Under the fixed-cutoff evaluation, the personalized LambdaMART model does not outperform the global popularity baseline.

One likely reason is the observed-only training formulation. The ranker mainly learns relative engagement strength among user-app pairs for which an interaction was already observed, while at inference time it must rank these apps against the full catalog.

Overall, the results suggest that popularity is a strong baseline for candidate coverage, while the personalized model shows some ability to improve ordering under the supplied evaluator.

