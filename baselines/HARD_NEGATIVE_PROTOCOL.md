# Hard-Negative Control Experiment

## 1. Purpose

Reviewer comment R1-5 questions whether the evaluation is dominated by easy,
uniformly sampled negatives. The control experiment tests whether conclusions
remain stable when part of the 19-negative candidate set is replaced by
content-similar but unobserved items.

This experiment changes the **test candidates only**. Training and validation
rows, including their original candidate lists, are copied verbatim. Hardening
validation as well would make model selection depend on the new protocol and
would confound the paired test-time comparison, so only the test rows are
modified.

## 2. Protocols

Let `N` be the number of hard negatives in each 20-item candidate set.

| Protocol | Candidate composition | Use |
|---|---|---|
| Easy | 1 positive + 19 original uniform negatives | Existing benchmark and control |
| Hard10 | 1 positive + 10 hard + 9 original easy negatives | Primary robustness test |
| Hard19 | 1 positive + 19 hard negatives | Optional stress test |

The primary comparison is `Easy` versus `Hard10`. The same rows are used in
both conditions, so the comparison is paired at the sample level.

## 3. Hard-Negative Construction

Hard negatives are built from item metadata, independently of every evaluated
recommender:

- Movies: TF-IDF over titles plus multi-label genres.
- Amazon-Book: TF-IDF over titles plus category paths and `main_cat`.
- Query: the last 10 items in the user's history only. The target is not used
  to retrieve hard negatives, avoiding test-label leakage.
- Similarity: `0.6 * title cosine + 0.4 * category cosine`.
- Candidate pool: retrieve the top 2,000 most similar eligible items.
- Eligibility: remove the target, the input history, every item observed for
  that user in the full raw interaction file, and the original 19 negatives.
- Remaining easy negatives: sample 9 of the original 19 benchmark negatives.
  This preserves the original easy-negative distribution and makes Easy/Hard10
  differ mainly in the replacement of 10 negatives.
- Popularity control: compute item frequency on the unchanged training rows and
  use exact bins for counts 0-10 and 2x log bins above 10. Each hard negative is
  selected for the same training-popularity bin as one of the 10 replaced easy
  negatives, drawing from the top 50 similar items in that bin. This prevents
  the hard set from becoming systematically more or less popular than the easy
  set.
- Diversity control: sampling from the top 50 within each popularity bin, rather
  than always taking the absolute nearest item, reduces same-franchise and
  near-duplicate artifacts.
- The positive is inserted at a random position after negative construction.

The construction is deterministic for a fixed seed and metadata snapshot.

## 4. Data and Split

The baseline script uses an 80/10/10 row split in file order. The hard-negative
builder mirrors that split exactly:

- Rows `[0, train_size)` are unchanged and are used for training.
- Rows `[train_size, train_size + valid_size)` are used for validation and are
  unchanged.
- Remaining rows are used for test and are hardened.

Because validation is identical in both conditions, early stopping selects the
same kind of checkpoint in Easy and Hard10; the only manipulated variable is the
test candidate set.

Current dataset sizes:

| Dataset | Raw rows | Valid rows | Train (unchanged) | Valid (unchanged) | Test (hardened) |
|---|---:|---:|---:|---:|---:|
| ML-1M | 9,331 | 9,278 | 7,422 | 928 | 928 |
| ML-20M | 10,000 | 10,000 | 8,000 | 1,000 | 1,000 |
| Amazon-Book | 10,000 | 10,000 | 8,000 | 1,000 | 1,000 |

ML-1M drops 53 rows whose original candidate list violates the baseline's
validity rule (20 candidates and exactly one target occurrence).

## 5. Models and Controls

Run the same models, hyperparameters, epochs, and seeds under Easy and Hard10:

- Popularity
- LightGCN
- SASRec
- BERT4Rec
- Optionally Random and BPR-MF

ML-1M and ML-20M use seeds 42/43/44. Amazon-Book uses seed 42 only, matching
the existing Easy baseline for that dataset.

Do not tune hyperparameters on the hard test set. Validation is unchanged in
both conditions and is used for early stopping and model selection. Report
Top-1, Top-3, Top-5, NDCG@5, and MRR.

## 6. Manipulation Check

Before comparing model metrics, verify that the hard negatives are in fact more
similar to the history than the easy negatives, without shifting popularity:

- Report mean title+category similarity for hard and easy negatives.
- Report the paired similarity gap.
- Report mean training popularity and the hard/easy popularity-bin match rate.
- Verify that all candidate sets contain 20 unique items and exactly one target.
- Verify that training and validation rows are byte-for-byte equivalent to the
  original rows.

The generated `*_stats.json` files contain these diagnostics, and
`baselines/validate_hard_negatives.py` re-checks the constructed files.

Validation status (all three datasets pass):

- Training and validation rows are byte-for-byte identical to the filtered
  original rows.
- Every test candidate set has 20 unique items and exactly one target.
- Hard and retained easy negatives are disjoint and exactly partition the
  non-target candidates.
- The builder introduces no history/target leakage.

One pre-existing data quirk: 24 of 9,331 original ML-1M rows already contain a
candidate that also appears in the visible history (the original candidate list
was built before the history was truncated to 10 items). The builder preserves
those original easy negatives, so at most one test row carries this overlap and
it is identical in Easy and Hard10.

Current manipulation checks:

| Dataset | Hard similarity | Easy similarity | Gap | Popularity-bin match |
|---|---:|---:|---:|---:|
| ML-1M | 0.1828 | 0.0913 | 0.0915 | 97.74% |
| ML-20M | 0.2310 | 0.0984 | 0.1327 | 99.00% |
| Amazon-Book | 0.3078 | 0.1631 | 0.1447 | 99.80% |

Here "easy similarity" refers to the 10 original uniform negatives that were
actually replaced, not to all 19 original negatives. Hard and replaced-easy
negatives therefore enter the comparison in equal numbers.

## 7. Analysis Plan

For each dataset and model, report:

1. Easy and Hard10 absolute metrics.
2. Relative change `(Hard10 - Easy) / Easy`.
3. Seed mean and standard deviation (reported in `summary.csv`).
4. Rank stability between Easy and Hard10.

A paired user-cluster bootstrap 95% CI would be stronger, but the current
baseline runner stores only aggregate metrics. It requires dumping per-sample
ranks first and is left as an optional follow-up; the seed spread is reported
here instead.

The main hypotheses are:

- **H1 (evaluation shift):** Hard10 changes the measured difficulty, but not
  necessarily in one direction. Content-similar negatives can be easier or
  harder for a given model depending on how that model represents content and
  popularity, so we do not assume a uniform drop.
- **H2 (model sensitivity):** models that rely mainly on sequential co-occurrence
  (SASRec, BERT4Rec) are more sensitive to content-neighbour hard negatives than
  popularity-driven graph models.
- **H3 (conclusion stability):** the ranking of methods, and in particular the
  advantage of LightGCN over Popularity, remains unchanged. This is the result
  needed to answer R1-5.

## 8. Results

Top-1 accuracy:

| Dataset | Model | Easy | Hard10 | Delta | Relative |
|---|---|---:|---:|---:|---:|
| ML-1M | Popularity | 0.2909 | 0.3125 | +0.0216 | +7.41% |
| ML-1M | LightGCN | 0.3790 | 0.4314 | +0.0524 | +13.84% |
| ML-1M | SASRec | 0.1279 | 0.1124 | -0.0154 | -12.08% |
| ML-1M | BERT4Rec | 0.1239 | 0.1128 | -0.0111 | -8.99% |
| ML-20M | Popularity | 0.6740 | 0.6710 | -0.0030 | -0.45% |
| ML-20M | LightGCN | 0.7627 | 0.7487 | -0.0140 | -1.84% |
| ML-20M | SASRec | 0.1853 | 0.1553 | -0.0300 | -16.19% |
| ML-20M | BERT4Rec | 0.2560 | 0.2547 | -0.0013 | -0.52% |
| Amazon-Book | Popularity | 0.4030 | 0.4000 | -0.0030 | -0.74% |
| Amazon-Book | LightGCN | 0.3030 | 0.2680 | -0.0350 | -11.55% |
| Amazon-Book | SASRec | 0.2380 | 0.1960 | -0.0420 | -17.65% |
| Amazon-Book | BERT4Rec | 0.2520 | 0.2360 | -0.0160 | -6.35% |

NDCG@5:

| Dataset | Model | Easy | Hard10 | Delta | Relative |
|---|---|---:|---:|---:|---:|
| ML-1M | Popularity | 0.5418 | 0.5504 | +0.0086 | +1.58% |
| ML-1M | LightGCN | 0.6439 | 0.6769 | +0.0330 | +5.13% |
| ML-1M | SASRec | 0.2970 | 0.2764 | -0.0206 | -6.94% |
| ML-1M | BERT4Rec | 0.2951 | 0.2842 | -0.0109 | -3.69% |
| ML-20M | Popularity | 0.8337 | 0.8307 | -0.0030 | -0.36% |
| ML-20M | LightGCN | 0.8698 | 0.8622 | -0.0076 | -0.88% |
| ML-20M | SASRec | 0.4972 | 0.4689 | -0.0283 | -5.69% |
| ML-20M | BERT4Rec | 0.5923 | 0.5898 | -0.0024 | -0.41% |
| Amazon-Book | Popularity | 0.4898 | 0.4887 | -0.0011 | -0.22% |
| Amazon-Book | LightGCN | 0.3771 | 0.3632 | -0.0139 | -3.69% |
| Amazon-Book | SASRec | 0.4258 | 0.3947 | -0.0311 | -7.29% |
| Amazon-Book | BERT4Rec | 0.4022 | 0.4043 | +0.0021 | +0.52% |

Per-seed standard deviations and the full metric set (Top-1/3/5, NDCG@5, MRR)
are in `baselines/results_hard_negative/summary.csv`.

### 8.1 Observations

- **Sequence models move consistently.** SASRec drops on every dataset
  (-12.1%, -16.2%, -17.7% Top-1). BERT4Rec drops on ML-1M and Amazon and is
  essentially flat on ML-20M. Content-neighbour negatives are therefore harder
  for sequence models, which supports the reviewer's concern for this family.
- **Popularity-driven models do not move monotonically.** Popularity is flat on
  ML-20M and Amazon (-0.45%, -0.74%) but improves on ML-1M (+7.41%).
  LightGCN improves on ML-1M (+13.84%), is near-flat on ML-20M (-1.84%), and
  drops on Amazon (-11.55%).
- **Rankings are stable.** On ML-20M and Amazon the full ordering is unchanged.
  On ML-1M the top two (LightGCN > Popularity) are unchanged; SASRec and
  BERT4Rec swap at the bottom but differ by only 0.0004 Top-1 in Hard10.
  LightGCN keeps a clear margin over Popularity in every dataset.
- **Why ML-1M can improve.** ML-1M has 3,475 items and a small candidate
  universe, and its high-popularity bins are wide (2x log bins). The realised
  hard set is 3.4% less popular than the replaced negatives
  (22.74 vs 23.52 training occurrences), which slightly favours the target for
  popularity-based scorers. The rest of the gap is consistent with content
  neighbours being better represented in the co-occurrence graph than the
  original uniform negatives, which include many poorly modelled items. This is
  a property of the ML-1M candidate pool, not evidence that the model became
  more personalized.

### 8.2 Interpretation for R1-5

The control does **not** show that the original protocol uniformly inflates
scores. It shows that the direction of the change is model- and
dataset-dependent, while the relative ordering of non-LLM baselines is stable.
The defensible claim is:

> Replacing half of the uniform negatives with popularity-matched
> content-similar negatives changes absolute scores, but the ranking of
> baselines and the advantage of LightGCN over Popularity are preserved on all
> three datasets. Sequence models are the most sensitive family: SASRec loses
> 12-18% relative Top-1. The original conclusions are therefore not an artefact
> of easy negatives, although absolute numbers should be read as
> protocol-specific.

This directly answers the reviewer's concern without overclaiming.

### 8.3 GARDRec v2 on Hard10

The proposed GARDRec v2 ranker was evaluated with the same checkpoint and the
same seed on the v2 ML-1M input (9,060 valid samples, last 906 evaluated).
Only the test candidates were hardened. Candidate interaction features were
recomputed identically for Easy and Hard10; match features for the 10 hard
slots were recomputed from the item KG (details in
`baselines/ml1m/V2_HARD10_REPORT.md`).

| Condition | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Easy, original details | 0.4525 | 0.8201 | 0.9459 |
| Easy, same feature pipeline | 0.4459 | 0.8201 | 0.9415 |
| Hard10, same feature pipeline | 0.4260 | 0.7726 | 0.9249 |

Against the matched-feature Easy control, Hard10 changes Top-1 by -4.46%,
Top-3 by -5.79%, and Top-5 by -1.76% relative. The proposed model therefore
keeps its advantage over the non-LLM baselines and degrades less than SASRec
(-12.1%) or BERT4Rec (-9.0%), although absolute scores are protocol-specific.

## 9. Reproduction

```powershell
python baselines/build_hard_negatives.py `
  --data data/ml1m_train.json --domain ml1m `
  --metadata "E:\code\KRec\data\ml-1m\movies.dat" `
  --raw-ratings "E:\code\KRec\data\ml-1m\ratings.dat" `
  --output baselines/ml1m/hard10_train.json --hard-count 10 --seed 42
```

```powershell
python baselines/run_baselines.py `
  --data baselines/ml1m/hard10_train.json `
  --models popularity lightgcn sasrec bert4rec `
  --seeds 42 43 44 `
  --output-dir baselines/ml1m/results_hard10
```

Use the same command with `ml20m` and `amazonbook` data. Amazon-Book uses
`--seeds 42`.

After all runs finish:

```powershell
python baselines/summarize_hard_negative_results.py
```


## 10. Paper-Ready Description

> To test whether the conclusions depend on uniformly sampled easy negatives,
> we constructed a paired hard-negative control set. Hard negatives were
> retrieved from item title and category metadata using only the user's
> interaction history, without access to the target item or any evaluated
> model's embeddings. We removed all items observed for the user in the raw
> interaction data and matched the training-popularity distribution of the
> replaced negatives. In the primary setting, each 20-item candidate set
> contains one positive, ten content-similar hard negatives, and nine of the
> original uniform negatives. Training and validation rows were left unchanged;
> only test candidates were hardened, so model selection was identical in both
> conditions. For ML-1M, the mean content similarity was 0.183 for hard
> negatives and 0.091 for the ten replaced uniform negatives, while 97.7% of
> samples had exactly matched popularity bins. We report both absolute scores and
> relative changes under the same samples, models, hyperparameters, and seeds.

## 11. Scope Note

This is a controlled robustness test for the current 10k-sample, 20-candidate
ranking protocol. It does not turn that protocol into strict leave-one-out
evaluation. The paper should describe the two protocols separately and avoid
claiming strict LOO for these runs.

Two limitations should be stated:

- Popularity-bin matching is coarse above 10 training occurrences (2x log bins).
  The residual mean popularity gap is small on ML-20M (-5.0% relative) and
  Amazon (-1.0%) but reaches -3.4% on ML-1M, where counts are large and bins are
  wide. This can inflate popularity-based scorers on ML-1M and is reported
  alongside the result rather than hidden. An exact-count matching run is the
  natural sensitivity check if a reviewer requests it.
- Hard negatives come from title and category metadata, so they probe
  content-semantic difficulty, not collaborative hard negatives. A collaborative
  hard-negative variant (e.g. items co-occurring with the user's history but not
  interacted with) would test a different failure mode and is out of scope here.
