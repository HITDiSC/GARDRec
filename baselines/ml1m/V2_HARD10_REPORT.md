# GARDRec v2 on the Hard10 Evaluation Set

## 1. Setup

- Model: GARDRec v2, checkpoint `/home/tx/gardrec_checkpoints/v2_ml1m.pth`.
- Data: `data/dataset/ML1M/parse_data/train_from_10000_data_id_20.json`
  (9,989 raw rows -> 9,060 with a non-empty history, 20 candidates, and a valid target).
- Split inside v2: first 8,154 samples unused at eval time, last 906 samples evaluated.
- Hard10 file: `baselines/ml1m/v2_hard10_train.json`
  (remote `/data/WY/GARDRec/baselines/ml1m/v2_hard10_train.json`),
  SHA256 `ee25021b143e1e9817d21bf53de51c1c2ac79606902a115027370765729e40d8`.
  This is the file that was actually evaluated.
- Only the last 906 rows are hardened. The first 8,154 rows are copied verbatim.
- Hard negatives are restricted to the 3,763-item GCN vocabulary so that
  `build_samples` never replaces a candidate at eval time.
- Construction diagnostics: hard similarity 0.1997 vs 0.0919 for the 10 replaced
  uniform negatives, similarity gap 0.1079, popularity-bin match 99.45%,
  residual popularity gap -0.095 training occurrences (-0.40%).
- Seed 42 is used everywhere.

## 2. Feature handling

The v2 ranker consumes two candidate-specific feature blocks.

- `interaction_features` (8 selected dims): recomputed for **all 20 slots** with
  `preprocess_retrieval_augmented_samples_patch.py`, using the unchanged
  training prefix. This is done identically for Easy and Hard10, so the
  comparison is not affected by the re-run.
- `candidate_match_features` (8 dims): the original values are retained for the
  target and the 9 retained easy negatives. Values for the 10 hard negatives are
  recomputed from the item KG with the same definition
  (`total_overlap_count`, `total_overlap_ratio`, genre/director/actor/keyword/
  writer/company overlap). On a 40-sample check against the original Easy
  features, the recomputation reproduces the overlap counts and bucket values
  on 89.9% of candidate rows; the remaining differences come from a slightly
  different KG snapshot. The per-sample denominator is taken from the stored
  Easy features so the ratio stays on the original scale.

A control run applies the same patched interaction-feature file to the
unmodified Easy data. This isolates feature-pipeline drift from the effect of
the candidate set.

## 3. Results

| Condition | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Easy, original details (as reported before) | 0.4525 | 0.8201 | 0.9459 |
| Easy, same feature pipeline as Hard10 | 0.4459 | 0.8201 | 0.9415 |
| Hard10, same feature pipeline | 0.4260 | 0.7726 | 0.9249 |

Change from Easy with the same feature pipeline:

| Metric | Easy | Hard10 | Delta | Relative |
|---|---:|---:|---:|---:|
| Top-1 | 0.4459 | 0.4260 | -0.0199 | -4.46% |
| Top-3 | 0.8201 | 0.7726 | -0.0475 | -5.79% |
| Top-5 | 0.9415 | 0.9249 | -0.0166 | -1.76% |

Change from the originally reported Easy number:

| Metric | Easy | Hard10 | Delta | Relative |
|---|---:|---:|---:|---:|
| Top-1 | 0.4525 | 0.4260 | -0.0265 | -5.85% |
| Top-3 | 0.8201 | 0.7726 | -0.0475 | -5.79% |
| Top-5 | 0.9459 | 0.9249 | -0.0210 | -2.22% |

Feature-pipeline drift alone (Easy original -> Easy patched): Top-1 -0.0066,
Top-3 0.0000, Top-5 -0.0044.

## 4. Interpretation

- v2 degrades under Hard10, but much less than SASRec (-12.1% relative Top-1)
  and BERT4Rec (-9.0%) on the same dataset family. The relative Top-1 drop is
  -4.5% to -5.9%.
- Top-3 falls most (-5.8%), which is consistent with content-similar negatives
  occupying the upper ranks rather than displacing the target completely.
- v2 remains well above the non-LLM baselines on the Easy protocol
  (0.4459 vs LightGCN 0.3790 Top-1 under the baseline 9,278-row protocol).
  The numbers are not directly comparable across protocols because the sample
  sets differ.
- For R1-5, the result supports the same conclusion as the baseline study:
  content-similar hard negatives lower absolute scores, but the proposed model
  keeps its advantage and does not collapse.

## 5. Caveats

- Hard-negative match features for 10 slots per test sample are recomputed from
  a KG snapshot that is close to, but not bit-identical with, the one used for
  the original Easy details. About 10% of recomputed rows differ in at least one
  feature. This can move the Hard10 number by a small amount.
- The v2 pipeline randomly substitutes history items that are missing from the
  GCN vocabulary (1,870 item occurrences across 1,688 rows). Histories are
  unchanged between Easy and Hard10 and the seed is fixed, so the substitution
  pattern is identical in both conditions, but it is a pre-existing property of
  the v2 input pipeline.
- This is still the 20-candidate ranking protocol, not strict leave-one-out.

## 6. Reproduction

Remote commands:

```bash
cd /data/WY/GARDRec
python baselines/build_hard_negatives_v2.py   --data data/dataset/ML1M/parse_data/train_from_10000_data_id_20.json   --domain ml1m --metadata /tmp/v2work/movies.dat   --observed-json /tmp/v2work/observed.json   --catalog-file /tmp/v2work/catalog.json   --output baselines/ml1m/v2_hard10_train.json --hard-count 10 --seed 42

python preprocess_retrieval_augmented_samples_patch.py   --input-file baselines/ml1m/v2_hard10_train.json   --retrieval-details output/ml1m/rank_save/retrieval_augmented_details_safe_interactions.json   --domain ml1m   --output-file output/ml1m/rank_save/retrieval_augmented_details_v2_hard10_patch.json

python gcn_recommend_v2.py --domain ml1m --eval_only   --model /data/WY/models/Qwen2-7B-Instruct   --input-file baselines/ml1m/v2_hard10_train.json   --retrieval-details output/ml1m/rank_save/retrieval_augmented_details_v2_hard10.json   --resume_checkpoint /home/tx/gardrec_checkpoints/v2_ml1m.pth   --max_prompt_length 384
```

Raw metrics are stored in:

- `baselines/ml1m/v2_easy_original_eval_metrics.json`
- `baselines/ml1m/v2_easy_patch_eval_metrics.json`
- `baselines/ml1m/v2_hard10_eval_metrics.json`
- `baselines/ml1m/v2_hard10_remote_stats.json`
