# Easy vs Hard10 结果汇总

协议：1 正 + 19 原始均匀负例（Easy） vs 1 正 + 10 内容难负例 + 9 原始均匀负例（Hard10）。
训练集与验证集完全不变，仅测试候选集变化。

## ML-1M

| 模型 | Easy Top-1 | Hard10 Top-1 | Δ Top-1 | 相对变化 | Easy NDCG@5 | Hard10 NDCG@5 | Δ NDCG@5 | Easy MRR | Hard10 MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.2909 | 0.3125 | +0.0216 | +7.41% | 0.5418 | 0.5504 | +0.0086 | 0.4933 | 0.5057 |
| lightgcn | 0.3790 | 0.4314 | +0.0524 | +13.84% | 0.6439 | 0.6769 | +0.0330 | 0.5807 | 0.6173 |
| sasrec | 0.1279 | 0.1124 | -0.0154 | -12.08% | 0.2970 | 0.2764 | -0.0206 | 0.2984 | 0.2851 |
| bert4rec | 0.1239 | 0.1128 | -0.0111 | -8.99% | 0.2951 | 0.2842 | -0.0109 | 0.2997 | 0.2866 |

## ML-20M

| 模型 | Easy Top-1 | Hard10 Top-1 | Δ Top-1 | 相对变化 | Easy NDCG@5 | Hard10 NDCG@5 | Δ NDCG@5 | Easy MRR | Hard10 MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.6740 | 0.6710 | -0.0030 | -0.45% | 0.8337 | 0.8307 | -0.0030 | 0.7959 | 0.7934 |
| lightgcn | 0.7627 | 0.7487 | -0.0140 | -1.84% | 0.8698 | 0.8622 | -0.0076 | 0.8477 | 0.8384 |
| sasrec | 0.1853 | 0.1553 | -0.0300 | -16.19% | 0.4972 | 0.4689 | -0.0283 | 0.4223 | 0.3959 |
| bert4rec | 0.2560 | 0.2547 | -0.0013 | -0.52% | 0.5923 | 0.5898 | -0.0024 | 0.5085 | 0.5058 |

## Amazon-Book

| 模型 | Easy Top-1 | Hard10 Top-1 | Δ Top-1 | 相对变化 | Easy NDCG@5 | Hard10 NDCG@5 | Δ NDCG@5 | Easy MRR | Hard10 MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.4030 | 0.4000 | -0.0030 | -0.74% | 0.4898 | 0.4887 | -0.0011 | 0.5016 | 0.5001 |
| lightgcn | 0.3030 | 0.2680 | -0.0350 | -11.55% | 0.3771 | 0.3632 | -0.0139 | 0.3982 | 0.3775 |
| sasrec | 0.2380 | 0.1960 | -0.0420 | -17.65% | 0.4258 | 0.3947 | -0.0311 | 0.4029 | 0.3718 |
| bert4rec | 0.2520 | 0.2360 | -0.0160 | -6.35% | 0.4022 | 0.4043 | +0.0021 | 0.3999 | 0.3988 |

## 结果来源

- ML-1M: Easy = `baselines/ml1m/results_80_10_10/ml1m_baselines.json`; Hard10 = `baselines/ml1m/results_hard10/ml1m_hard10_baselines.json`
- ML-20M: Easy = `baselines/ml20m/results/ml20m_baselines.json`; Hard10 = `baselines/ml20m/results_hard10/ml20m_hard10_baselines.json`
- Amazon-Book: Easy = `baselines/amazonbook/results/amazonbook_baselines.json`; Hard10 = `baselines/amazonbook/results_hard10/amazonbook_hard10_baselines.json`
