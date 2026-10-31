# [Adaptive-QuantRL] Phase 7 & 8: 최종 Test 검증 및 4대 연구 질문(RQ) 보고서

## 1. 연구 범위, 한계 및 Test 구간(2023~2024) 시장 환경
- **평가 구간**: `2023-01-01 ~ 2024-12-31` (배당 현금흐름 제외 비수정 SPY 가격 기준, 편도 기본 비용 `0.1%`)
- **SPY (Buy & Hold) 기준 성과**: 누적 수익률 `52.80%`, CAGR `23.77%`, 연 변동성 `12.88%`, Sharpe `1.7212`, MDD `10.29%`
- **한계 명시**: 본 결과는 배당을 제외한 가격 수익률 기준이며, 부분 관측 환경(POMDP) 및 단일 자산(SPY) 2023~2024 상승장 국면에 기반하므로 타 자산이나 표본이 부족한 국면에 무분별하게 일반화하지 않는다.

### Test 구간 ADX(14) 국면별 표본 크기 (거래일 수 $N$)
| adx_threshold | regime | n_days | proportion(%) | sufficient_sample(N>=30) |
| --- | --- | --- | --- | --- |
| 20.0 | Strong_Trend | 332 | 66.27 | True |
| 20.0 | Weak_Trend | 169 | 33.73 | True |
| 25.0 | Strong_Trend | 216 | 43.11 | True |
| 25.0 | Weak_Trend | 285 | 56.89 | True |
| 30.0 | Strong_Trend | 128 | 25.55 | True |
| 30.0 | Weak_Trend | 373 | 74.45 | True |

---

## 2.RQ별 핵심 검증 결과

### RQ 1: 전략적 차이 (강화학습 동적 비중 조절 vs 4대 전통적 벤치마크)
| Category | Model | mean_cum_return(%) | mean_cagr(%) | mean_vol(%) | mean_sharpe | std_sharpe | mean_mdd(%) | mean_stock_weight(%) | mean_cost($) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Traditional_Benchmark | 1_Buy_and_Hold | 52.8 | 23.77 | 12.88 | 1.7212 | 0.0 | 10.29 | 100.0 | 10.0 |
| Traditional_Benchmark | 2_Cash_100 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Traditional_Benchmark | 3_Fixed_50_50 | 24.05 | 11.45 | 6.43 | 1.7171 | 0.0 | 5.25 | 50.0 | 13.73 |
| Traditional_Benchmark | 4_Moving_Average_20_50 | 22.94 | 10.95 | 10.73 | 1.0219 | 0.0 | 9.39 | 73.65 | 95.86 |
| Exposure_Matched_Baseline | Exposure_Matched_Baseline | 22.99 | 10.97 | 6.17 | 1.717 | 0.0003 | 5.04 | 47.98 | 13.44 |
| Single_RL_Model | PPO_Reward_A | 4.72 | 2.31 | 7.03 | 0.334 | 0.3636 | 8.45 | 32.78 | 1094.15 |
| Reward_Ablation_Model | PPO_Reward_B | 4.01 | 1.99 | 7.34 | 0.3286 | 0.2296 | 8.25 | 33.07 | 1078.25 |
| Reward_Ablation_Model | PPO_Reward_C | 8.27 | 4.03 | 7.32 | 0.5404 | 0.4638 | 8.73 | 33.67 | 1131.39 |
| Single_RL_Model | SAC_Reward_A | 25.63 | 12.16 | 7.26 | 1.6317 | 0.1766 | 5.87 | 55.04 | 189.0 |
| Reward_Ablation_Model | SAC_Reward_B | 24.11 | 11.48 | 6.93 | 1.6077 | 0.1039 | 5.61 | 52.36 | 207.61 |
| Reward_Ablation_Model | SAC_Reward_C | 24.59 | 11.69 | 6.97 | 1.6314 | 0.1522 | 5.79 | 52.75 | 195.49 |
| Dynamic_Ensemble | Dynamic_Ensemble_ADX30_S75_W25 | 19.54 | 9.39 | 6.43 | 1.4269 | 0.0755 | 5.81 | 45.27 | 466.21 |

### RQ 2: 행동 특성 및 동적 앙상블 실효성 (Exposure-Matched Baseline 비교)
- **동결된 최적 앙상블 규칙**: `ADX30_S75_W25` (ADX $\ge 30.0$: PPO 가중치 `0.75`, ADX $< 30.0$: PPO 가중치 `0.25`)
- **Exposure-Matched Baseline 평균 목표 비중**: `47.98%`
- **검증 결론**: 동적 앙상블(`Dynamic_Ensemble_ADX30_S75_W25`)이 Exposure-Matched Baseline 대비 평균 Sharpe `-0.2901`, 평균 CAGR `-1.58%p`를 기록하여, 평균 주식 노출 비중을 통제한 조건에서는 유의미한 초과 성과(시장 타이밍 효과)가 부족함을 투명하게 보고합니다.

### RQ 3: 위험 페널티의 일관성 (Reward Ablation: A vs B vs C)
- **PPO**: Reward A $\to$ B 평균 MDD 변화 `-0.20%p` (일관성: `2/5` Seeds MDD 개선/유지) | Reward B $\to$ C 평균 변동성 변화 `-0.02%p` (일관성: `2/5` Seeds 변동성 개선/유지)
- **SAC**: Reward A $\to$ B 평균 MDD 변화 `-0.26%p` (일관성: `4/5` Seeds MDD 개선/유지) | Reward B $\to$ C 평균 변동성 변화 `+0.04%p` (일관성: `3/5` Seeds 변동성 개선/유지)

### RQ 4: 재현성 및 거래 비용 민감도 분석 (편도 0.1%, 0.3%, 0.5%)
| one_way_cost(%) | Category | Model | mean_cagr(%) | mean_sharpe | mean_mdd(%) | mean_cost($) | excess_cagr_vs_Fixed5050(%) | excess_sharpe_vs_Fixed5050 | excess_cagr_vs_ExpMatched(%) | excess_sharpe_vs_ExpMatched | excess_sharpe_vs_MA20_50 | outperforms_Fixed5050(Sharpe>0) | outperforms_ExpMatched(Sharpe>0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | Traditional_Benchmark | 1_Buy_and_Hold | 23.77 | 1.7212 | 10.29 | 10.0 | 12.32 | 0.0041 | 12.8 | 0.0042 | 0.6993 | True | True |
| 0.1 | Traditional_Benchmark | 2_Cash_100 | 0.0 | 0.0 | 0.0 | 0.0 | -11.45 | -1.7171 | -10.97 | -1.717 | -1.0219 | False | False |
| 0.1 | Traditional_Benchmark | 3_Fixed_50_50 | 11.45 | 1.7171 | 5.25 | 13.73 | 0.0 | 0.0 | 0.48 | 0.0001 | 0.6952 | False | True |
| 0.1 | Traditional_Benchmark | 4_Moving_Average_20_50 | 10.95 | 1.0219 | 9.39 | 95.86 | -0.5 | -0.6952 | -0.02 | -0.6951 | 0.0 | False | False |
| 0.1 | Exposure_Matched_Baseline | Exposure_Matched_Baseline | 10.97 | 1.717 | 5.04 | 13.44 | -0.48 | -0.0001 | 0.0 | 0.0 | 0.6951 | False | False |
| 0.1 | Single_RL_Model | PPO_Reward_A | 2.31 | 0.334 | 8.45 | 1094.15 | -9.14 | -1.3831 | -8.66 | -1.383 | -0.6879 | False | False |
| 0.1 | Reward_Ablation_Model | PPO_Reward_B | 1.99 | 0.3286 | 8.25 | 1078.25 | -9.46 | -1.3885 | -8.98 | -1.3884 | -0.6933 | False | False |
| 0.1 | Reward_Ablation_Model | PPO_Reward_C | 4.03 | 0.5404 | 8.73 | 1131.39 | -7.42 | -1.1767 | -6.94 | -1.1766 | -0.4815 | False | False |
| 0.1 | Single_RL_Model | SAC_Reward_A | 12.16 | 1.6317 | 5.87 | 189.0 | 0.71 | -0.0854 | 1.19 | -0.0853 | 0.6098 | False | False |
| 0.1 | Reward_Ablation_Model | SAC_Reward_B | 11.48 | 1.6077 | 5.61 | 207.61 | 0.03 | -0.1094 | 0.51 | -0.1093 | 0.5858 | False | False |
| 0.1 | Reward_Ablation_Model | SAC_Reward_C | 11.69 | 1.6314 | 5.79 | 195.49 | 0.24 | -0.0857 | 0.72 | -0.0856 | 0.6095 | False | False |
| 0.1 | Dynamic_Ensemble | Dynamic_Ensemble_ADX30_S75_W25 | 9.39 | 1.4269 | 5.81 | 466.21 | -2.06 | -0.2902 | -1.58 | -0.2901 | 0.405 | False | False |
| 0.3 | Traditional_Benchmark | 1_Buy_and_Hold | 23.64 | 1.7132 | 10.29 | 30.0 | 12.34 | 0.0163 | 12.81 | 0.0169 | 0.7763 | True | True |
| 0.3 | Traditional_Benchmark | 2_Cash_100 | 0.0 | 0.0 | 0.0 | 0.0 | -11.3 | -1.6969 | -10.83 | -1.6963 | -0.9369 | False | False |
| 0.3 | Traditional_Benchmark | 3_Fixed_50_50 | 11.3 | 1.6969 | 5.27 | 41.14 | 0.0 | 0.0 | 0.47 | 0.0006 | 0.76 | False | True |
| 0.3 | Traditional_Benchmark | 4_Moving_Average_20_50 | 9.94 | 0.9369 | 9.75 | 285.18 | -1.36 | -0.76 | -0.89 | -0.7594 | 0.0 | False | False |
| 0.3 | Exposure_Matched_Baseline | Exposure_Matched_Baseline | 10.83 | 1.6963 | 5.06 | 40.28 | -0.47 | -0.0006 | 0.0 | 0.0 | 0.7594 | False | False |
| 0.3 | Single_RL_Model | PPO_Reward_A | -8.0 | -1.1851 | 18.09 | 2956.81 | -19.3 | -2.882 | -18.83 | -2.8814 | -2.122 | False | False |
| 0.3 | Reward_Ablation_Model | PPO_Reward_B | -8.14 | -1.0776 | 18.41 | 2915.45 | -19.44 | -2.7745 | -18.97 | -2.7739 | -2.0145 | False | False |
| 0.3 | Reward_Ablation_Model | PPO_Reward_C | -6.57 | -0.9196 | 16.02 | 3047.34 | -17.87 | -2.6165 | -17.4 | -2.6159 | -1.8565 | False | False |
| 0.3 | Single_RL_Model | SAC_Reward_A | 10.29 | 1.4 | 6.19 | 556.58 | -1.01 | -0.2969 | -0.54 | -0.2963 | 0.4631 | False | False |
| 0.3 | Reward_Ablation_Model | SAC_Reward_B | 9.42 | 1.3411 | 6.01 | 610.14 | -1.88 | -0.3558 | -1.41 | -0.3552 | 0.4042 | False | False |
| 0.3 | Reward_Ablation_Model | SAC_Reward_C | 9.77 | 1.3814 | 6.14 | 575.45 | -1.53 | -0.3155 | -1.06 | -0.3149 | 0.4445 | False | False |
| 0.3 | Dynamic_Ensemble | Dynamic_Ensemble_ADX30_S75_W25 | 4.81 | 0.7559 | 6.65 | 1339.03 | -6.49 | -0.941 | -6.02 | -0.9404 | -0.181 | False | False |
| 0.5 | Traditional_Benchmark | 1_Buy_and_Hold | 23.52 | 1.705 | 10.29 | 50.0 | 12.36 | 0.0285 | 12.83 | 0.0296 | 0.8544 | True | True |
| 0.5 | Traditional_Benchmark | 2_Cash_100 | 0.0 | 0.0 | 0.0 | 0.0 | -11.16 | -1.6765 | -10.69 | -1.6754 | -0.8506 | False | False |
| 0.5 | Traditional_Benchmark | 3_Fixed_50_50 | 11.16 | 1.6765 | 5.29 | 68.49 | 0.0 | 0.0 | 0.47 | 0.0011 | 0.8259 | False | True |
| 0.5 | Traditional_Benchmark | 4_Moving_Average_20_50 | 8.95 | 0.8506 | 10.12 | 471.35 | -2.21 | -0.8259 | -1.74 | -0.8248 | 0.0 | False | False |
| 0.5 | Exposure_Matched_Baseline | Exposure_Matched_Baseline | 10.69 | 1.6754 | 5.08 | 67.06 | -0.47 | -0.0011 | 0.0 | 0.0 | 0.8248 | False | False |
| 0.5 | Single_RL_Model | PPO_Reward_A | -17.31 | -2.6041 | 32.64 | 4458.73 | -28.47 | -4.2806 | -28.0 | -4.2795 | -3.4547 | False | False |
| 0.5 | Reward_Ablation_Model | PPO_Reward_B | -17.28 | -2.3949 | 32.88 | 4399.2 | -28.44 | -4.0714 | -27.97 | -4.0703 | -3.2455 | False | False |
| 0.5 | Reward_Ablation_Model | PPO_Reward_C | -16.1 | -2.2791 | 31.14 | 4578.77 | -27.26 | -3.9556 | -26.79 | -3.9545 | -3.1297 | False | False |
| 0.5 | Single_RL_Model | SAC_Reward_A | 8.45 | 1.1681 | 6.54 | 910.71 | -2.71 | -0.5084 | -2.24 | -0.5073 | 0.3175 | False | False |
| 0.5 | Reward_Ablation_Model | SAC_Reward_B | 7.42 | 1.0741 | 6.42 | 996.37 | -3.74 | -0.6024 | -3.27 | -0.6013 | 0.2235 | False | False |
| 0.5 | Reward_Ablation_Model | SAC_Reward_C | 7.87 | 1.1315 | 6.52 | 941.16 | -3.29 | -0.545 | -2.82 | -0.5439 | 0.2809 | False | False |
| 0.5 | Dynamic_Ensemble | Dynamic_Ensemble_ADX30_S75_W25 | 0.43 | 0.0852 | 8.38 | 2138.26 | -10.73 | -1.5913 | -10.26 | -1.5902 | -0.7654 | False | False |
