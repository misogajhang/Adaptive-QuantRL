# Adaptive-QuantRL: Regime-Adaptive PPO/SAC Ensemble & Risk-Penalized Reinforcement Learning for Dynamic Portfolio Allocation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-brightgreen.svg)](https://stable-baselines3.readthedocs.io/)
[![Tests Passed](https://img.shields.io/badge/Pytest-All%20Passed-success.svg)](#4-원클릭-재현-가이드-how-to-reproduce)

[📄 Technical Report (한글 기술 보고서 PDF)](./paper/Adaptive_QuantRL_Report_KR.pdf) | [📄 Academic Paper (영문 IEEE 논문 PDF)](./paper/Adaptive_QuantRL_Paper_EN.pdf)

**Adaptive-QuantRL**은 단일 자산(SPY) 동적 자산배분 환경에서 **On-Policy(PPO)**와 **Off-Policy(SAC)** 강화학습 알고리즘의 행동적 차이를 규명하고, **보상 함수 위험 페널티(Drawdown & Variance Ablation)** 및 **ADX(14) 시장 국면 기반 Action-Level 동적 앙상블**의 실효성을 **엄격하게 통제된 실험(8-Phase Controlled Protocol)**으로 검증한 퀀트/데이터 사이언스 연구 프로젝트입니다.

---

## 1. 프로젝트 개요 및 퀀트/데이터 사이언스 문제 정의

금융 시계열에 강화학습(RL)을 적용한 수많은 기존 연구들은 비현실적인 체결 가정과 데이터 누수(Data Leakage)로 인해 백테스트 성과를 과대평가하는 문제를 안고 있습니다. 본 프로젝트는 실전 퀀트 리서치 기준에 부합하도록 다음 **4대 실험 통제 및 회계 무결성 원칙**을 엔지니어링 단계에서 강제했습니다.

1. **선행 참조 편향(Look-Ahead Bias) 원천 차단**:
   - **시계열 홀드아웃 분할(Chronological Holdout Split)**: Train(`2010-01-01 ~ 2019-12-31`, 2,515일), Validation(`2020-01-01 ~ 2022-12-31`, 755일), Test(`2023-01-01 ~ 2024-12-31`, 501일)를 엄격히 분리하고, 각 구간의 기술적 지표 계산을 위한 사전 Warm-up 버퍼는 평가 수익률에 일절 산입하지 않습니다.
   - **Train 전용 정규화(Scaler Isolation)**: 16개 시장 특징량(Market Features)의 `StandardScaler` 및 관측용 로그 분산(`obs_log_variance`) 정규화 통계량은 오직 **Train(2010~2019)** 구간에서만 적합(fit)하여 체크포인트 메타데이터에 동결합니다.
2. **현실적 2단계 주문 체결 및 거래비용 회계 ($t$일 종가 관측 $\to$ $t+1$일 시가 체결 $\to$ $t+1$일 종가 평가)**:
   - 에이전트가 $t$일 장 마감(`Close_t`) 상태를 관측해 목표 비중 $w_t \in [0, 1]$을 결정하면, $t+1$일 시가(`Open_{t+1}`)에 **야간 갭(Overnight Gap) 손익을 선반영**한 뒤 리밸런싱을 수행합니다.
   - 매 거래마다 **편도 0.1% (수수료 0.05% + 슬리피지 0.05%)** 비용을 정확히 1회 차감하며, 총자산 회계 항등식($\text{Equity}_t = \text{Cash}_t + \text{Shares}_t \times \text{Price}_t$)을 단위 테스트(Assertion)로 검증합니다.
3. **관측 공간(Observation)과 보상 함수(Reward)의 스케일 분리**:
   - 신경망 입력 안정성을 위해 관측 벡터에는 로그 정규화 분산 $\text{Normalize}(\ln(\text{Var}^{raw}_{30,t} + 10^{-8}))$을 주입하되, 보상 함수 페널티 계산에는 반드시 **음수가 될 수 없는 원본 분산($\text{Var}^{raw}_{30,t} \ge 0$)**만 사용하여 분산 페널티가 부호 반전으로 인해 오히려 변동성을 장려하는 수학적 오류를 방지했습니다.
4. **노출 통제 기준선(Exposure-Matched Baseline) 및 다중 시드(5 Seeds) 검증**:
   - 동적 전략의 성과가 단순한 평균 베타(주식 노출 비중) 차이에서 기인한 것인지, 실제 시장 타이밍(Market Timing) 알파에서 기인한 것인지 분리하기 위해 Validation에서 동결된 시드별 평균 주식 비중(`47.98%`)을 고정 보유하는 **Exposure-Matched Baseline**을 도입했습니다.
   - 모든 실험은 5개 독립 난수 시드(`11, 22, 33, 44, 55`) × 2개 알고리즘(`PPO, SAC`) × 3개 보상 체계(`Reward A, B, C`) = **총 30개 독립 모델**의 평균 및 표준편차로 평가합니다.

> **연구 범위 및 한계 명시**: 본 실험은 투자 행동 및 리밸런싱 메커니즘 분석에 집중하기 위해 배당 현금흐름을 제외한 비수정(Unadjusted) SPY 가격 수익률을 기준으로 하며(모든 벤치마크 동일 적용), 부분 관측 환경(POMDP) 및 단일 자산의 특정 상승장 구간(Test: 2023~2024) 결과를 타 자산이나 모든 거시 국면에 무분별하게 일반화하지 않습니다.

---

## 2. 핵심 결과 하이라이트 요약표 (Test: 2023~2024, 편도 비용 0.1%)

Validation(2020~2022)에서 하이퍼파라미터와 앙상블 규칙을 완전히 동결(Freeze)한 뒤, 미관측 Test 구간(`2023-01-01 ~ 2024-12-31`, 501 거래일)에서 결정론적(Deterministic) 정책으로 평가한 최종 성과표입니다.

| 구분 (Category) | 전략 / 모델 (Model) | 누적 수익률 (%) | 연평균 수익률 CAGR (%) | 연 변동성 (%) | Sharpe Ratio (Mean ± Std) | 최대 낙폭 MDD (%) | 평균 주식 비중 (%) | 누적 거래비용 ($) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Benchmark** | `1_Buy_and_Hold` (SPY 100%) | 52.80 | **23.77** | 12.88 | **1.7212** ± 0.0000 | 10.29 | 100.00 | 10.00 |
| **Benchmark** | `2_Cash_100` (현금 100%) | 0.00 | 0.00 | 0.00 | 0.0000 ± 0.0000 | 0.00 | 0.00 | 0.00 |
| **Benchmark** | `3_Fixed_50_50` (고정 50:50) | 24.05 | 11.45 | 6.43 | 1.7171 ± 0.0000 | 5.25 | 50.00 | 13.73 |
| **Benchmark** | `4_Moving_Average_20_50` (MA 교차) | 22.94 | 10.95 | 10.73 | 1.0219 ± 0.0000 | 9.39 | 73.65 | 95.86 |
| **Exposure Control** | `Exposure_Matched_Baseline` (고정 47.98%) | 22.99 | 10.97 | 6.17 | 1.7170 ± 0.0003 | **5.04** | 47.98 | 13.44 |
| **Single RL (PPO)** | `PPO_Reward_A` (로그수익률 단독) | 4.72 | 2.31 | 7.03 | 0.3340 ± 0.3636 | 8.45 | 32.78 | 1,094.15 |
| **Ablation (PPO)** | `PPO_Reward_B` (+낙폭 페널티) | 4.01 | 1.99 | 7.34 | 0.3286 ± 0.2296 | 8.25 | 33.07 | 1,078.25 |
| **Ablation (PPO)** | `PPO_Reward_C` (+낙폭·분산 페널티) | 8.27 | 4.03 | 7.32 | 0.5404 ± 0.4638 | 8.73 | 33.67 | 1,131.39 |
| **Single RL (SAC)** | `SAC_Reward_A` (로그수익률 단독) | **25.63** | **12.16** | 7.26 | **1.6317** ± 0.1766 | 5.87 | 55.04 | 189.00 |
| **Ablation (SAC)** | `SAC_Reward_B` (+낙폭 페널티) | 24.11 | 11.48 | 6.93 | 1.6077 ± 0.1039 | **5.61** | 52.36 | 207.61 |
| **Ablation (SAC)** | `SAC_Reward_C` (+낙폭·분산 페널티) | 24.59 | 11.69 | 6.97 | 1.6314 ± 0.1522 | 5.79 | 52.75 | 195.49 |
| **Dynamic Ensemble** | `Dynamic_Ensemble_ADX30_S75_W25` | 19.54 | 9.39 | 6.43 | 1.4269 ± 0.0755 | 5.81 | 45.27 | 466.21 |

---

## 3. 핵심 발견 3가지 (Key Empirical Findings)

### ① SAC의 거래비용 억제 및 안정적 위험조정 성과 (Sharpe 1.63, MDD 5.61%)
* **Off-Policy 엔트로피 정규화의 회전율 억제 효과**: 리플레이 버퍼(Replay Buffer)와 최대 엔트로피 목적함수를 사용하는 **SAC**는 연속적인 행동 공간($w_t \in [0, 1]$)에서 급격한 포지션 스위칭을 자제하고 평균 `52~55%`의 안정적인 주식 노출을 유지했습니다.
* 그 결과, 2년(501 거래일) 동안 누적 거래비용을 **`$189 ~ $208`** 수준으로 억제하며 **`SAC_Reward_A` 기준 CAGR `12.16%`, Sharpe `1.6317`, MDD `5.87%`**(`SAC_Reward_B` 기준 **MDD `5.61%`**)를 달성했습니다.
* 이는 전통적 추세추종 전략인 **`Moving_Average_20_50` (CAGR `10.95%`, Sharpe `1.0219`, MDD `9.39%`)을 수익률, 위험조정수익률, 최대낙폭 모든 지표에서 압도**하며, `Buy_and_Hold`(MDD `10.29%`) 대비 최대 낙폭을 절반 가까이 줄인 우수한 방어력입니다. 또한 편도 거래비용을 `0.5%`(5배)로 높인 극한 스트레스 테스트에서도 SAC는 **Sharpe `1.17` (CAGR `8.45%`)**로 양(+)의 알파와 강건성을 유지했습니다.

### ② PPO 결합 동적 앙상블의 회전율 비용 잠식 (Exposure-Matched Baseline 대비 열세 원인 규명)
* **고회전 On-Policy(PPO)의 거래비용 누수**: 반면 `PPO` 모델들은 국소적인 수익률 변화에 과민 반응하여 Test 구간 동안 초기 자본(`$10,000`)의 약 11%에 달하는 **`$1,078 ~ $1,131`의 누적 거래비용**을 지불했고, 이로 인해 순수익률이 CAGR `1.99% ~ 4.03%` (Sharpe `0.33 ~ 0.54`)로 크게 훼손되었습니다.
* **Exposure-Matched Baseline을 통한 엄밀한 귀인 분석(Attribution)**: Validation 27개 규칙 중 1위로 선정된 동적 앙상블(`ADX30_S75_W25`: $\text{ADX} \ge 30$에서 PPO `0.75` + SAC `0.25`, $\text{ADX} < 30$에서 PPO `0.25` + SAC `0.75`)은 Test 구간에서 **Sharpe `1.4269`, MDD `5.81%`**를 기록해 단일 PPO 및 이동평균 전략은 크게 앞섰습니다.
* 그러나 동일한 평균 주식 노출 비중(`47.98%`)을 고정 유지하는 **`Exposure_Matched_Baseline` (CAGR `10.97%`, Sharpe `1.7170`, 거래비용 `$13.44`)**과 비교하면 **Sharpe `-0.2901`, CAGR `-1.58%p` 열세**를 보였습니다. 이는 추세 국면 전환 시 PPO 가중치가 섞이면서 발생한 추가 리밸런싱 거래비용(`$466.21`)이 국면 전환 타이밍 이득을 상쇄했음을 보여주며, **노출 통제 기준선 없이는 앙상블의 성과를 과대해석할 수 있음**을 실증적으로 입증했습니다.

### ③ Reward B (Drawdown 페널티)의 Seed별 MDD 방어 일관성 (4/5 Seeds 개선)
* **사전 스케일 보정(Pre-Training Calibration)**: 일간 로그수익률 절대값 평균($|\ln(E_{t+1}/E_t)| \approx 0.0043$) 대비 위험 페널티가 $10\% \sim 12\%$ 비중을 차지하도록 보정($\lambda_d = 0.02, \lambda_v = 10.0$)하여 정책이 전량 현금(`0%`)으로 붕괴(Collapse)하는 현상을 방지했습니다.
* **일관된 하방 위험 제어**: SAC에 낙폭 페널티를 부여한 `SAC_Reward_B`는 기본 `SAC_Reward_A` 대비 **Validation 구간 전체 1위(Sharpe `0.2753`, MDD `20.69%`)**를 기록했을 뿐 아니라, Test 구간에서도 평균 MDD를 `5.87%` $\to$ **`5.61%` (`-0.26%p`)**로 낮추고 시드 간 Sharpe 표준편차를 `0.1766` $\to$ **`0.1039`**로 41% 축소했습니다. 특히 **5개 독립 Random Seed 중 4개 시드(`4/5` Seeds: Seed 22, 33, 44, 55)**에서 일관되게 MDD가 개선되었으며, Seed 33에서는 **MDD `4.53%`**를 달성했습니다.

---

## 4. 원클릭 재현 가이드 (How to Reproduce)

본 저장소는 데이터 수집부터 전처리, 30개 강화학습 모델 독립 학습, 27개 동적 앙상블 검증, 비용 민감도 분석 및 최종 마크다운 리포트 생성까지 **단 3개의 명령어로 100% 동일하게 재현**됩니다.

### Step 1: 가상환경 생성 및 패키지 설치
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: 단위 테스트 검증 (회계 항등식, 누수 차단, 앙상블 로직)
```bash
pytest
```

### Step 3: 전체 파이프라인 순차 실행 (Phase 1 ~ Phase 8)
```bash
# [Phase 1] SPY 15년 비수정 OHLC 데이터 검증, Holdout 분할, Train 전용 Scaler 적합 및 4대 벤치마크 평가
python run_phase1.py

# [Phase 3] 사전 보상 스케일 측정 및 30개 독립 RL 모델 본 학습 (PPO/SAC x Reward A/B/C x 5 Seeds, 10,080 steps)
python run_phase3.py --overwrite

# [Phase 4~8] Validation 모델/앙상블 동결 -> Test(2023~2024) 최종 평가 -> 비용 민감도(0.1%, 0.3%, 0.5%) 및 RQ 리포트 생성
python run_phase4_to_8.py
```

---

## 5. 디렉토리 구조 (Repository Structure)

```text
├── config.py                         # 전 구간 실험 통제 파라미터, 하이퍼파라미터, 시드(11~55) 중앙 설정
├── data_pipeline.py                  # Phase 1: 비수정 SPY 데이터 분할, Warm-up 처리, Train-only Scaler
├── baseline_models.py                # Phase 1: 4대 전통적 벤치마크 (B&H, Cash, Fixed 50:50, MA 20/50)
├── envs/
│   └── trading_env.py                # Phase 2: 야간 갭 선반영 -> 시가 체결 -> 종가 평가 TradingEnv & Reward A/B/C
├── training/
│   ├── measure_reward_scale.py       # Phase 3: 사전 보상 구성요소 스케일/분위수 측정 및 페널티 계수 보정
│   └── train_models.py               # Phase 3: Track 1(앙상블) & Track 2(Ablation) 30개 모델 학습/검증
├── src/
│   ├── ensemble/
│   │   └── dynamic_ensemble.py       # Phase 5 & 6: ADX(14) 국면 분류 및 27개 공유 환경 동적 앙상블 최적화
│   └── evaluation/
│       ├── validate_rewards.py       # Phase 4: Validation 보상 Ablation 평가 및 체크포인트 동결
│       └── run_test_evaluation.py    # Phase 7 & 8: Test 결정론적 평가, 비용 민감도(0.1~0.5%), RQ 보고서 생성
├── tests/                            # Phase 1~8 전 단계 무결성 검증 Pytest 스위트
├── results/                          # 실험 결과 CSV, 동결 메타데이터 JSON 및 최종 마크다운 보고서
│   └── phase7_8/
│       └── phase8_rq_verification_report.md
├── paper/
│   ├── Adaptive_QuantRL_Report_KR.pdf # 한글 기술 보고서 PDF
│   ├── Adaptive_QuantRL_Paper_EN.pdf  # 영문 IEEE 2단 학술 논문 PDF
│   ├── report_kr.md                  # 한글 기술 보고서 Markdown 원본
│   └── paper.tex                     # IEEE/ACM 2단 스타일 학술 논문 LaTeX 소스
├── run_phase1.py                     # Phase 1 실행 스크립트
├── run_phase3.py                     # Phase 3 실행 스크립트
└── run_phase4_to_8.py                # Phase 4 ~ Phase 8 통합 실행 스크립트
```
