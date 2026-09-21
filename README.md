# 🌳 시계열/공간 데이터 기반 도시 수목 생존 예측 및 고위험군 분류 알고리즘 구축

시계열/공간 데이터 기반으로 도시 수목(묘목)의 **생존을 예측**하고 고위험군을 분류하는 프로젝트입니다.
묘목의 환경·생물·화학 데이터를 활용해 나무의 생존을 예측하고, 지속 가능한 도시 수목원 조성을 위한 전략을 도출합니다.

> **프로젝트 성격**
> SKN 부트캠프 팀 프로젝트(4인) → 이후 **개인적으로 코드를 전면 재작성하며 복기**했습니다.
> 초기 모델이 "결과가 정해진 뒤에 기록되는 값(Time 등)"을 변수로 써서 성능이 비정상적으로 높게 나온다는 걸 발견했고,
> 이 문제를 바로잡기 위해 코드를 처음부터 다시 짰습니다.

---

## 📌 한눈에 보기

| 항목 | 내용 |
|------|------|
| 문제 정의 | 묘목의 **생존 여부(분류)** 와 **시간에 따른 생존(생존 분석)** 예측 |
| 데이터 | [Kaggle - Tree Survival Prediction](https://www.kaggle.com/datasets/yekenot/tree-survival-prediction/data) (2,783행 × 24열) |
| 관측 단위 | 심어진 묘목 한 그루 |
| 접근 | ① 이진 분류(Logistic / RF / XGBoost) → ② 생존 분석(Cox PH / Random Survival Forest) |
| 핵심 주제 | **데이터 누수 탐지**와 "정직한 성능" 확보 |
| 사용 기술 | Python, Scikit-learn, XGBoost, lifelines(Kaplan-Meier, Cox PH), scikit-survival(RSF), pandas, numpy |

---

## 📂 디렉토리 구조

```
Urban-Tree-Survival-Risk-Analysis
│
├─ data
│   ├─ raw                       # 원본 데이터 (Kaggle Tree_Data.csv)
│   └─ processed                 # 노트북 실행으로 생성 (clf/surv 데이터셋, 인코더)
│
├─ notebooks                     # 분석·모델링 파이프라인 (복기 버전)
│   ├─ 01_EDA.ipynb              # 탐색적 분석 + 데이터 누수 발견
│   ├─ 02_preprocessing.ipynb    # 누수 방지 전처리 + 데이터셋 구성
│   ├─ 03_classification.ipynb   # 이진 분류 3종 + 누수 대조 실험
│   └─ 04_survival_analysis.ipynb # 생존 분석 (Kaplan-Meier / Cox / RSF)
│
├─ requirements.txt
└─ README.md
```

---

## 🗂️ 데이터

미국 삼림 묘목 이식 실험 데이터로, 각 행은 심어진 묘목 한 그루의 관측 기록입니다.

| 구분 | 컬럼 | 내용 |
|------|------|------|
| 환경 | `Light_ISF`, `Light_Cat`, `Soil`, `Sterile` | 조도(연속·등급), 토양 출처, 살균 여부 |
| 생물 | `Species`, `Myco`, `SoilMyco`, `AMF`, `EMF`, `Conspecific` | 수종, 균근 타입(묘목/토양), 균근 감염 비율 |
| 화학 | `Phenolics`, `Lignin`, `NSC` | 식물 내 방어·구조·저장 물질 |
| 결과 | `Time`, `Event`, `Harvest`, `Alive` | 사건 발생 시각 / 사망·수확·생존 플래그 |

**라벨 통합**: `Event`/`Harvest`/`Alive` 세 플래그를 하나의 상태로 병합
→ 사망(0) 1,587 · 수확(1) 704 · 생존(2) 491

출처: [Kaggle - Tree Survival Prediction](https://www.kaggle.com/datasets/yekenot/tree-survival-prediction/data)

---

## 🛠️ 기술 스택

- **언어/환경**: Python, venv
- **데이터**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **분류**: scikit-learn, XGBoost
- **생존 분석**: lifelines (Kaplan-Meier, Cox PH), scikit-survival (Random Survival Forest)

---

## 1. 개요

묘목의 환경·생물·화학 데이터를 활용해 나무의 생존을 예측하는 프로젝트입니다.
부트캠프 팀 프로젝트를 마친 뒤, 초기 모델이 **결과가 정해진 뒤에 기록되는 값(Time 등)을 변수로 써서
성능이 비정상적으로 높게 나온다**는 걸 발견했고, 이 문제를 바로잡기 위해 코드를 처음부터 다시 짰습니다.

---

## 2. 담당 역할과 문제 해결 과정

- **데이터 누수 발견 및 제거**
  EDA 과정에서 `Time`(사건 발생 시각) 값 하나만으로도 생존·사망을 **99.9% 정확도**로 분류할 수 있다는 걸 확인했습니다.
  살펴보니 생존 개체는 전부 `Time=115.5`, 수확 개체는 대부분 `Time≈24.5`로 찍혀 있었는데, 이는 결과가 이미 정해진 뒤에 기록되는 값이었습니다.
  `Time`과 그로부터 파생된 컬럼(`Event`, `Harvest`, `Alive`)을 입력 변수에서 모두 제외했습니다.

- **재현성 확보**
  범주형 변수는 Label Encoding으로 통일하고, 데이터 분할 시 `random_state`를 고정해 실험을 다시 돌려도 같은 결과가 나오도록 만들었습니다.

- **누수 전후 비교 실험**
  일부러 `Time`을 다시 피처에 넣고 RandomForest로 돌려보니 **Accuracy 0.998, ROC-AUC 1.00**이 나왔습니다.
  반대로 `Time`을 뺀 상태에서는 **Accuracy 0.83, ROC-AUC 0.90** 수준이었습니다.
  이 비교를 통해 "누수가 있으면 성능이 비현실적으로 높아진다"는 걸 직접 증명했습니다.
  이후 Logistic Regression, RandomForest, XGBoost 세 모델로 공정 비교했고, ROC-AUC는 모두 **0.89~0.90대**로 나왔습니다.

- **생존 분석으로 확장**
  분류 모델은 "지금 이 순간 살았는지 죽었는지"만 답할 수 있다는 한계가 있었습니다.
  "언제 위험해지는가"까지 보기 위해 Kaplan-Meier로 생존 곡선을, Cox 비례위험 모델과 Random Survival Forest로 시간에 따른 위험도를 추정했습니다.
  RSF 기준 **C-index 0.762**를 확보했습니다.

---

## 3. 성과

- **누수 제거 전후 비교**: Accuracy 0.998 → 0.83, ROC-AUC 1.00 → 0.90
  (수치는 낮아졌지만 실제 예측 상황에 맞는 신뢰할 수 있는 성능 확보)
- **최종 분류 모델 (RandomForest 기준)**: Accuracy 0.8269, ROC-AUC 0.9002
- **생존 분석 모델 (RSF)**: C-index 0.762

### 분류 모델 비교 (누수 제거)

| 모델 | Accuracy | F1 | ROC-AUC |
|------|----------|-----|---------|
| **Logistic** | **0.8486** | **0.7296** | 0.8938 |
| RandomForest | 0.8269 | 0.6250 | **0.9002** |
| XGBoost | 0.8149 | 0.5926 | 0.8954 |

### 누수 대조 실험 — 일부러 `Time`을 피처로 추가하면

| | ROC-AUC | Accuracy |
|--|:--:|:--:|
| 누수 제거 (RandomForest) | 0.90 | 0.83 |
| 누수 포함 (RandomForest + `Time`) | **1.00** | **0.998** |

→ `Time` 하나로 성능이 거의 완벽해지지만, 예측 시점에 알 수 없는 값이므로 **실제로는 쓸모없는 모델**입니다.

### 생존 분석 모델 비교

| 모델 | C-index |
|------|:-------:|
| Cox 비례위험 (lifelines) | 0.746 |
| **Random Survival Forest** (scikit-survival) | **0.762** |

- Kaplan-Meier로 전체·집단별 생존 곡선 추정 (중앙 생존 시간 52.5일)
- Cox 위험비로 각 요인의 방향 해석 (`Myco`, `Sterile` 등이 생존에 유리)
- RSF로 개체별 생존 곡선 예측 → 어떤 묘목을 언제 집중 관리할지 판단 근거

---

## 4. 분석 파이프라인 (notebooks)

### 01. EDA — 데이터 누수 발견
- 변수 의미·결측치·라벨 정의 확인
- **핵심 발견**: `Time`(사건 발생 시각)만으로 생존/사망을 **99.9% 정확도**로 분류
  - 생존 개체는 전부 `Time=115.5`, 수확 개체는 거의 `Time≈24.5` → 결과가 정해진 뒤 기록되는 값
- 생존 요인 탐색: 균근 타입(EMF 59.6% vs AMF 1.6%), 수종(참나무류 ~60%), 화학 성분

### 02. 전처리 & 데이터셋 구성
- 라벨 통합, 결측치 처리(EMF→0), Phenolics 음수 보정
- **결과 파생 컬럼(`Time`/`Event`/`Harvest`/`Alive`)과 식별자 컬럼 제거**
- 범주형 Label Encoding, `random_state` 고정 분할
- 두 데이터셋 분리 저장: `clf_dataset`(분류용), `surv_dataset`(생존분석용)

### 03. 분류 — "높은 성능 ≠ 좋은 모델"
- Logistic / RandomForest / XGBoost 3종 공정 비교 (누수 제거)
- 누수 대조 실험으로 "누수가 성능을 부풀린다"는 것을 정량 증명

### 04. 생존 분석 — 시간에 따른 위험 예측
- 분류의 한계("관측 종료 시점 한 순간"만 봄)를 넘어 생존 분석으로 확장
- 여기서는 `Time`을 **예측 대상(생존 시간)** 으로 쓰므로 정당
- Kaplan-Meier / Cox PH / Random Survival Forest

---

## 🚀 실행 방법

```bash
# 1. 가상환경 생성 및 활성화 (Windows / Python 3.13)
python -m venv project_env
project_env\Scripts\activate

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 분석 노트북 실행 (Jupyter 커널: Python (project_env))
#    notebooks/01 → 04 순서로 실행
#    02를 실행하면 data/processed/의 전처리 결과가 생성됩니다.
```

> `data/processed/`는 `02_preprocessing.ipynb` 실행으로 재생성되므로 저장소에서 제외했습니다.
> `data/raw/Tree_Data.csv`(재현 시작점)는 저장소에 포함되어 있습니다.

---

## 💭 회고

숫자를 높이는 것보다 **"믿을 수 있는 숫자를 만드는 것"** 에 집중한 프로젝트였습니다.

`Time` 하나만 넣어도 ROC-AUC가 거의 1.0까지 올라갔지만, 그건 예측 시점엔 알 수 없는 정보였기 때문에
실제로는 아무 쓸모가 없었습니다. 누수를 걷어내자 성능 지표는 낮아졌지만, 대신 **실제로 써먹을 수 있는 정직한 모델**을 얻었습니다.

또한 "한 시점의 생존 여부"와 "시간에 따른 위험"은 서로 다른 질문이라서,
**분류와 생존 분석이라는 다른 도구가 각각 필요하다**는 것도 배웠습니다.
