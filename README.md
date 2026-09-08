# 🌳 도시 수목 생존 예측 (Urban Tree Survival Risk Analysis)

묘목의 환경·생물·화학 조건 데이터를 기반으로 **나무의 생존을 예측**하고,
지속 가능한 도시 수목원 조성을 위한 데이터 기반 전략을 도출하는 프로젝트입니다.

> **이 저장소에 대하여**
> SKN 부트캠프 미니 팀 프로젝트(가지가지팀, 4인)로 시작했습니다.
> 이후 **개인적으로 코드를 처음부터 다시 작성하며, 팀 버전에서 아쉬웠던 부분(데이터 누수, 재현성, 구조)을 개선**했습니다.
> 아래 문서는 개선된 개인 복기 버전을 기준으로 작성되었으며, 팀 원본과의 차이를 명시했습니다.

---

## 📌 한눈에 보기

| 항목 | 내용 |
|------|------|
| 문제 정의 | 묘목의 **생존 여부(분류)** 와 **시간에 따른 생존(생존 분석)** 예측 |
| 데이터 | [Kaggle - Tree Survival Prediction](https://www.kaggle.com/datasets/yekenot/tree-survival-prediction/data) (2,783행 × 24열) |
| 관측 단위 | 심어진 묘목 한 그루 |
| 접근 | ① 이진 분류(Logistic / RF / XGBoost) → ② 생존 분석(Cox PH / Random Survival Forest) |
| 핵심 주제 | **데이터 누수 탐지**와 "정직한 성능" 확보 |
| 실행 환경 | Python 3.13, `project_env` (venv) |

---

## 🔍 팀 원본 → 개인 복기 개선점

이번 복기의 핵심은 **"성능 수치를 높이는 것"이 아니라 "믿을 수 있는 수치를 만드는 것"** 이었습니다.

| 구분 | 팀 원본 버전 | 개인 복기 버전 (개선) |
|------|--------------|------------------------|
| 코드 구조 | 노트북 6~7개 파편화 + 모듈 클래스 | **번호 붙은 파이프라인 노트북 4개** (01→04) |
| 데이터 누수 | 분류에 `Time`(결과 파생값) 포함 가능 | **`Time`·결과 파생 컬럼 전면 제외** |
| 성능의 정직성 | 높은 정확도(누수 영향) | **누수 제거 후 현실적 성능 + 누수 대조 실험으로 증명** |
| 재현성 | `train_test_split`에 seed 없음 | **모든 분할 `random_state=42` 고정** |
| Phenolics 음수 | 노트북마다 처리 방식 상이(`abs` / `x-min`) | **전처리에서 일관된 보정으로 통일** |
| 결과물 | 산발적 산출물 | **분류용·생존분석용 데이터셋 분리 저장** |

> 팀 버전에서 분류 모델이 `Time`을 피처로 사용하면 성능이 크게 부풀려집니다.
> 이 복기에서는 누수를 제거해 성능 지표는 낮아졌지만, **실제 예측 상황에 훨씬 가까운 정직한 성능**을 얻었습니다.

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

## ⚙️ 분석 파이프라인 (notebooks)

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
동일한 train/test로 3개 모델을 공정 비교했습니다 (누수 제거).

| 모델 | Accuracy | F1 | ROC-AUC |
|------|----------|-----|---------|
| **Logistic** | **0.8486** | **0.7296** | 0.8938 |
| RandomForest | 0.8269 | 0.6250 | **0.9002** |
| XGBoost | 0.8149 | 0.5926 | 0.8954 |

**누수 대조 실험** — 일부러 `Time`을 피처로 추가하면:

| | ROC-AUC | Accuracy |
|--|:--:|:--:|
| 누수 제거 (RandomForest) | 0.90 | 0.83 |
| 누수 포함 (RandomForest + `Time`) | **1.00** | **0.998** |

→ `Time` 하나로 성능이 거의 완벽해지지만, 예측 시점에 알 수 없는 값이므로 **실제로는 쓸모없는 모델**입니다.

### 04. 생존 분석 — 시간에 따른 위험 예측
분류는 "관측 종료 시점 한 순간"만 봅니다. 실무의 "언제 위험한가"에 답하기 위해 생존 분석으로 확장했습니다.
여기서는 `Time`을 **예측 대상(생존 시간)** 으로 쓰므로 정당합니다.

| 모델 | C-index |
|------|:-------:|
| Cox 비례위험 (lifelines) | 0.746 |
| **Random Survival Forest** (scikit-survival) | **0.762** |

- Kaplan-Meier로 전체·집단별 생존 곡선 추정 (중앙 생존 시간 52.5일)
- Cox 위험비로 각 요인의 방향 해석 (`Myco`, `Sterile` 등이 생존에 유리)
- RSF로 개체별 생존 곡선 예측 → 어떤 묘목을 언제 집중 관리할지 판단 근거

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

## 🛠️ 기술 스택

- **언어/환경**: Python 3.13, venv
- **데이터**: pandas, numpy
- **시각화**: matplotlib, seaborn
- **분류**: scikit-learn, XGBoost
- **생존 분석**: lifelines (Kaplan-Meier, Cox PH), scikit-survival (Random Survival Forest)

---

## 💭 회고

### 팀 프로젝트에서
Kaggle 실데이터로 EDA부터 전처리, 분류, 생존 분석까지 데이터 분석 전 과정을 경험했습니다.
특히 "분류만으로는 시간에 따른 위험을 못 본다"는 한계를 인식하고 생존 분석으로 확장한 설계가 좋았습니다.

### 복기하면서 배운 것
코드를 처음부터 다시 짜며 가장 크게 느낀 점은 **"높은 성능 지표가 항상 좋은 모델을 의미하지 않는다"** 였습니다.

- `Time` 하나를 넣으면 ROC-AUC가 0.90에서 1.00으로 뛰지만, 그 값은 예측 시점에 알 수 없는 정보였습니다 (데이터 누수).
- 누수를 제거하자 성능 지표는 낮아졌지만, **실제 예측 상황에 가깝고 재현 가능한 정직한 파이프라인**을 얻었습니다.
- 같은 데이터라도 질문(한 시점의 생존 vs 시간에 따른 위험)에 따라 **분류와 생존 분석**이라는 다른 도구가 필요하다는 것을 배웠습니다.

낮아진 숫자를 그대로 두고 "왜 낮아졌는지"를 설명할 수 있는 것이,
숫자만 높은 모델보다 훨씬 가치 있다고 생각합니다.
