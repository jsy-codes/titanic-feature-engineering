# Titanic 특성 공학(Feature Engineering) 파이프라인 구현 및 성능 비교 실험

빅데이터 분석 과제3 — Titanic 생존 예측(이진 분류)에서 결측치 처리·인코딩·스케일링·파생 변수·변수 선택이 성능에 미치는 영향을 비교한다.

## 파일 구성
| 파일 | 설명 |
|---|---|
| `titanic_feature_engineering.ipynb` | 전체 실험 코드 (실행 결과 포함) |
| `requirements.txt` | 사용한 라이브러리 버전 |
| `train.csv` | Kaggle Titanic 학습 데이터 (직접 내려받아 같은 폴더에 두기) |

## 실행 방법
```bash
pip install -r requirements.txt
jupyter notebook titanic_feature_engineering.ipynb   # 위에서부터 순서대로 실행 (약 15분, CPU 1코어 기준)
```
실행하면 `figures/`(그래프)와 `results/`(결과표 CSV)가 생성된다. 난수 시드는 모두 42로 고정했다.

## 실험 설계 요약
- 학습 80% / 홀드아웃 20% 층화 분할, 학습 80%에서 5-fold × 2회 반복 Stratified CV로 모든 비교 수행
- 전처리는 `Pipeline` + `ColumnTransformer`로 구성해 학습 fold로만 fit (데이터 누수 방지)
- 비교 항목: 결측치(Mean/Median/Most Frequent/Drop NA), 인코딩(One-Hot/Label), 스케일링(None/Standard/MinMax/Robust), 파생 변수 6종 Ablation, 변수 선택(RF 중요도 top-N/SelectKBest/RFE)
- 모델: Logistic Regression, Random Forest, XGBoost, LightGBM / 가산점: GridSearchCV, SHAP, Permutation Importance
