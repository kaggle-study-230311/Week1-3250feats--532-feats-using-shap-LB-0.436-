# ****Costa Rican Household Poverty Level Prediction****

1. 목표 : 코스타리카 가구 특성에 대한 데이터셋을 기반으로 PMT(대리 수단 테스트) 알고리즘의 개선
2. 배경
    - 사회적 지원 프로그램 운영을 위한 최빈곤층을 선정하는 알고리즘 개발의 필요성
    - 최빈층은 일반적으로 자격을 증명하는 수입 및 지출 기록을 제출하기 어려운 환경임
    - 그래서 PMT(대리 수단 테스트)를 통해 다양한 가구의 특성을 고려하는 모델로 접근
3. 평가 - `macro F1 score`
    1. macro F1 score?
        
        ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b847aabd-b4ca-4585-a95a-ffd8f458c1c6/Untitled.png)
        

1. 데이터셋 특징
    1. 142개의 컬럼
    - **Id** - 각 행의 고유값
    - **Target** - the target is an ordinal variable indicating groups of income levels. 소득 수준을 나타내는 서수 변수
        - 1 = extreme poverty 극빈층
        - 2 = moderate poverty 중등도 빈곤
        - 3 = vulnerable households 취약 가구
        - 4 = non vulnerable households 비취약 가구
    - **idhogar** - this is a unique identifier for each household. This can be used to create household-wide features, etc. All rows in a given household will have a matching value for this identifier.
    - **parentesco1** - indicates if this person is the head of the household. 세대주
2. 노트북 특징
    - Feature engineering에 집중하고 shap를 사용해 중요한 피쳐만 남김
    

## 0. Import library

## 1. Check datasets

1.  ****Read dataset****
    - read_csv
    - print .shape
    - train / test .head()
2. ****Make description df****
3. ****Check null data****
    - **`df_train.isnull().sum()`**을 통해 train 데이터셋에서 결측치가 몇 개 있는지를 세고, **`sort_values(ascending=False)`**를 통해 결측치 수가 많은 feature부터 정렬
    - **`df_train.isnull().count()`**로 전체 데이터 수를 구하고, 결측치 수를 이 값으로 나누어 각 feature별 결측치 비율을 계산 **`100 *`**을 통해 비율을 퍼센트로 변환한 후, **`sort_values(ascending=False)`**로 결측치 비율이 높은 feature부터 정렬
    - 마지막으로, **`pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])`**을 사용하여 각 feature별 결측치 수와 비율을 하나의 데이터프레임으로 합치고**`keys=['Total', 'Percent']`**로 각 컬럼의 이름을 지정함.**`head(20)`**을 사용하여 상위 20개 feature만 출력
4. ****Filll missing values****
    - edjefa & edjfe
        - `edjefa`와 `edjefe` 변수에 대해, 해당 가구의 가장 높은 학력을 `yes`로 표기되어 있는 경우 해당 가구 대표자의 학력으로 대체
        - `edjefa`와 `edjefe` 변수의 값이 `yes`인 경우, 해당 변수의 의미를 명확히 알 수 없으므로, 이를 4로 채움
        - `edjef` 변수를 생성합니다. 이 변수는`edjefa`와 `edjefe`변수 중 더 높은 학력을 가진 대표자의 학력을 나타냄
        - `v14a`와 `sanitario1` 변수에 대해 일부 행에서 화장실이 있음에도 불구하고 `sanitario1`이 0으로 표시되어 있거나, 상수 수돗물이 없음에도 불구하고 `v14a`1로 표시되어 있는 경우, 이를 수정하여 일관성을 유지함
    - ****rez_esz, SQBmeaned****
    - ****meaneduc****
    - ****v18q1****

## 2. Feature engineering

1. ****Object features****
    - dependecy
    - ****edjefe****
    - ****edjefa****
    - ****roof and electricity****
2. ****Extract cat features****
3. ****Make new features using continuous feature****
    - ****Squared features****
    - ****Family features****
    - ****Rent per family features****
    - ****Room per family features****
    - ****BedRoom per family features****
    - ****Tabulet per family features****
    - ****phone per family features****
    - ****rez_esc(Years behind in school) per family features****
    - ****Rich features****
    - ****Remove feature with only one value****
    - ****Check whether both train and test have same features****
4. ****aggregation features****
    - ****Aggregation for family features****

## 3. Feature selection using shap

- SHAP?
    - ****Shapley Additive exPlanations****
        
        [[개념정리]SHAP(Shapley Additive exPlanations)](https://velog.io/@sjinu/개념정리SHAPShapley-Additive-exPlanations)
        
    - 로이드 섀플리가 만든 이론 위에 피처 간 독립성을 근거로 덧셈이 가능하게 넓힌 기법
- 정의된 함수 및 코드해석
    - `def evaluate_macroF1_lgb(truth, predictions):`
        - `truth`와 `predictions` 두 개의 입력 인자를 받음
        - 먼저 `predictions`배열을 `truth`에 있는 고유한 값의 수와 같은 행 수를 가지는 행렬로 재구성하고, 열 수는`predictions`의 길이를 `truth`에 있는 고유한 값의 수로 나눔
        - 그런 다음, 행렬의 각 행에서 가장 큰 값을 가진 인덱스를 찾아서 예측된 레이블을 가져옴
        - 마지막으로, scikit-learn의 `f1_score` 함수를 사용하여 `truth`와 예측된 레이블 사이의 매크로 F1 점수를 계산하고, 점수 이름인 `macroF1`, 계산된 F1 점수, 그리고 높은 점수가 더 좋은지를나타내는 부울 플래그가 있는 튜플을 반환
    - `def print_execution_time(start):`
        - `print_execution_time`이라는 함수로, 시작 시간을 나타내는 `start` 값을 입력 인자로 받음
        - 함수는 현재 시간을 측정하여 `end` 변수에 저장하고, `start`와 `end` 시간의 차이를 계산함
        - 이 시간 차이를 시, 분, 초로 변환하여 `Execution ended in`과 함께 출력함
        - 출력된 결과 : `Execution ended in [시간]h [분]m [초]s`
        - 이 때, 시간, 분, 초는 0으로 채워진 2자리 숫자로 출력되며, 시간과 분 사이, 분과 초 사이에는 구분자로 `:` 대신 `h`와 `m`이 사용됨
        - 마지막으로, 출력된 결과를 꾸며주기 위해 `*` 문자가 앞뒤로 20개씩 삽입됨
    - `def extract_good_features_using_shap_LGB(params, SEED):`
        
        SHAP 값을 사용하여 LGBM 모델에서 좋은 특성을 추출하는 함수
        
        - **`extract_good_features_using_shap_LGB`** 함수는 **`params`**와 **`SEED`** 두 개의 입력 인자를 받음
        - **`params`**는 LGBM 모델의 하이퍼파라미터를 포함하는 딕셔너리이고, **`SEED`**는 재현성을 위한 시드 값
        - 함수 내에서는 LGBMClassifier 모델을 초기화하고, **`StratifiedKFold`**를 사용하여 5-겹 교차 검증을 수행함  그리고 각 폴드에서 모델을 학습하고 평가함
        - **`fit`** 메서드를 호출할 때 **`eval_set`** 인자를 사용하여 검증 세트를 지정하고, **`evaluate_macroF1_lgb`** 함수를 평가 메트릭으로 사용함
        - SHAP 값을 계산하여 **`feat_importance_df`** 데이터프레임에 저장하고, 각 특성의 SHAP 값과 LGBM 모델의 특성 중요도를 **`fold_importance_df`** 데이터프레임에 저장
        - 이러한 과정을 거쳐 모든 폴드에서 추출된 특성 중요도 및 SHAP 값을 계산하여, 이를 평균 내고 **`shap_values`**에 대한 내림차순으로 정렬하여 반환하는 **`feat_importance_df_shap`** 데이터프레임을 생성
        - 현재 코드에서는 **`feat_importance_df_shap`**를 반환하고, 주석 처리된 코드는 SHAP 값의 누적 합이 0.999 이하가 되는 특성만 추출함
    - 50개의 랜덤 params 생성
        
        LightGBM 모델에서 SHAP 값을 이용하여 feature importance를 추출하는 작업을 여러번 반복하며 각 iteration 별로 추출된 feature importance 결과를 합쳐 최종적으로 모든 iteration에 대한 feature importance를 구하는 작업
        
        1. **`total_shap_df = pd.DataFrame()`** : 모든 iteration에 대한 feature importance 결과를 저장할 DataFrame 생성
        2. **`NUM_ITERATIONS = 50`** : 50번의 iteration을 진행할 것이라는 변수를 생성
        3. **`for SEED in range(NUM_ITERATIONS):`** : 50번의 iteration을 반복
        4. **`params = {...}`** : LightGBM 모델의 하이퍼파라미터 값들을 랜덤하게 설정하여 딕셔너리 형태로 생성
            - **`max_depth`**: 결정 트리의 최대 깊이를 의미하며 이 값이 크면 모델은 복잡해지고 과적합될 가능성이 높아짐
            - **`learning_rate`**: 경사 하강법에서 학습 속도를 결정하는 파라미터로 이 값이 작으면 모델이 더 많은 학습 단계를 거치게 되므로 더욱 정확한 결과를 얻을 수 있지만, 학습 시간이 오래 걸림
            - **`colsample_bytree`**: 결정 트리 생성 시 열 샘플링 비율을 결정함 이 값이 작으면 모델이 덜 복잡해지고 과적합될 가능성이 줄어듦
            - **`subsample`**: 훈련 데이터에서 행 샘플링 비율을 결정함. 이 값이 작으면 모델이 덜 복잡해지고 과적합될 가능성이 줄어듦
            - **`min_split_gain`**: 분기를 결정하는 데 필요한 최소 이득 값을 결정함. 이 값이 작으면 모델이 더 복잡해지고 과적합될 가능성이 높아짐
            - **`num_leaves`**: 각 결정 트리의 최대 잎 노드 수를 결정함 이 값이 크면 모델이 더 복잡해지고 과적합될 가능성이 높아짐
            - **`reg_alpha`**: L1 정규화 파라미터를 결정함 이 값이 크면 모델이 덜 복잡해지고 과적합될 가능성이 줄어듦
            - **`reg_lambda`**: L2 정규화 파라미터를 결정함. 이 값이 크면 모델이 덜 복잡해지고 과적합될 가능성이 줄어듦
            - **`bagging_freq`**: 부스팅에서 배깅을 수행하는 빈도를 결정하며 이 값이 크면 모델이 더 복잡해지고 과적합될 가능성이 높아짐
            - **`min_child_weight`**: 잎 노드에서 필요한 최소 가중치를 결정하며 이 값이 크면 모델이 덜 복잡해지고 과적합될 가능성이 줄어듦
        5. **`temp_shap_df = extract_good_features_using_shap_LGB(params, SEED)`** : 이전에 정의한 **`extract_good_features_using_shap_LGB()`** 함수를 이용하여 feature importance를 추출하고, 각 iteration 별로 추출된 feature importance 결과를 저장할 DataFrame 생성
        6. **`total_shap_df = pd.concat([total_shap_df, temp_shap_df])`** : 현재 iteration에서 추출한 feature importance 결과를 이전 iteration에서 추출한 결과와 합쳐서 저장하는 작업을 수행
        
        이렇게 50번의 iteration을 모두 완료하면 **`total_shap_df`** DataFrame에는 각 feature들의 평균 SHAP값과 중요도(feat_imp) 등이 저장되어 있으며, 이를 바탕으로 feature selection 등의 후처리 작업을 진행할 수 있음
        

## 4. Model development

- 사용모델 LGBM
- 흐름
    - 모델 함수 정의
    - 학습
    - feature_importance 시각화
    - submission 생성
    - 하이퍼 파라미터 튜닝_Randomized serach
- GridSearch? Randomized serach?
    - `GridSearch` 검증하고 싶은 하이퍼파라미터들의 수치를 정해주고 그 조합을 모두 검증.
    - `**Randomized serach`** 검증하려는 하이퍼파라미터들의 값 범위를 지정해주면 **무작위로 값을 지정**
    해 그 조합을 모두 검증.
- 정의된 함수 및 코드해석
    - `def LGB_OOF(params, categorical_feats, N_FOLDs, SEED=1989)`
        
        LightGBM 모델의 Out-of-Fold(OOF) 방식으로 학습을 수행하고, 학습한 모델로 테스트 데이터에 대한 예측 결과를 반환하는 함수
        
        - **`params`**는 LightGBM 모델의 하이퍼파라미터 값들을 포함하는 딕셔너리
        - **`categorical_feats`**는 범주형 변수의 열 이름을 포함하는 리스트.
        - **`N_FOLDs`**는 OOF 학습을 위한 교차 검증을 수행하는 폴드 수를 나타냄
        - **`SEED`**는 난수 생성을 위한 시드 값으로 기본값은 1989
        - **`clf`**는 LightGBMClassifier 모델 객체를 생성하고 이때 **`params`**로 전달된 하이퍼파라미터 값들을 모델에 설정하며, **`num_class`**는 종속변수의 클래스 수를 나타냄
        - OOF 학습을 위해 **`kf`**는 StratifiedKFold 객체로 생성되며, 폴드 수는 **`N_FOLDs`**로 설정됨
        - **`predicts_result`** 리스트는 각 폴드에서 테스트 데이터에 대한 예측 결과를 저장하는 리스트
        - **`feat_importance_df`**는 각 폴드에서 생성된 SHAP(Shapley Additive Explanations) 값을 기반으로 각 변수의 중요도와 SHAP 값을 저장하는 데이터프레임
        - 반복문에서 각 폴드를 순회하면서 다음을 수행
            - **`train_index`**와 **`test_index`**를 사용하여 학습 데이터와 검증 데이터를 나누고 LightGBM 모델 학습시작
            - **`eval_set`**은 모델 평가를 위한 데이터를 지정
            - **`early_stopping_rounds`**는 조기 중단을 위한 라운드 수를 설정합니다.
            - 학습한 모델에서 SHAP 값을 계산,  SHAP 값은 변수가 모델 예측에 미치는 영향력을 나타내며, 모델의 설명력을 높이는 데 사용됨
            - 변수의 중요도와 SHAP 값을 데이터프레임에 저장
            - 테스트 데이터에 대한 예측 결과를 **`predicts_result`** 리스트에 추가
        - OOF 방식으로 학습한 모델에서 반환된 **`predicts_result`**는 각 폴드에서 예측한 테스트 데이터의 클래스 레이블을 저장하는 리스트입니다. 이를 이용하여 예측 결과를 앙상블하거나 다양한 모델 평가 지표를 계산할 수 있습니다.
    - params =
        
        
    - clf =
        1. lgb.LGBMClassifier 함수를 사용하여 LightGBM 모델을 생성
        2. `objective` 파라미터에는 분류 문제의 경우 `multiclass`를 지정
        3. `max_depth` 파라미터는 트리의 최대 깊이를 나타냄
        4. `learning_rate` 파라미터는 학습 속도를 나타냅니다. 이 값이 작을수록 더욱 안정적인 학습이 가능하지만, 학습 시간이 길어질 수 있음
        5. `silent` 파라미터는 학습과정에서 출력되는 로그 메시지를 제어합니다. True로 설정하면 출력되지 않음
        6. `metric` 파라미터는 모델 성능 평가를 위한 지표를 나타냅니다. 이 코드에서는 'multi_logloss'를 사용함
        7. `n_jobs` 파라미터는 모델 학습에 사용되는 CPU 코어의 개수를 제어합니다. -1로 설정하면 모든 코어를 사용함
        8. `n_estimators` 파라미터는 부스팅 반복 횟수를 나타냄
        9. `class_weight` 파라미터는 클래스 불균형을 고려한 가중치를 지정
        10. `colsample_bytree` 파라미터는 각 트리에서 사용될 feature의 비율을 지정
        11. `min_split_gain` 파라미터는 분할을 수행할 때, 최소한으로 필요한 손실 감소량을 나타냄
        12. `bagging_freq` 파라미터는 샘플링하는 빈도를 제어
        13. `min_child_weight` 파라미터는 최소한으로 필요한 자식 노드의 샘플 개수를 나타냄
        14. `num_leaves` 파라미터는 트리의 최대 leaf 개수를 나타냄
        15. `subsample` 파라미터는 각 반복에서 사용될 샘플 비율을 나타냄
        16. `reg_alpha` 파라미터와 `reg_lambda` 파라미터는 L1 정규화와 L2 정규화를 나타냄
        17. `num_class` 파라미터는 분류 문제에서 클래스의 개수를 나타냄
        18. `bagging_seed`와 seed 파라미터는 랜덤 시드를 지정
    - ****Randomized serach****
        
        하이퍼파라미터 튜닝을 위한 랜덤 서치 
        
        1. optimized_param과 lowest_cv 변수를 초기화합니다. optimized_param은 튜닝된 하이퍼파라미터 값을 저장하고, lowest_cv는 해당 값에 대한 검증 세트의 최소 logloss 값을 저장
        2. total_iteration 변수만큼 반복하여 하이퍼파라미터를 랜덤하게 생성하고, LightGBM 모델을 학습 및 검증
        3. 생성된 하이퍼파라미터를 params 변수에 저장합니다. 이때, 'application', 'metric', 'num_class' 등 모델 학습에 필요한 파라미터를 지정합니다.
        4. lgb.Dataset 함수를 사용하여 학습 데이터와 레이블을 저장
        5. lgb.cv 함수를 사용하여 LightGBM 모델을 교차 검증합니다. 교차 검증을 수행할 때, 위에서 저장한 학습 데이터와 레이블, 생성된 하이퍼파라미터, 그리고 교차 검증을 위한 파라미터(nfold, shuffle 등)를 지징
        6. cv_results 변수에서 'multi_logloss-mean' 값들 중 최솟값을 서칭
        7. 최솟값이 현재까지의 최소값(lowest_cv)보다 작은 경우, optimized_param을 현재 파라미터 값으로 갱신
        8. 최종적으로 optimized_param 변수에는 최적의 하이퍼파라미터 값이 저장
