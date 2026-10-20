# 튜닝 확인 (설정은 검증 구간으로 선택, 테스트는 한 번)

| 모델 | 설정 | 검증 MAE | 테스트 MAE | 선택된 설정 |
|---|---|---|---|---|
| LightGBM(PM2.5만) | default | 8.340 | 5.129 | 기본 |
| LightGBM(PM2.5만) | tuned | 7.930 | 5.111 | {'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 7, 'min_child_samples': 160, 'subsample': 1.0, 'colsample_bytree': 0.9, 'reg_lambda': 1.0} |
| LightGBM(+기상) | default | 7.934 | 4.904 | 기본 |
| LightGBM(+기상) | tuned | 7.675 | 4.858 | {'n_estimators': 1000, 'learning_rate': 0.05, 'num_leaves': 7, 'min_child_samples': 160, 'subsample': 1.0, 'colsample_bytree': 0.9, 'reg_lambda': 1.0} |
| 1D-CNN+기상 | default | 7.324 | 4.550 | {'hidden': 64, 'drop': 0.1, 'lr': 0.002, 'wd': 0.0001} |
| 1D-CNN+기상 | tuned | 7.222 | 4.591 | {'hidden': 32, 'drop': 0.1, 'lr': 0.002, 'wd': 0.0001} |
| LSTM+기상 | default | 7.180 | 4.562 | {'hidden': 64, 'drop': 0.1, 'lr': 0.002, 'wd': 0.0001} |
| LSTM+기상 | tuned | 7.062 | 4.498 | {'hidden': 32, 'drop': 0.1, 'lr': 0.002, 'wd': 0.0001} |