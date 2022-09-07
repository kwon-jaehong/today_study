
SVM,pca 등 구현체 
https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/svm.py


--------------------
https://realblack0.github.io/2020/03/29/normalization-standardization-regularization.html
한국어로 셋다 정규화라고 해석됨.... 셋다 다른것임


Normalization 
값의 범위(scale)를 0~1 사이의 값으로 바꾸는 것
학습 전에 scaling하는 것
머신러닝에서 scale이 큰 feature의 영향이 비대해지는 것을 방지
딥러닝에서 Local Minima에 빠질 위험 감소(학습 속도 향상)


Standardization
값의 범위(scale)를 평균 0, 분산 1이 되도록 변환
학습 전에 scaling하는 것
머신러닝에서 scale이 큰 feature의 영향이 비대해지는 것을 방지
딥러닝에서 Local Minima에 빠질 위험 감소(학습 속도 향상)
정규분포를 표준정규분포로 변환하는 것과 같음
Z-score(표준 점수)
-1 ~ 1 사이에 68%가 있고, -2 ~ 2 사이에 95%가 있고, -3 ~ 3 사이에 99%가 있음
-3 ~ 3의 범위를 벗어나면 outlier일 확률이 높음
표준화로 번역하기도 함


Regularization
weight를 조정하는데 규제(제약)를 거는 기법
Overfitting을 막기위해 사용함
L1 regularization, L2 regularizaion 등의 종류가 있음

---------------------
