https://www.youtube.com/watch?v=pJCcGK5omhE&t=1260s
-> 요약 정리한 내용 https://sanghyu.tistory.com/13?category=1122189


정규화 왜하나?
-> 오버핏팅 방지


-------------------
## 선행 지식
최소제곱법
--------------------
https://realblack0.github.io/2020/03/29/normalization-standardization-regularization.html
한국어로 셋다 정규화라고 해석됨.... 셋다 다른것임
-----------------
## 참고자료
https://wikidocs.net/60751
https://light-tree.tistory.com/125
https://gaussian37.github.io/dl-concept-regularization/

https://junklee.tistory.com/29
https://huidea.tistory.com/154

------------------------------------
## 파이토치 구현 
https://cding.tistory.com/109
https://androidkt.com/how-to-add-l1-l2-regularization-in-pytorch-loss-function/
----------------------



### Normalization 
값의 범위(scale)를 0~1 사이의 값으로 바꾸는 것
학습 전에 scaling하는 것
머신러닝에서 scale이 큰 feature의 영향이 비대해지는 것을 방지
딥러닝에서 Local Minima에 빠질 위험 감소(학습 속도 향상)


### Standardization
값의 범위(scale)를 평균 0, 분산 1이 되도록 변환
학습 전에 scaling하는 것
머신러닝에서 scale이 큰 feature의 영향이 비대해지는 것을 방지
딥러닝에서 Local Minima에 빠질 위험 감소(학습 속도 향상)
정규분포를 표준정규분포로 변환하는 것과 같음
Z-score(표준 점수)
-1 ~ 1 사이에 68%가 있고, -2 ~ 2 사이에 95%가 있고, -3 ~ 3 사이에 99%가 있음
-3 ~ 3의 범위를 벗어나면 outlier일 확률이 높음
표준화로 번역하기도 함


### Regularization
weight를 조정하는데 규제(제약)를 거는 기법
Overfitting을 막기위해 사용함
L1 regularization, L2 regularizaion 등의 종류가 있음
---------------------


------------------
## 좋은 모델이란?
-> 
1. 현재 데이터(training data)를 잘 설명하는 모델
2. 미래 데이터(testing data)에 대한 예측 성능이 좋은 모델

--------------------
조건 : 현재 데이터(training data)를 잘 설명하는 모델
-> 트레이닝 error를 잘 minimize 하는 모델
 
미래데이터에 대한 bias,variance 둘다 낮춰야함

----------
바이어스는 포기하더라도 variance는 어떻게 줄일수 있을까?

정규화 개념
-> 파라미터에 계수를 줌으로서 패널티 부여













