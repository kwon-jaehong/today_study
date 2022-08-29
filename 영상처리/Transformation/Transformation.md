Transformation 




https://darkpgmr.tistory.com/79
https://blog.daum.net/shksjy/228

--------------
영상의 기하학적 변환





선형 변환(Linear Transformation)이란 벡터의 합과 스칼라 곱 성질을 만족하는 벡터 공간 사이의 함수이다.


rigid transformaition (강체 변환)
rigid 변환은 다른말로 유클리디언 변환이라고 함, 모양과 크기를 유지한체로 위치와,방향(rotation)만 바꿀수 있는 변환
즉, 회전과 평생이동(translation)만을 허용하는 변환


Projective Transformation(원근변환)


---------------------------
https://wingnim.tistory.com/94
https://zzzinho.tistory.com/12


affine 변환

2차원 좌표계를 한축을 더해 3차원 행렬로, 만들고 이동량, 회전
-> 선형변환과 이동변환까지 포함한 변환
-> 트랜슬레이션, 쉬어,스케일링,회전등을 통틀어 어파인 변환이라 한다.
-> 2d 기준 2x3행렬에 크기,회전, 이동정보를 넣어 연산
-> 6개의 


perspective transform(투시변환)은? 또는 projective transformation은?
-> 본질적으로, 한 평면을 다른 평면으로 매핑, 점을 통해 매핑

-> 3x3 행렬로 표현됨
점 4개의 이동관계를 알고있어야 


translation transformation -> 이동변환
영상의 가로 또는 세로방향으로 특정 크기만큼 이동시키는 변환

https://gaussian37.github.io/vision-concept-homogeneous_coordinate/

Homogeneous coordinate 동차 좌표계


shear transformation - 전단 변환 
-> 층 밀림 변환, x축과, y축 방향에 대해 따로 정의
-> 이동을 하긴하는데 변위가 다름
 
영상 리사이즈
양선형 보간 (2x2 이웃 픽셀 참조)
3차회선 보간 (4x4이웃 픽셀 참조)
lanczos 보간 (8x8 이웃 픽셀 참조)

-----------------
캐니에지
패캠 강의 듣기
- 단일에지를 가지도록 스텝3을 함


스텝 1 가우시안 필터링 (노이즈 제거)
스텝 2 그래디언트 계산 (주로 소벨 마스크 사용), 그래디언트 값과, 그레디언트 방향을 계산, 4구역으로 단순화
스텝 3 비최대 억제 (논 멕시마 서프레션)
스텝 4 이중 임계값을 이용 히스테리시스 에지 트래킹

하나의 에지에서 




































