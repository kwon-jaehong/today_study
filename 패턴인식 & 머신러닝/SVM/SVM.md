support vector machines

--------------
선행 지식

퍼셉트론
최적화 이론
미적분


----------------------------
https://hleecaster.com/ml-svm-concept/
https://www.youtube.com/watch?v=Lpr__X8zuE8
----------------------------------
https://www.youtube.com/watch?v=qFg8cDnqYCI

https://www.youtube.com/watch?v=eZtrD6pYaaE

서포트 벡터 머신(이하 SVM)은 결정 경계(Decision Boundary), 즉 분류를 위한 기준 선을 정의하는 모델이다. 지도학습
-> 분류 알고리즘, 바이너리(2가지) classification, 기본적으로(오리지날 SVM은) 선형모델임


고차원 데이터를 linear separation(선형 분리)를 실시한다
-> 2차원에서는 선으로, 3차원에서는 면으로, 4차원 이상은 hyperplan
-> d-dim에서 두범주를 잘 구분하는 d-1차원의 hyperplane을 찾자라는것임

margin이 클수록 capacity term이 작아진다


초평면에 대한 개념은 
-> https://www.youtube.com/watch?v=JW2BsQZoqpw&t=152s 
5분 20초


기존에 어떤 모델로 분류를 하게되면, 트레이닝 데이터에대해 완벽하게 핏팅이된다 - 오버핏팅 ( 실제 데이터에 대해서는 좋지 않음 )
일반화 능력과 트레이닝 핏팅과는 trade-off 관계이다.
- 기존 모델에 training 에러를 너무 줄이게 되면 generalization ability (일반화 능력)이 떨어진다.



 
이에 svm은 트레이닝 에러를 줄여도, 일반화 능력을 뛰어난 모델을  svm은 statistical learning (통계학적 학습)에 이론을 근거로 하였기때문에 성능이 뛰어나다
----------------------------------
데이터가 주어졌을때
$wT + b = x$를 찾자

선을 그을때 기준은? -> 마진(margin)이라는 개념
-> 마진을 최대화 하는 선을 찾자

margin은 각 클래스에서 가장 가까운 관측치 사이의 거리

![](../../%EC%9D%B4%EB%AF%B8%EC%A7%80/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D/Screenshot%20from%202022-08-26%2008-53-23.png)

--------------
https://www.youtube.com/watch?v=yKkzx9x2yug

서포트 벡터
n개의 속성을 가진 데이터에는 최소 n+1개의 서포트 벡터가 존재한다는 걸 알 수 있다.




--------------------------
구현코드
https://zephyrus1111.tistory.com/211


soft svm
완벽하게 나눌수 없으므로 에러를 허용하는 svm






















