
픽쳐 피라미드 네트워크

다양한 스케일에 걸쳐서 특징을 추출



목적 : 상위레이어에서 추출한 정보를 하위레이어에 전달하면서, 해상도별 최적화된 피쳐를 뽑는것
예시) 첫번째 피쳐는 큰물체, 가면갈수록 작은물체를 보는데 최적화

![](../%EC%9D%B4%EB%AF%B8%EC%A7%80/%EB%85%BC%EB%AC%B8/FPN/Screenshot%20from%202022-08-09%2013-30-48.png)


-------------------

바텀업 패스웨이 : 각각의 이미지를 다운샘플링하면서 중간에 결과를 추출할 패스웨이 
! FPN의 백본이라고 보면됨

1x1 컨브, nearest neighbor를 통해 업샘플링


![](../%EC%9D%B4%EB%AF%B8%EC%A7%80/%EB%85%BC%EB%AC%B8/FPN/Screenshot%20from%202022-08-09%2013-37-01.png)
![](../%EC%9D%B4%EB%AF%B8%EC%A7%80/%EB%85%BC%EB%AC%B8/FPN/Screenshot%20from%202022-08-09%2013-38-40.png)

![](../%EC%9D%B4%EB%AF%B8%EC%A7%80/%EB%85%BC%EB%AC%B8/FPN/Screenshot%20from%202022-08-09%2013-38-40.png)
------------------------------



















