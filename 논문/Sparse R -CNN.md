https://www.youtube.com/watch?v=EvPMwKALqvs


https://junha1125.github.io/blog/artificial-intelligence/2021-04-27-Sparse-R-CNN/


-----------------------
특징
NMS(억제하지 않고)를 쓰지않음
원본이미지 전체의 피쳐맵을 사용하지 않음
------------------------------------
현재 나와있는 object detector들은 dense region proposal에 기반하고 있다라고 보면됨
(욜로,ssd,faster-RCNN)



proposal box
proposal 피쳐 20분 38초 implicit knowledge.



기존에 앵커박스를 잡고 하는게 dense한 방법이다
중간에 제안 ROI를 줄이는게  dense to sparse


! 중첩,많은수 오브젝트 디텍션, 작은물체 등 도메인에 따라 모델의 하이퍼 파라미터를 튜닝해야됨 

1.스파스한 박스
2.스파스한 피쳐

다시 그림
![](../%EC%9D%B4%EB%AF%B8%EC%A7%80/%EB%85%BC%EB%AC%B8/Sparse%20R%20-CNN/Screenshot%20from%202022-08-10%2010-50-30.png)






-------------------------
A fixed small set of learnable proposal boxes (N ×4) are used as region proposals, instead of the predictions from Region Proposal Network (RPN). These proposal boxes are represented by 4-d parameters ranging from 0 to 1, denoting normalized center coordinates, height and width. The parameters of proposal boxes will be updated with the back-propagation algorithm during training. Thanks to the learnable property, we find in our experiment that the effect of initialization is minimal, thus making the framework much more flexible.

지역 제안 네트워크(RPN)의 예측 대신 학습 가능한 제안 상자(N×4)의 고정된 작은 세트가 지역 제안으로 사용된다. 이러한 제안 상자는 정규화된 중심 좌표, 높이 및 폭을 나타내는 0부터 1까지의 4-d 매개변수로 표시됩니다. 제안 상자의 매개 변수는 교육 중에 후방 전파 알고리듬으로 업데이트됩니다. 학습 가능한 속성 덕분에, 우리는 실험에서 초기화의 효과가 미미하여 프레임워크를 훨씬 더 유연하게 만든다는 것을 발견했다.



Though the 4-d proposal box is a brief and explicit expression to describe objects, it provides a coarse localization of objects and a lot of informative details are lost, such as object pose and shape. Here we introduce another concept termed proposal feature (N ×d), it is a high-dimension (e.g., 256) latent vector and is expected to encode the rich instance characteristics. The number of proposal features is same as boxes, and we will discuss how to use it next.

4-d 제안 상자는 객체를 설명하는 간략하고 명시적인 표현이지만 객체의 대략적인 현지화를 제공하며 객체 포즈 및 모양과 같은 많은 정보 세부 정보가 손실된다. 여기서 제안 기능(N×d)이라는 또 다른 개념을 소개하는데, 이는 고차원(예: 256) 잠재 벡터이며 리치 인스턴스 특성을 인코딩할 것으로 예상된다. 제안 기능의 개수는 박스와 같으며, 사용 방법은 다음에 논의하도록 하겠습니다.













