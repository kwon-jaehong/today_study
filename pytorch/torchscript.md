
https://www.youtube.com/watch?v=Dk88zv1KYMI

torchscript
토치 스크립트란 무엇인가?
a statically-type subset of python meant for ml.
-> ml를 의미하는 파이썬의 정적 유형 하위 집합입니다.

meant for consumption by the pytorch just-in-time compiler(jit), which performs runtime optimization on your torch model.
-> pytorch just-in-time 컴파일러(jit)에서 사용하기 위한 것입니다. 토치 모델에서 런타임 최적화를 수행합니다.
-> just in time (적시에, 제 시간에)

the preferred method for serializing your trained model and deploying it for production inference.
-> 훈련된 모델을 직렬화하고 프로덕션 추론을 위해 배포하는 데 선호되는 방법입니다.
------------------------------

https://happy-jihye.github.io/dl/torch-2/
파이토치를 이용해 실험해 보면 알겠지만,

autograd 모드를 끈다
-> 원래 파이토치는 그래프를 생성시, 자동적으로 auto grad(자동 미분)을 지원한다
학습시 특정 모듈만 켜진다.
-> 드롭아웃 모듈은 학습시에만 켜짐
-> batchnorm도 트래이닝시에만 켜짐


Pytorch 는 Python의 특징을 많이 가지고 있는 프레임워크이다. 때문에 Portability와 Performance, 이 두가지 측면에서 약세를 보였고, 이를 해결하기 위해 Torchscript는 코드를 Eager mode에서 Script mode로 변환한다.
---------------------------------------

https://www.youtube.com/watch?v=NywbN1aKUXQ
일반 연구원들이 파이토치 모델을 실험&개발한 것을 (코드포함) 그대로 서비스환경으로 올릴수는 없음(할수있지만 매우 비효율적임)

이유
1. 멀티쓰레딩 지원 안됨 (GPU는 멀티스레딩이 지원안됨)
2. 추론이 모바일/cpu에서 거의 이루어질것임
3. C++을 사용할 경우 (파이썬으로 개발했는데...)

즉, 파이썬 + 파이토치에서 개발한 결과물&동작들을 다른 환경에서도 작동하게 하고싶음

직렬화란?
-> 메모리 그대로 저장장치에 저장하는것
단순히 생각하면 특정 정보(객체)등이 메모리상에 올라가 있는데 물리적으로 여기저기 분포되어 있거나 또는 메모리 특성에 맞게 data들이 관리 되고 있습니다. 바이트코드로 변환
https://juneyr.dev/2018-08-23/serialize 
그림참조

cuda는 멀티쓰레딩이 안됨(구조적으로 맞지 않음)


pytorch에서 연산 그래프를 그릴때(메모리 적재) 주로 파이썬의 메모리 담당기관에 대해 의존성이 있음
-> 파이썬에 의존하지않게 만들어야함

jit (just in time)
jit의 역활, 중간 언어로 코드를 바꿔줌
인터프리터 실행해줌



torchscript :  tracing
-> 트래이스된 모델은 2가지 정보를 담고 있다, 그래프 정보, code 정보
그래프를 역추적 하는 방식이라 if문이나 이런것들은 정보를 담을수 없다

torchscript : scripting
-> 파이토치 스크립트 자체를 토치 스크립트로 변환
-> 내가 느낀점은, 스크립트만 쓰는게 나을듯 하다
































