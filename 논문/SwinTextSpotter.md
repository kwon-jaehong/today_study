
-------------------

아키텍쳐
    swin trans 1장
    FPN + 실제 코드 아웃풋 쉐이프 한장 
    ROI align 1장 가볍게
    RCNN (디텍터 2장)


-------------------


특징
->
마스크맵을 이용해 인식기에서 인식을 집중할 수 있도록 함

엔드투 엔드로 안하면?
크레프트 + CRNN 이라면?
글자영역을 자름 -> 넘김 -> 인식기에서 잡음까지 다봄
-> 어텐션 스코어가 없어서 그럼





SwinTextSpotter

초록
최근의 최첨단 방법은 일반적으로 단순히 백본을 공유하여 감지 및 인식을 통합하므로 두 작업 간의 기능 상호 작용을 직접 활용하지 않습니다.
'다이나믹 헤드'가 있는트렌스포머 인코더를 디텍터로 사용하여 
두 작업을 새로운 인식 변환 메커니즘으로 통합하여 인식 손실을 통해 텍스트 위치를 명
시적으로 안내합니다.

연구 소개
주로 OCR 전략은 2가지
추출기 + 인식기를 따로 개발해서 이어붙임
단점
1. error accumulation between these two tasks
imprecise detection result may heavily hinder the performance of text recognition.부정확한 감지 결과는 텍스트 인식 성능을 크게 저하시킬 수 있습니다.

2. separate optimization of the two tasks might not maximize the final performance of text spotting.두 작업의 개별 최적화는 텍스트 스포팅의 최종 성능을 최대화하지 못할 수 있습니다.

3. intensive memory consumption and low inference efficiency
집중적인 메모리 소비와 낮은 추론 효율성




end to end로 하는 방법이 있음
단점

첫째, 탐지가 단순히 입력 특징의 시각적 정보를 기반으로 하는 경우 탐지기는 배경 잡음에 의해 주의가 산만해지는 경향이 있으며 그림과 같이 일관되지 않은 탐지를 제안합니다.

둘째, 인식 손실이 감지기를 최적화하거나 인식기가 감지 기능을 활용 하지 않기 때문에 백본을 공유하여 감지와 인식 간의 상호 작용이 충분하지 않습니다.




우리는 제안한다SwinTextSpotter, 종단 간 학습 가능한 Transformer 기반 프레임워크로 텍스트 감지와 인식 간의 더 나은 시너지 효과를 향해 나아가고 있습니다.




-----------------------------


방법론

The overall architecture of SwinTextSpotter is presented in Figure 2, which consists of four components:
(1) a backbone based on Swin-Transformer [31]; 
(2) a querybased text detector; 
(3) a Recognition Conversion module to bridge the text detector and recognizer; 
and (4) an attentionbased recognizer.

SwinTextSpotter의 전체 아키텍처는 네 가지 구성 요소로 구성된 그림 2에 나와 있습니다.
(1) Swin-Transformer [31]에 기반한 백본;
(2) 쿼리 기반 텍스트 검출기;
(3) 텍스트 감지기와 인식기를 연결하는 인식 변환 모듈;
및 (4) 주의 기반 인식기.


As illustrated in the green arrows of Figure 2, in the first stage of detection, we first randomly initialize trainable parameters to be the boxes bbox0 and proposal features f-prop(0).
그림 2의 녹색 화살표에서 볼 수 있듯이 첫 번째 감지 단계에서 먼저 학습 가능한 매개변수를 무작위로 초기화하여 상자 bbox0 및 제안 기능 f-prop(0)이 되도록 합니다.

To make the proposal features contain global information, we use global average pooling to extract the image features and add them into f-prop(0).
proposal features에 전역 정보가 포함되도록 하기 위해 전역 평균 풀링을 사용하여 이미지 기능을 추출하고 f-prop(0)에 추가합니다.


We then extract the RoI features using bbox0. The RoI features and f-prop(0) are fed into the Transformer encoder with dynamic head.
그런 다음 bbox0을 사용하여 RoI 기능을 추출합니다. RoI 기능과 f-prop(0)은 동적 헤드가 있는 Transformer 인코더에 제공됩니다.

The output of the Transformer encoder is flattened and forms the proposal features f-prop(1) , which will be fed into the detection head to output the detection result.
Transformer 인코더의 출력은 flattened되고 proposal features-> f-prop(1) 을 형성하며, 이는 감지 헤드에 공급되어 감지 결과를 출력합니다.


The box(bbox(k−1)) and proposal feature f-prop(k−1) will serve as the input to later(k-th) stage of detection.
box(bbox(k−1))와 제안 특징 f-prop(k−1)은 검출의 나중(k번째) 단계에 대한 입력으로 사용됩니다.

The proposal feature f-prop(k) recurrently updates itself by fusing the RoI features with previous f-prop(k−1), which makes proposal features preserve the information from previous stages.

proposal feature f-prop(k)는 RoI features을 이전 f-prop(k-1)과 융합하여 반복적으로 자체 업데이트하므로 proposal feature이 이전 단계의 정보를 보존합니다.


We repeat such refinement for totally K stages, resembling the iterative structure in the query-based detector [4,13,45,68]. Such design allows more robust detection in sizes and aspect ratios [45]. More details of the detector is explained in Section 3-2.
쿼리 기반 검출기 [4,13,45,68]의 반복 구조와 유사한 완전히 K 단계에 대해 이러한 정제를 반복합니다.
이러한 디자인은 크기와 종횡비에서 보다 강력한 감지를 가능하게 합니다[45]. 검출기에 대한 자세한 내용은 섹션 3-2에 설명되어 있습니다.



Since the recognition stage (orange arrows) requires higher rate of resolution than detection, we use the final detection stage output box bboxK to obtain the RoI features whose resolution is four times as much as that in the detection stage.
인식 단계(주황색 화살표)는 탐지보다 더 높은 분해능을 요구하므로 최종 탐지 단계 출력 상자 bbox-K를 사용하여 탐지 단계의 해상도보다 4배 더 높은 RoI features을 얻습니다.


In order to keep the resolution of features consistent with the detector when fused with proposal features, we down-sample the RoI features to get three feature maps of descending sizes, denoting by {a1, a2, a3}. 
proposal feature과 융합할 때 features의 해상도를 검출기와 일치하도록 유지하기 위해 RoI 기능을 다운샘플링하여 {a1, a2, a3}으로 표시되는 내림차순 크기의 세 가지 기능 맵을 얻습니다.



Then we obtain detection features f-det by fusing the smallest a3 and the proposal features f-prop-K.
그런 다음 가장 작은 a3와 proposal features (f-prop-K)를 융합하여 탐지 특징 f-det를 얻습니다.


The detection features (f-det) in recognition stage contain all previous detection information. 
인식 단계의 감지 기능(f-det)에는 이전의 모든 감지 정보가 포함됩니다.


Finally the {a1, a2, a3} and the detection features(f-det) are sent into Recognition Conversion and recognizer for generating the recognition result. More details of Recognition Conversion and recognizer are explained in Section 3-3 and Section 3-4, respectively.
마지막으로 {a1, a2, a3} 및 감지 기능(f-det)은 인식 결과를 생성하기 위해 인식 변환 및 인식기로 전송됩니다. Recognition Conversion과 Recognizer에 대한 자세한 내용은 각각 3-3절과 3-4절에서 설명합니다.






3.1 딜레이트 swin-transformer

Vanilla convolutions operate locally at fixed size
바닐라 컨볼루션은 고정 크기에서 로컬로 작동합니다.

which causes low efficacy in connecting remote features
멀리 떨어진 피쳐들을 연결할 때 효율성이 낮습니다.

For text spotting, however, modeling the relationships between different texts is critical since scene texts from the same image share strong similarity, such as their backgrounds and text styles.
그러나 텍스트 스포팅의 경우 동일한 이미지의 장면 텍스트가 배경 및 텍스트 스타일과 같은 강한 유사성을 공유하므로 서로 다른 텍스트 간의 관계를 모델링하는 것이 중요합니다.


Considering the global modeling capability and computational efficiency, we choose Swin-Transformer [31] with a Feature Pyramid Network (FPN) [25] to build our backbone.
글로벌 모델링 기능과 계산 효율성을 고려하여 Swin-Transformer[31]와 FPN(Feature Pyramid Network)[25]을 선택하여 백본을 구축합니다.


Given the blanks existing between words in a line of text, the receptive field should be large enough to help distinguish whether adjacent texts belong to the same text line.
텍스트 줄에서 단어 사이에 공백이 있는 경우 수신 필드는 인접한 텍스트가 동일한 텍스트 줄에 속하는지 구별하는 데 도움이 될 만큼 충분히 커야 합니다


To achieve such receptive field, as illustrated in Figure 3, we incorporate two dilated convolution layers one vanilla convolution layer and one residual structure into the original Swin-Transformer, which also introduce the properties of CNN to Transformer.
이러한 수용 필드를 달성하기 위해 그림 3과 같이 두 개의 확장된 컨볼루션 레이어를 하나의 바닐라 컨볼루션 레이어와 하나의 잔여 구조를 원래 Swin-Transformer에 통합하고 CNN의 속성도 Transformer에 도입합니다.


3.2 쿼리 기반의 디텍터
https://lynnshin.tistory.com/56
https://arxiv.org/pdf/2011.12450v2.pdf

We use a query based detector to detect the text. Based on Sparse R-CNN, the query based detector is built on ISTR which treats detection as a set-prediction problem. 
쿼리 기반 감지기를 사용하여 텍스트를 감지합니다. Sparse R-CNN을 기반으로 하는 쿼리 기반 탐지기는 탐지를 집합 예측 문제로 취급하는 ISTR에 구축됩니다.

Our detector uses a set of learnable proposal boxes, alternative to replace massive candidates from the RPN and a set of learnable proposal features, representing highlevel semantic vectors of objects.
우리의 탐지기는 객체의 고수준 의미 벡터를 나타내는 학습 가능한 제안 상자 세트와 RPN의 방대한 후보를 대체하는 학습 가능한 제안 기능 세트를 사용합니다.

The detector is empirically designed to have six query stages.
With the Transformer encoder with dynamic head, latter stages can access the information in former stages stored in the proposal features. 
탐지기는 6개의 쿼리 단계를 갖도록 경험적으로 설계되었습니다.
동적 헤드가 있는 Transformer 인코더를 사용하면 후기 단계에서 제안 기능에 저장된 이전 단계의 정보에 액세스할 수 있습니다.

Through multiple stages of refinement, the detector can be applied to text at any scale.
여러 단계의 개선을 통해 감지기는 모든 규모의 텍스트에 적용할 수 있습니다.

The detection information in previous stages is therefore embedded into the convolutions. The convolutions conditioned on the previous proposal features is used to encode the RoI features.
따라서 이전 단계의 감지 정보가 컨볼루션에 포함됩니다. 이전 제안 기능에 대한 컨볼루션은 RoI 기능을 인코딩하는 데 사용됩니다.


The output features of the convolutions is fed into a linear projection layer to produce the f-prop for next stage.
The f-prop is subsequently fed the into prediction head to generate bbox-k and mask-k. 
컨볼루션의 출력 기능은 다음 단계를 위한 f-prop을 생성하기 위해 선형 투영 레이어에 입력됩니다.
f-prop는 bbox-k 및 mask-k를 생성하기 위해 예측 헤드에 연속적으로 공급됩니다.


To reduce computation, the 2D mask is transformed into 1D mask vector by the Principal Component Analysis  so the mask-k is a one-dimensional vector
계산을 줄이기 위해 2D 마스크는 주성분 분석에 의해 1D 마스크 벡터로 변환되므로 mask-k는 1차원 벡터입니다.


When k = 1, the bbox0 and f-prop are randomly initialized parameters, which is the input of the first stage.
k = 1일 때 bbox0과 f-prop은 무작위로 초기화된 매개변수이며 첫 번째 단계의 입력입니다.


During training, these parameters are updated via back propagation and learn the inductive bias of the high-level semantic features of text.
훈련 중에 이러한 매개변수는 역전파를 통해 업데이트되고 텍스트의 고수준 의미론적 특징의 귀납적 편향을 학습합니다.


We view the text detection task as a set-prediction problem. Formally, we use the bipartite match to match the predictions and ground truths. The matching cost becomes:.
우리는 텍스트 감지 작업을 예측 집합 문제로 봅니다. 공식적으로 우리는 예측과 진실을 일치시키기 위해 bipartite match를 사용합니다. 일치 비용은 다음과 같습니다.

https://deep-learning-study.tistory.com/748


The losses for regressing the bounding boxes are L1 loss LL1 and generalized IoU loss Lgiou.
경계 상자를 회귀하기 위한 손실은 L1 손실 LL1 및 일반화된 IoU 손실 Lgiou입니다.


We compute the mask loss Lmask following [13], which calculates the cosine similarity between the prediction mask and ground truth.
예측 마스크와 정답 사이의 코사인 유사도를 계산하는 [13]에 따라 마스크 손실 Lmask를 계산합니다.


The detection loss is similar to the matching cost but we use the L2 loss and dice loss [33] to replace the cosine similarity as in [13].
검출 손실은 매칭 비용과 유사하지만 [13]에서와 같이 코사인 유사도를 대체하기 위해 L2 손실과 주사위 손실 [33]을 사용합니다.




3.3. Recognition Conversion (인식 변환) -> 인식기에 들어가기전 디텍션 정보와 이미지 피쳐정보 융합

To better coordinate the detection and recognition, we propose Recognition Conversion (RC) to spatially inject the features from detection head into the recognition stage, as illustrated in Figure 5.
감지 및 인식을 보다 잘 조정하기 위해 그림 5와 같이 감지 헤드에서 인식 단계로 기능을 공간적으로 주입하는 인식 변환(RC)을 제안합니다.


The RC consists of the Transformer encoder [48] and four up-sampling structures.
RC는 Transformer 인코더[48]와 4개의 업샘플링 구조로 구성됩니다.


The input of RC are the f-det and three down-sampling features {a1, a2, a3}.
RC의 입력은 f-det와 3개의 다운샘플링 특성 {a1, a2, a3}입니다.


Then through a stack of upsampling operation Eu() and Sigmoid function φ(), three masks {M1, M2, M3} for text regions are generated:
그런 다음 업샘플링 연산 Eu() 및 Sigmoid 함수 φ()의 스택을 통해 텍스트 영역에 대한 세 개의 마스크{M1, M2, M3}가 생성됩니다.


With the masks {M1, M2, M3} and the input features {a1, a2, a3}, we further integrate these features effectively under the following pipeline:
마스크 {M1, M2, M3} 및 입력 기능 {a1, a2, a3}을 사용하여 다음 파이프라인에서 이러한 기능을 효과적으로 통합합니다.

![](../%EC%9D%B4%EB%AF%B8%EC%A7%80/%EB%85%BC%EB%AC%B8/SwinTextSpotter/Screenshot%20from%202022-08-09%2015-53-29.png)

마스크 맵은 디텍션 피쳐를 시그모드한곳에서 생성됨 -> 확률값으로 변환
-> 확률을 마스크맵을 곱해준다? -> 그정보만 살리겟다



where {r1, r2, r3} denote the recognition features.
The r3 is the fused features in Figure 5, which is finally sent to the recognizer at the highest resolution.
여기서 {r1, r2, r3}은 인식 기능을 나타냅니다.
r3은 그림 5의 융합된 기능으로, 최종적으로 최고 해상도로 인식기로 전송됩니다.



As shown in the blue dashed lines in Figure 5, the gradient of the recognition loss Lreg can be back-propagated to the detection features, enabling RC to implicitly improve the detection head through the recognition supervision.
그림 5의 파란색 점선에서 볼 수 있듯이 인식 손실 Lreg의 기울기는 감지 기능으로 역전파되어 RC가 인식 감독을 통해 감지 헤드를 암시적으로 개선할 수 있습니다.



Generally, to suppress the background, the fused features will be multiplied by a maskK predicted by detection head (with the supervision of Lmask).
일반적으로 배경을 억제하기 위해 융합된 기능은 감지 헤드(Lmask의 감독하에)에 의해 예측된 maskK로 곱해집니다.


However, the background noise still remains in the feature maps as the detection box is not tight enough.
Such issue can be alleviated by the proposed RC since RC uses the detection features to generate tight masks to suppress the background noise, which is supervised by the recognition loss apart from the detection loss. 
그러나 감지 상자가 충분히 조밀하지 않기 때문에 배경 잡음은 여전히 기능 맵에 남아 있습니다.
이러한 문제는 RC가 감지 기능을 사용하여 배경 노이즈를 억제하기 위해 타이트한 마스크를 생성하기 때문에 제안된 RC에 의해 완화될 수 있으며, 이는 감지 손실과 별개로 인식 손실에 의해 감독됩니다.


As shown in the upper right corner of Figure 5, M3 suppresses more background noise than maskK, where M3 has higher activation in texts region and lower in the background.
그림 5의 오른쪽 상단 모서리에서 볼 수 있듯이 M3는 maskK보다 더 많은 배경 잡음을 억제합니다. 여기서 M3는 텍스트 영역에서 활성화가 더 높고 배경에서 더 낮습니다.


Therefore the masks {M1, M2, M3} produced by RC, which will be applied to the recognition features {r1, r2, r3}, makes recognizer easier to concentrate on the text regions.
따라서 인식 특성 {r1, r2, r3}에 적용될 RC에 의해 생성된 마스크 {M1, M2, M3}은 인식기가 텍스트 영역에 더 쉽게 집중할 수 있도록 합니다.



With RC, the gradient of recognition loss not only flows back to the backbone network, but also to the proposal features.
따라서 인식 특성 {r1, r2, r3}에 적용될 RC에 의해 생성된 마스크 {M1, M2, M3}은 인식기가 텍스트 영역에 더 쉽게 집중할 수 있도록 합니다.

Optimized by both detection supervision and recognition supervision, the proposal features can better encode the high-level semantic information of the texts. Therefore, the proposed RC can incentivize the coordination between detection and recognition.
탐지 감독과 인식 감독 모두에 의해 최적화된 제안 기능은 텍스트의 높은 수준의 의미 정보를 더 잘 인코딩할 수 있습니다. 따라서 제안된 RC는 탐지와 인식 간의 조정을 장려할 수 있습니다.




3.4. Recognizer

After applying RC on the feature map, background noise is effectively suppressed and thus the text regions can be bounded more precisely.
피처 맵에 RC를 적용하면 배경 노이즈가 효과적으로 억제되어 텍스트 영역을 보다 정확하게 경계할 수 있습니다.

This enables us to merely use a sequential recognition network to obtain promising recognition results without rectification modules such as TPS [3], RoISlide [8], Bezier-Align [28] or character-level segmentation branch used in MaskTextSpotter [21].
이를 통해 TPS[3], RoISlide[8], Bezier-Align[28] 또는 MaskTextSpotter[21]에서 사용되는 문자 수준 분할 분기와 같은 수정 모듈 없이 순차 인식 네트워크를 사용하여 유망한 인식 결과를 얻을 수 있습니다.


To enhance the fine-grained feature extraction and sequence modeling, we adopt a bi-level self-attention mechanism, inspired by [61], as the recognition encoder.
세분화된 특징 추출 및 시퀀스 모델링을 향상시키기 위해 인식 인코더로 [61]에서 영감을 받은 이중 수준 self-attention 메커니즘을 채택합니다.


The two-level self-attention mechanism (TLSAM) contains both finegrained and coarse-grained self-attention mechanisms for local neighborhood regions and global regions, respectively.
TLSAM(two-level self-attention mechanism)에는 로컬 이웃 지역 및 글로벌 지역에 대한 세분화된 자체 주의 메커니즘과 대략적 자체 주의 메커니즘이 모두 포함되어 있습니다.


Therefore, it can effectively extract fine-grained features while maintaining global modeling capability. As for the decoder, we simply follow MaskTextSpotter by using the Spatial Attention Module (SAM) [22]. The recognition loss is as follow:
따라서 글로벌 모델링 기능을 유지하면서 세분화된 기능을 효과적으로 추출할 수 있습니다. 디코더의 경우 SAM(Spatial Attention Module)을 사용하여 MaskTextSpotter를 따르기만 하면 됩니다[22]. 인식손실은 다음과 같습니다.


wherein T is the max length of the sequence and p(yi) is the probability of sequence.
여기서 T는 시퀀스의 최대 길이이고 p(yi)는 시퀀스의 확률입니다.














잡다한 지식
------------------------------------------
https://www.youtube.com/watch?v=GXYfQsj8RU0
ROI 풀링 vs ROI 정렬(align)
둘다 관심영역의 피쳐를 추출하기 위한작업이다
-> 왜? 관심영역은 각기 크기가 다르기 때문에 통일된 크기로 맞추어 줘야함

ROI 풀링은 관심영역을 그리드로 나누고, 그 그리드 안에서 맥스풀링

roi align은?
-> roi 풀링 과정에서 좌표 양자화를 할때 (소수점 떌떼), 오정렬이 생길수 있음
양자화 없이 k(그리드)로 나누고, bilinear interpolation(쌍선형보간법)으로 보간 하고 맥스풀링 실행
--------------------------------------------------
H-mean 계산 방법
https://neverabandon.tistory.com/60
https://sumniya.tistory.com/26

-------------------------------
ROI피쳐는 이미지에서 그영역에서 땡겨오는 피쳐


---------------

1.이미지 노말라이제이션

2.백본에 넣음 (swin 트랜스포머 + FPN)
인풋 : 이미지(노말라이제이션된)
아웃풋 : FPN 피쳐값

3. ROI 후보영역 좌표값 초기화
-> init_proposal_boxes 박스 좌표가 학습가능한 파라미터로 무작위 초기화
직접 데이터를 찍어보니 가운데에서 싹다 보는걸로 초기화


4.FPN 피쳐값 합침

------------------------------------------




























































