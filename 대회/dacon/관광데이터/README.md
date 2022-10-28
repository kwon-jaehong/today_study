일단 데이터가 극단적인 불균형임
- focal loss, 언더 샘플링, 
https://www.youtube.com/watch?v=CU2GF0du36o&t=1s



느낌점 : mlflow 에폭별 시간 체크 기능 넣자
mlflow 모델 저장공간을 좀더 콤팩트하게 하자 (실험관리요으로)

트랜스포머 구조는.... 폴드 같은거 쓰지 말자, 데이터가 적거나 불균형 할 때에는 
강력한 모델 하나가 나은것 같다 X (아니였음)
텍스트 번역은 넣어도 괜찮을듯 함

5폴드 , 베스트 f1 그냥 보팅시 가장 좋았음 -> 웨이트 필요없엇네
10폴드는 어케나오려나


TTA 기법 활용 못해본게 아쉬움
https://inspaceai.github.io/2019/12/20/Test_Time_Augmentation_Review/
테스트 데이터를 증강해서 여러 방면으로 해석하는것!


 
2022 10 24 기준
1등 0.866
나 0.859

7280×0.866 = 6304
7280×0.86 = 6260
약 40개 차이



-----------------------------

계층 분류
https://arxiv.org/pdf/1709.09890.pdf
https://github.com/AdicherlaVenkataSai/HCNN/blob/master/VGG19HCNN.ipynb
https://github.com/Ugenteraan/Deep_Hierarchical_Classification

단어 검색 hierarchical label classification pytorch
구글 학술 B-CNN 타고  hierarchical classification deep learning
B-CNN: branch convolutional neural network for hierarchical classification

paperswi : Hierarchical Multi-label Classification


  다중분류 
  https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/



----------------------------
트랜스 포머 구조 input token max length에 대한 대응 방법
-> split 하고 결과값을 mean 시킴 
https://www.youtube.com/watch?v=yDGo9z_RlnE
----------------------------

버트 파인튜닝 참고 사이트
https://mccormickml.com/2019/07/22/BERT-fine-tuning/#a1-saving--loading-fine-tuned-model







