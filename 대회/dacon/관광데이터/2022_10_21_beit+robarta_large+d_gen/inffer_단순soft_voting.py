import random
import torch
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
from d_set import CustomDataset,MyCollate
from torch.utils.data import DataLoader
from model import CustomModel
import pandas as pd
from transformers import  AutoTokenizer
import torch.nn.functional as F
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
CFG = {
    'IMG_SIZE':224,
    'BATCH_SIZE':32,
    'SEED':41,
    'NUM_WORKERS':4
}
dir_path = "/data/mrjaehong/handwriting_gen/today_study/대회/dacon/관광데이터/test.csv"
tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base",cache_dir="./temp")

ex_id = '0'
run_id = '8d70c3f025d248d4bcf473e17930a880'
fold_epoch_best_model = [[0,14],[1,18],[2,15],[3,15],[4,18]]


# ex_val_df_dir = os.path.join()




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(41) # Seed 고정


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
test_transform = A.Compose([
                        A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])


# tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large",cache_dir="./temp")

## 라벨링 cat3를 dict로 작업
df = pd.read_csv('../train.csv')
set_level_list_3 = list(set(list(df['cat3'])))
set_level_list_3.sort()            
label_info= {label:i for i,label in enumerate(set_level_list_3)}
num2label= {i:label for i,label in enumerate(set_level_list_3)}



## 데이터 로더
pad_token_id = tokenizer.pad_token_id


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv('../test.csv')
test_dataset = CustomDataset(test_df,dir_path,tokenizer,test_transform,label_info,infer=True)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id,infer=True))
submit = pd.read_csv('../sample_submission.csv')






total_logit_list = []
total_pred_softmax = []
for fold,epoch in fold_epoch_best_model:
    fold_pred_softmax_val = None
    model = CustomModel(tokenizer,len(label_info),0.5,0.15,17).to(device)
    model.load_state_dict(torch.load('./fold_'+str(fold)+'_epoch_'+str(epoch)+'_model.pth')) 

    model.eval()
    model_preds_correct = []
    with torch.no_grad():
        for data_batch in tqdm(test_loader,desc="fold_"+str(fold)+"_검증 중..."):

            img = data_batch['image']
            text = data_batch['text']
            mask = data_batch['mask']

            img = img.float().to(device)
            text = text.to(device)
            mask = mask.to(device)
            
            ## 모델을 통과해서 나온 로짓값
            model_pred = model(img, text,mask,device) 
            
            ## 이상하게... 모델 로직값 단순 더한게 제출 점수가 더 높음....            
            if str(type(fold_pred_softmax_val)) == "<class 'NoneType'>":
                fold_pred_softmax_val = F.softmax(model_pred.detach().cpu(), dim=1).detach().cpu().numpy()                
            else:
                fold_pred_softmax_val =  np.concatenate((fold_pred_softmax_val, F.softmax(model_pred.detach().cpu(), dim=1).detach().cpu().numpy()), axis=0)
            ## 폴드별 클래스 결과값(max)
            model_preds_correct += model_pred.argmax(1).detach().cpu().numpy().tolist()
        total_logit_list.append(fold_pred_softmax_val)
            
    
    
    result = []
    for j in model_preds_correct:
        result.append(num2label[j])
    submit['fold_'+str(fold)] = result




## 종합 결과값 추출, 각 폴드별 logit값에 mean해주고 argmax값 뽑음 => soft voting
result = []
## 로짓값(소프트 맥스 한거) 평균
mean_pred = np.mean(total_logit_list,0)
total_model_pred = np.argmax(mean_pred,1)
for j in total_model_pred:
    result.append(num2label[j])
submit["soft_voitng"] = result

submit.to_csv('./soft_voting.csv', index=False)

