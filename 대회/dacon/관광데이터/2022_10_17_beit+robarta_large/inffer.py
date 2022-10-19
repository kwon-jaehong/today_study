import random
import torch
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from d_set import CustomDataset,MyCollate
from torch.utils.data import DataLoader
from model import CustomModel
import pandas as pd
from transformers import  AutoTokenizer

CFG = {
    'IMG_SIZE':224,
    'BATCH_SIZE':6,
    'SEED':41,
    'NUM_WORKERS':4
}


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
                        A.Resize(224,224),
                        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, always_apply=False, p=1.0),
                        ToTensorV2()
                        ])



tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base",cache_dir="./temp")
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
dir_path = "/data/mrjaehong/handwriting_gen/today_study/대회/dacon/관광데이터/test.csv"
test_df = pd.read_csv('../test.csv')
test_dataset = CustomDataset(test_df,dir_path,tokenizer,test_transform,label_info,infer=True)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id,infer=True))

submit = pd.read_csv('../sample_submission.csv')

total_logit_list = []
for i in range(0,5):
    fold_pred = None
    model = CustomModel(tokenizer,len(label_info),0.4,0.15,17).to(device)
    model.load_state_dict(torch.load('fold_'+str(i)+'_best_f1_model.pth')) # 결과값 0.85214
    model.eval()
    model_preds = []
    count = 0
    with torch.no_grad():
        for data_batch in test_loader:
            count  += len(data_batch['image'])

            img = data_batch['image']
            text = data_batch['text']
            mask = data_batch['mask']

            img = img.float().to(device)
            text = text.to(device)
            mask = mask.to(device)
            
            model_pred = model(img, text,mask,device) 
            if str(type(fold_pred)) == "<class 'NoneType'>":
                fold_pred = model_pred.detach().cpu().numpy()
            else:
                fold_pred =  np.concatenate((fold_pred, model_pred.detach().cpu().numpy()), axis=0)
            ## 폴드별 하드보팅용 결과값
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
        
        total_logit_list.append(fold_pred)
            
    
    result = []
    for j in model_preds:
        result.append(num2label[j])
    submit['fold_'+str(i)] = result


## logit값을 softmax해줘야함 단순히 더해주면 안됨
# 각 폴드별 모델 웨이트값을 softmax해준뒤 곱해준다!!!!
## 일단 소프트 보팅 1개랑 (이파일은 보존), 웨이트 소프트 맥스한거 버젼 하나더 만들어야함

## 종합 결과값 추출, 각 폴드별 logit값에 mean해주고 argmax값 뽑음 => soft voting
result = []
mean_pred = np.mean(total_logit_list,0)
total_model_pred = np.argmax(mean_pred,1)
for j in total_model_pred:
    result.append(num2label[j])
submit["soft_voitng"] = result



submit.to_csv('./temp_best2.csv', index=False)
