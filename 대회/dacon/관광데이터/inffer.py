import random
import torch
import os
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from dataset import CustomDataset,MyCollate
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from model2 import CustomModel
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from kobert_tokenizer import KoBERTTokenizer
from torch.optim.lr_scheduler import LambdaLR
import logging as log
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, Sequence


# log.basicConfig(filename='./log2.txt', level=log.DEBUG)

# writer = SummaryWriter('runs/g2_experiment_1')
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        if len(target.shape) ==0:
            target = target.view(-1)
            
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return focal_loss
    
    
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':20,
    'LEARNING_RATE':2e-5,
    'BATCH_SIZE':6,
    'SEED':41,
    'TRAIN_RATE':0.9,
    'NUM_WORKERS':4
}
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(CFG['SEED']) # Seed 고정


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


train_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])
test_transform = A.Compose([
                            A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE']),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                            ])


tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

print("데이터셋 구성 중")
train_all_dataset = CustomDataset('./train.csv',tokenizer,train_transform)
label_info = train_all_dataset.label_3_level_2num
print("데이터셋 구성 완료")

dataset_size = len(train_all_dataset)
train_size = int(dataset_size * CFG['TRAIN_RATE'])
validation_size = dataset_size - train_size

## 데이터셋 나누기
train_dataset, validation_dataset = random_split(train_all_dataset, [train_size, validation_size])

## 데이터 로더
pad_token_id = tokenizer.pad_token_id
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id))
validation_loader = DataLoader(validation_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id))
    
    
    
model = CustomModel(tokenizer,len(train_all_dataset.label_1_level_2num),len(train_all_dataset.label_2_level_2num),len(label_info))
model.to(device)
model.load_state_dict(torch.load('./save_model/13g2.pth'))


    
test_dataset = CustomDataset('./test.csv',tokenizer,test_transform,infer=True)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], num_workers = CFG['NUM_WORKERS'],pin_memory=True, collate_fn = MyCollate(pad_idx=pad_token_id,infer=True))
    

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
        
        cat1_p,cat2_p,model_pred = model(img, text,mask,device) 
        model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()

result = []
for i in model_preds:
    result.append(train_all_dataset.num2label_3_level[i])


submit = pd.read_csv('./sample_submission.csv')
submit['cat3'] = result

submit.to_csv('./g2_best.csv', index=False)
